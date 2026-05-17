using System.Collections.Immutable;
using System.Text;
using Google.Protobuf;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Onnx;

namespace Onnxify.ModelGenerator;

/// <summary>
/// Generates strongly typed <c>Microsoft.ML.OnnxRuntime</c> wrappers for ONNX models included in a consuming project.
/// </summary>
[Generator]
public sealed class OnnxModelGenerator : IIncrementalGenerator
{
    private const string GLOBAL_PROJECT_DIRECTORY_KEY = "build_property.MSBuildProjectDirectory";
    private const string GLOBAL_PROJECT_DIR_KEY = "build_property.ProjectDir";
    private const string GLOBAL_ROOT_NAMESPACE_KEY = "build_property.RootNamespace";
    private const string GLOBAL_ASSEMBLY_NAME_KEY = "build_property.AssemblyName";
    private const string ADDITIONAL_FILE_CLASS_NAME_KEY = "build_metadata.AdditionalFiles.OnnxifyModelClassName";
    private const string ADDITIONAL_FILE_NAMESPACE_KEY = "build_metadata.AdditionalFiles.OnnxifyModelNamespace";

    private static readonly DiagnosticDescriptor _invalidModelDescriptor = new(
        id: "OMG001",
        title: "Unable to read ONNX model",
        messageFormat: "Unable to load ONNX model '{0}': {1}",
        category: "Onnxify.ModelGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true
    );

    private static readonly DiagnosticDescriptor _unsupportedTensorDescriptor = new(
        id: "OMG002",
        title: "Unsupported ONNX model input or output",
        messageFormat: "Model '{0}' uses unsupported {1} '{2}' with ONNX type '{3}'. Only tensor inputs and outputs backed by Microsoft.ML.OnnxRuntime-compatible element types are supported.",
        category: "Onnxify.ModelGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true
    );

    private static readonly DiagnosticDescriptor _duplicateTypeDescriptor = new(
        id: "OMG003",
        title: "Duplicate generated model type",
        messageFormat: "Multiple ONNX model inputs would generate the same wrapper type '{0}'. Override at least one file with the metadata OnnxifyModelClassName or OnnxifyModelNamespace.",
        category: "Onnxify.ModelGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true
    );

    private static readonly DiagnosticDescriptor _externalDataDescriptor = new(
        id: "OMG004",
        title: "Model uses external tensor data",
        messageFormat: "Model '{0}' references external tensor data. The generated wrapper can load the .onnx file, but runtime inference also requires any sibling external data files to be deployed alongside it.",
        category: "Onnxify.ModelGenerator",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true
    );

    /// <summary>
    /// Registers incremental pipeline steps that analyze ONNX model files and emit typed inference wrappers.
    /// </summary>
    /// <param name="context">The Roslyn incremental generator initialization context.</param>
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var analyzedModels = context.AdditionalTextsProvider
            .Where(static file => string.Equals(Path.GetExtension(file.Path), ".onnx", StringComparison.OrdinalIgnoreCase))
            .Combine(context.AnalyzerConfigOptionsProvider)
            .Select(static (pair, cancellationToken) => Analyze(pair.Left, pair.Right, cancellationToken))
            .Collect();

        context.RegisterSourceOutput(analyzedModels, static (productionContext, results) =>
        {
            Emit(productionContext, results);
        });
    }

    private static ModelAnalysisResult Analyze(
        AdditionalText file,
        AnalyzerConfigOptionsProvider optionsProvider,
        CancellationToken cancellationToken
    )
    {
        cancellationToken.ThrowIfCancellationRequested();

        var fileName = Path.GetFileName(file.Path);
        var diagnostics = new List<Diagnostic>();

        ModelProto model;
        try
        {
            var bytes = File.ReadAllBytes(file.Path);
            model = ModelProto.Parser.ParseFrom(bytes);
        }
        catch (InvalidProtocolBufferException ex)
        {
            diagnostics.Add(
                Diagnostic.Create(
                    _invalidModelDescriptor,
                    location: Location.None,
                    fileName,
                    new InvalidDataException("Unable to parse ONNX protobuf payload.", ex).Message
                )
            );

            return new ModelAnalysisResult(null, diagnostics.ToImmutableArray());
        }
        catch (Exception ex)
        {
            diagnostics.Add(
                Diagnostic.Create(
                    _invalidModelDescriptor,
                    location: Location.None,
                    fileName,
                    ex.Message
                )
            );

            return new ModelAnalysisResult(null, diagnostics.ToImmutableArray());
        }

        var graph = model.Graph;
        var initializerNames = new HashSet<string>(
            graph?.Initializer.Select(static x => x.Name) ?? Enumerable.Empty<string>(),
            StringComparer.Ordinal
        );

        var namespaceName = ResolveNamespace(file, optionsProvider);
        var className = ResolveClassName(file, optionsProvider);
        var projectRelativePath = ResolveProjectRelativePath(file.Path, optionsProvider);

        var inputPropertyNames = new HashSet<string>(StringComparer.Ordinal);
        var outputPropertyNames = new HashSet<string>(StringComparer.Ordinal);
        var inputMethodParameterNames = new HashSet<string>(StringComparer.Ordinal);

        var inputs = new List<ModelTensorContract>();
        foreach (var input in graph?.Input ?? Enumerable.Empty<ValueInfoProto>())
        {
            if (!TryCreateTensorContract(
                ownerFileName: fileName,
                valueInfo: input,
                kind: "input",
                hasDefaultInitializer: initializerNames.Contains(input.Name),
                usedPropertyNames: inputPropertyNames,
                usedMethodParameterNames: inputMethodParameterNames,
                diagnostics: diagnostics,
                contract: out var contract
            ))
            {
                return new ModelAnalysisResult(null, diagnostics.ToImmutableArray());
            }

            inputs.Add(contract ?? throw new InvalidOperationException("Failed to create input tensor contract."));
        }

        var outputs = new List<ModelTensorContract>();
        foreach (var output in graph?.Output ?? Enumerable.Empty<ValueInfoProto>())
        {
            if (!TryCreateTensorContract(
                ownerFileName: fileName,
                valueInfo: output,
                kind: "output",
                hasDefaultInitializer: false,
                usedPropertyNames: outputPropertyNames,
                usedMethodParameterNames: null,
                diagnostics: diagnostics,
                contract: out var contract
            ))
            {
                return new ModelAnalysisResult(null, diagnostics.ToImmutableArray());
            }

            outputs.Add(contract ?? throw new InvalidOperationException("Failed to create output tensor contract."));
        }

        if (graph?.Initializer.Any(HasExternalData) == true)
        {
            diagnostics.Add(
                Diagnostic.Create(
                    _externalDataDescriptor,
                    location: Location.None,
                    fileName
                )
            );
        }

        var specification = new ModelGenerationSpecification(
            fileName,
            projectRelativePath,
            namespaceName,
            className,
            inputs.ToImmutableArray(),
            outputs.ToImmutableArray()
        );

        return new ModelAnalysisResult(specification, diagnostics.ToImmutableArray());
    }

    private static void Emit(
        SourceProductionContext context,
        ImmutableArray<ModelAnalysisResult> results
    )
    {
        foreach (var result in results)
        {
            foreach (var diagnostic in result.Diagnostics)
            {
                context.ReportDiagnostic(diagnostic);
            }
        }

        var successfulSpecifications = results
            .Where(static x => x.Specification is not null)
            .Select(static x => x.Specification!)
            .ToArray();

        var duplicates = successfulSpecifications
            .GroupBy(static x => x.FullyQualifiedClassName, StringComparer.Ordinal)
            .Where(static x => x.Count() > 1)
            .ToDictionary(static x => x.Key, static x => x.ToArray(), StringComparer.Ordinal);

        foreach (var duplicate in duplicates)
        {
            foreach (var _ in duplicate.Value)
            {
                context.ReportDiagnostic(
                    Diagnostic.Create(
                        _duplicateTypeDescriptor,
                        location: Location.None,
                        duplicate.Key
                    )
                );
            }
        }

        foreach (var specification in successfulSpecifications)
        {
            if (duplicates.ContainsKey(specification.FullyQualifiedClassName))
            {
                continue;
            }

            context.AddSource($"{specification.ClassName}.g.cs", GenerateSource(specification));
        }
    }

    private static string GenerateSource(ModelGenerationSpecification specification)
    {
        var inputTypeName = $"{specification.ClassName}Inputs";
        var outputTypeName = $"{specification.ClassName}Outputs";

        return $$"""
        // <auto-generated/>
        #nullable enable
        
        using System;
        using System.Collections.Generic;
        using System.IO;
        using Microsoft.ML.OnnxRuntime;
        using Microsoft.ML.OnnxRuntime.Tensors;

        namespace {{specification.NamespaceName}}
        {
            {{Indent(BuildWrapperType(specification, inputTypeName, outputTypeName), 1)}}

            {{Indent(BuildInputType(specification, inputTypeName), 1)}}

            {{Indent(BuildOutputType(specification, outputTypeName), 1)}}
        }

        #nullable restore
        """;
    }

    private static string BuildInputType(
        ModelGenerationSpecification specification,
        string inputTypeName
    )
    {
        var props = specification.Inputs
            .Select(static input =>
            {
                return $$"""
                {{BuildInputPropertyDocumentation(input)}}
                public {{(input.IsRequired ? "required " : string.Empty)}}Tensor<{{input.ElementClrTypeName}}>{{(input.IsRequired ? string.Empty : "?")}} {{input.PropertyName}} { get; init; }
                """;
            })
            .ToArray();

        var code = string.Join("\n\n", props);

        return $$"""
        {{BuildTensorCollectionTypeDocumentation(
            summary: $"Collects the tensors supplied to {specification.ClassName}.",
            tensors: specification.Inputs,
            roleLabel: "Input property"
        )}}
        public sealed class {{inputTypeName}}
        {
            {{Indent(code, 1)}}
        }
        """;
    }

    private static string BuildOutputType(
        ModelGenerationSpecification specification,
        string outputTypeName
    )
    {
        var props = specification.Outputs
            .Select(static output =>
            {
                return $$"""
                {{BuildOutputPropertyDocumentation(output)}}
                public Tensor<{{output.ElementClrTypeName}}> {{output.PropertyName}} => GetTensor<{{output.ElementClrTypeName}}>("{{Escape(output.OnnxName)}}");
                """;
            })
            .ToArray();

        var code = string.Join("\n\n", props);

        return $$"""
        {{BuildTensorCollectionTypeDocumentation(
            summary: $"Provides typed access to the outputs produced by {specification.ClassName}.",
            tensors: specification.Outputs,
            roleLabel: "Output property"
        )}}
        public sealed class {{outputTypeName}} : IDisposable
        {
            private readonly IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _results;
            private readonly Dictionary<string, DisposableNamedOnnxValue> _values;

            internal {{outputTypeName}}(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
            {
                if (results is null)
                {
                    throw new ArgumentNullException(nameof(results));
                }

                _results = results;
                _values = new Dictionary<string, DisposableNamedOnnxValue>(StringComparer.Ordinal);
                foreach (var value in results)
                {
                    _values[value.Name] = value;
                }
            }

            {{Indent(XmlSummary("Gets the raw ONNX Runtime outputs returned by the inference session."), 1)}}
            public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Raw => _results;
            
            {{Indent(code, 1)}}

            {{Indent(BuildGetTensorDocumentation(), 1)}}
            public Tensor<T> GetTensor<T>(string name)
            {
                if (string.IsNullOrWhiteSpace(name))
                {
                    throw new ArgumentException("Output name must be provided.", nameof(name));
                }

                if (!_values.TryGetValue(name, out var value))
                {
                    throw new KeyNotFoundException($"The output '{name}' was not returned by the model.");
                }

                var tensor = value.AsTensor<T>();
                if (tensor is null)
                {
                    throw new InvalidOperationException($"The output '{name}' is not a tensor of the requested type {typeof(T).FullName}.");
                }

                return tensor;
            }

            {{Indent(XmlSummary("Releases the native ONNX Runtime output values for this inference result."), 1)}}
            public void Dispose()
            {
                _results.Dispose();
            }
        }
        """;
    }

    private static string BuildWrapperType(
        ModelGenerationSpecification specification,
        string inputTypeName,
        string outputTypeName
    )
    {
        var outputNames = string.Join(
            "\n",
            specification.Outputs.Select(static output => $"\"{Escape(output.OnnxName)}\","));

        var constructors = $$"""
        {{XmlSummary($"Creates a {specification.ClassName} that loads the model from DefaultModelPath.")}}
        public {{specification.ClassName}}()
            : this(DefaultModelPath, null)
        {
        }

        {{XmlSummary($"Creates a {specification.ClassName} that loads the model from DefaultModelPath.")}}
        {{XmlParam("sessionOptions", "Optional ONNX Runtime session options used when constructing the inference session.")}}
        public {{specification.ClassName}}(SessionOptions? sessionOptions)
            : this(DefaultModelPath, sessionOptions)
        {
        }

        {{XmlSummary($"Creates a {specification.ClassName} from an ONNX model file path.")}}
        {{XmlParam("modelPath", "Path to the ONNX model file to load.")}}
        public {{specification.ClassName}}(string modelPath)
            : this(modelPath, null)
        {
        }

        {{XmlSummary($"Creates a {specification.ClassName} from an ONNX model file path.")}}
        {{XmlParam("modelPath", "Path to the ONNX model file to load.")}}
        {{XmlParam("sessionOptions", "Optional ONNX Runtime session options used when constructing the inference session.")}}
        public {{specification.ClassName}}(string modelPath, SessionOptions? sessionOptions)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                throw new ArgumentException("Model path must be provided.", nameof(modelPath));
            }

            Session = sessionOptions is null
                ? new InferenceSession(modelPath)
                : new InferenceSession(modelPath, sessionOptions);
        }

        {{XmlSummary($"Creates a {specification.ClassName} from the raw bytes of an ONNX model.")}}
        {{XmlParam("modelBytes", "The raw ONNX model bytes to load into the inference session.")}}
        public {{specification.ClassName}}(byte[] modelBytes)
            : this(modelBytes, null)
        {
        }

        {{XmlSummary($"Creates a {specification.ClassName} from the raw bytes of an ONNX model.")}}
        {{XmlParam("modelBytes", "The raw ONNX model bytes to load into the inference session.")}}
        {{XmlParam("sessionOptions", "Optional ONNX Runtime session options used when constructing the inference session.")}}
        public {{specification.ClassName}}(byte[] modelBytes, SessionOptions? sessionOptions)
        {
            if (modelBytes is null)
            {
                throw new ArgumentNullException(nameof(modelBytes));
            }

            Session = sessionOptions is null
                ? new InferenceSession(modelBytes)
                : new InferenceSession(modelBytes, sessionOptions);
        }
        """;

        var runMethods = specification.Inputs.Length == 0
            ? BuildParameterlessRunMethods(outputTypeName)
            : BuildInputRunMethods(specification, inputTypeName, outputTypeName);

        return $$"""
        {{XmlSummary($"Provides a typed ONNX Runtime wrapper for the model file '{specification.FileName}'.")}}
        public sealed class {{specification.ClassName}} : IDisposable
        {
            {{Indent(XmlSummary("Gets the model path relative to the consuming project directory."), 1)}}
            public const string MODEL_PROJECT_RELATIVE_PATH = {{ToVerbatimStringLiteral(specification.ProjectRelativePath)}};

            {{Indent(XmlSummary("Gets the default runtime path used to locate the ONNX model beside the application output."), 1)}}
            public static string DefaultModelPath => GetDefaultModelPath();

            {{Indent(BuildMetadataCollectionDocumentation("input", specification.Inputs), 1)}}
            public static IReadOnlyList<Onnxify.OnnxValue> Inputs { get; } = CreateInputs();

            {{Indent(BuildMetadataCollectionDocumentation("output", specification.Outputs), 1)}}
            public static IReadOnlyList<Onnxify.OnnxValue> Outputs { get; } = CreateOutputs();

            {{Indent(XmlSummary("Gets the output names requested from ONNX Runtime during inference."), 1)}}
            public static IReadOnlyList<string> OutputNames { get; } = new string[]
            {
                {{Indent(outputNames, 2)}}
            };

            {{Indent(XmlSummary("Gets the underlying ONNX Runtime inference session used by this wrapper."), 1)}}
            public InferenceSession Session { get; }

            private static IReadOnlyList<Onnxify.OnnxValue> CreateInputs()
            {
                {{Indent(BuildOnnxValueMetadata(specification.Inputs), 2)}}
            }

            private static IReadOnlyList<Onnxify.OnnxValue> CreateOutputs()
            {
                {{Indent(BuildOnnxValueMetadata(specification.Outputs), 2)}}
            }

            private static string GetDefaultModelPath()
            {
                return Path.Combine(
                    AppContext.BaseDirectory,
                    MODEL_PROJECT_RELATIVE_PATH
                        .Replace('\\', Path.DirectorySeparatorChar)
                        .Replace('/', Path.DirectorySeparatorChar)
                );
            }

            {{Indent(constructors, 1)}}

            {{Indent(runMethods, 1)}}

            {{Indent(XmlSummary("Releases the underlying ONNX Runtime inference session."), 1)}}
            public void Dispose()
            {
                Session.Dispose();
            }
        }
        """;
    }

    private static string BuildParameterlessRunMethods(string outputTypeName)
    {
        return $$"""
        {{XmlSummary("Runs inference for a model with no required inputs.")}}
        {{XmlReturns($"The typed {outputTypeName} wrapper over the ONNX Runtime outputs.")}}
        public {{outputTypeName}} Run()
        {
            return Run(null);
        }

        {{XmlSummary("Runs inference for a model with no required inputs.")}}
        {{XmlParam("runOptions", "Optional ONNX Runtime run options applied to this inference invocation.")}}
        {{XmlReturns($"The typed {outputTypeName} wrapper over the ONNX Runtime outputs.")}}
        public {{outputTypeName}} Run(RunOptions? runOptions)
        {
            var namedInputs = new List<NamedOnnxValue>(0);
            {{Indent(BuildRunInvocation(outputTypeName), 1)}}
        }
        """;
    }

    private static string BuildInputRunMethods(
        ModelGenerationSpecification specification,
        string inputTypeName,
        string outputTypeName
    )
    {
        var orderedInputs = specification.Inputs
            .OrderByDescending(static x => x.IsRequired)
            .ToArray();

        var namedInputs = string.Join(
            "\n",
            specification.Inputs.Select(static input =>
                input.IsRequired
                    ? $"namedInputs.Add(NamedOnnxValue.CreateFromTensor(\"{Escape(input.OnnxName)}\", inputs.{input.PropertyName} ?? throw new InvalidOperationException(\"Model input '{Escape(input.OnnxName)}' must be provided.\")));"
                    : $$"""
                    if (inputs.{{input.PropertyName}} is not null)
                    {
                        namedInputs.Add(NamedOnnxValue.CreateFromTensor("{{Escape(input.OnnxName)}}", inputs.{{input.PropertyName}}));
                    }
                    """));

        var signature = string.Join(", ", orderedInputs.Select(BuildTensorMethodParameterSignature));
        var signatureWithRunOptions = BuildRunMethodSignatureWithRunOptions(orderedInputs);
        var assignments = string.Join(
            "\n",
            orderedInputs.Select(static x => $"{x.PropertyName} = {x.MethodParameterName},"));

        return $$"""
        {{XmlSummary("Runs inference using the supplied input object.")}}
        {{XmlParam("inputs", $"The {inputTypeName} instance containing tensors for each required model input.")}}
        {{XmlReturns($"The typed {outputTypeName} wrapper over the ONNX Runtime outputs.")}}
        public {{outputTypeName}} Run({{inputTypeName}} inputs)
        {
            return Run(inputs, null);
        }

        {{XmlSummary("Runs inference using the supplied input object.")}}
        {{XmlParam("inputs", $"The {inputTypeName} instance containing tensors for each required model input.")}}
        {{XmlParam("runOptions", "Optional ONNX Runtime run options applied to this inference invocation.")}}
        {{XmlReturns($"The typed {outputTypeName} wrapper over the ONNX Runtime outputs.")}}
        public {{outputTypeName}} Run({{inputTypeName}} inputs, RunOptions? runOptions)
        {
            if (inputs is null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            var namedInputs = new List<NamedOnnxValue>({{specification.Inputs.Length}});
            {{Indent(namedInputs, 1)}}

            {{Indent(BuildRunInvocation(outputTypeName), 1)}}
        }

        {{XmlSummary("Runs inference using individual tensor arguments for each model input.")}}
        {{BuildTensorParameterDocumentation(orderedInputs)}}
        {{XmlReturns($"The typed {outputTypeName} wrapper over the ONNX Runtime outputs.")}}
        public {{outputTypeName}} Run({{signature}})
        {
            return Run(
                inputs: new {{inputTypeName}}
                {
                    {{Indent(assignments, 3)}}
                }, 
                runOptions: null
            );
        }

        {{XmlSummary("Runs inference using individual tensor arguments for each model input.")}}
        {{BuildTensorParameterDocumentation(orderedInputs)}}
        {{XmlParam("runOptions", "Optional ONNX Runtime run options applied to this inference invocation.")}}
        {{XmlReturns($"The typed {outputTypeName} wrapper over the ONNX Runtime outputs.")}}
        public {{outputTypeName}} Run({{signatureWithRunOptions}})
        {
            return Run(
                inputs: new {{inputTypeName}}
                {
                    {{Indent(assignments, 3)}}
                }, 
                runOptions: runOptions
            );
        }
        """;
    }

    private static string BuildRunInvocation(string outputTypeName)
    {
        return $$"""
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
        if (runOptions is null)
        {
            results = OutputNames.Count == 0
                ? Session.Run(namedInputs)
                : Session.Run(namedInputs, OutputNames);
        }
        else
        {
            results = Session.Run(namedInputs, OutputNames, runOptions);
        }

        return new {{outputTypeName}}(results);
        """;
    }

    private static string BuildOnnxValueMetadata(
        ImmutableArray<ModelTensorContract> tensors
    )
    {
        if (tensors.Length == 0)
        {
            return "return Array.Empty<Onnxify.OnnxValue>();";
        }

        var entries = string.Join("\n", tensors.Select(BuildOnnxValueMetadataEntry));

        return $$"""
        return new Onnxify.OnnxValue[]
        {
            {{Indent(entries, 1)}}
        };
        """;
    }

    private static string BuildOnnxValueMetadataEntry(
        ModelTensorContract tensor
    )
    {
        var tensorTypeExpression = BuildOnnxTensorTypeExpression(tensor);

        return $$"""
        new Onnxify.OnnxValue<Onnxify.OnnxTensorType>(
            name: "{{Escape(tensor.OnnxName)}}",
            type: {{Indent(tensorTypeExpression, 1)}}
        ),
        """;
    }

    private static string BuildOnnxTensorTypeExpression(ModelTensorContract tensor)
    {
        if (tensor.Shape.Any(static x => x.IsUnknown))
        {
            return $$"""
                new Onnxify.OnnxTensorType(
                    type: typeof({{tensor.ElementClrTypeName}}),
                    shape: null,
                    denotation: {{tensor.DenotationLiteral}}
                )
                """;
        }

        var dimensions = string.Join("\n", tensor.Shape.Select(BuildOnnxDimensionExpression));

        return $$"""
        Onnxify.OnnxTensorType.Create<{{tensor.ElementClrTypeName}}>(
            shape: new Onnxify.OnnxDimension[]
            {
                {{Indent(dimensions, 2)}}
            },
            denotation: {{tensor.DenotationLiteral}}
        )
        """;
    }

    private static string BuildOnnxDimensionExpression(ModelDimensionContract dimension)
    {
        if (dimension.NumericValueLiteral is not null)
        {
            return $"{dimension.NumericValueLiteral},";
        }

        if (dimension.SymbolicNameLiteral is not null)
        {
            return $"{dimension.SymbolicNameLiteral},";
        }

        throw new InvalidOperationException("Unknown ONNX dimensions cannot be expressed through the public Onnxify OnnxDimension API.");
    }

    private static bool TryCreateTensorContract(
        string ownerFileName,
        ValueInfoProto valueInfo,
        string kind,
        bool hasDefaultInitializer,
        HashSet<string> usedPropertyNames,
        HashSet<string>? usedMethodParameterNames,
        List<Diagnostic> diagnostics,
        out ModelTensorContract? contract
    )
    {
        contract = null;

        var tensorType = GetTensorType(valueInfo.Type, out var onnxTypeName, out var isOptionalType);
        if (tensorType is null)
        {
            diagnostics.Add(
                Diagnostic.Create(
                    _unsupportedTensorDescriptor,
                    location: Location.None,
                    ownerFileName,
                    kind,
                    valueInfo.Name,
                    onnxTypeName
                )
            );

            return false;
        }

        var elementType = (TensorProto.Types.DataType)tensorType.ElemType;
        if (!TryMapElementType(elementType, out var clrTypeName))
        {
            diagnostics.Add(
                Diagnostic.Create(
                    _unsupportedTensorDescriptor,
                    location: Location.None,
                    ownerFileName,
                    kind,
                    valueInfo.Name,
                    elementType.ToString()
                )
            );

            return false;
        }

        var propertyName = MakeUniqueIdentifier(
            baseName: ToPascalIdentifier(valueInfo.Name, kind == "input" ? "Input" : "Output"),
            usedNames: usedPropertyNames,
            fallbackBaseName: kind == "input" ? "Input" : "Output"
        );

        string? methodParameterName = null;
        if (usedMethodParameterNames is not null)
        {
            methodParameterName = MakeUniqueIdentifier(
                baseName: ToCamelIdentifier(propertyName),
                usedNames: usedMethodParameterNames,
                fallbackBaseName: "value"
            );
        }

        var shape = tensorType.Shape?.Dim
            .Select(ToDimensionContract)
            .ToImmutableArray() ?? ImmutableArray<ModelDimensionContract>.Empty;

        var typeDenotation = valueInfo.Type?.Denotation;
        var denotation = string.IsNullOrWhiteSpace(typeDenotation)
            ? null
            : typeDenotation;

        var denotationLiteral = denotation is null
            ? "\"\""
            : $"\"{Escape(denotation)}\"";

        var isOptional = kind == "input" &&
            (isOptionalType || hasDefaultInitializer);

        contract = new ModelTensorContract(
            valueInfo.Name,
            propertyName,
            methodParameterName,
            clrTypeName!,
            denotation,
            denotationLiteral,
            shape,
            !isOptional,
            isOptionalType,
            hasDefaultInitializer
        );

        return true;
    }

    private static TypeProto.Types.Tensor? GetTensorType(
        TypeProto? type,
        out string onnxTypeName,
        out bool isOptionalType
    )
    {
        isOptionalType = false;

        if (type is null)
        {
            onnxTypeName = "Unknown";
            return null;
        }

        switch (type.ValueCase)
        {
            case TypeProto.ValueOneofCase.TensorType:
                onnxTypeName = "Tensor";
                return type.TensorType;
            case TypeProto.ValueOneofCase.OptionalType:
                onnxTypeName = "Optional";
                isOptionalType = true;
                return type.OptionalType?.ElemType?.TensorType;
            case TypeProto.ValueOneofCase.SequenceType:
                onnxTypeName = "Sequence";
                return null;
            case TypeProto.ValueOneofCase.MapType:
                onnxTypeName = "Map";
                return null;
            case TypeProto.ValueOneofCase.OpaqueType:
                onnxTypeName = "Opaque";
                return null;
            case TypeProto.ValueOneofCase.SparseTensorType:
                onnxTypeName = "SparseTensor";
                return null;
            default:
                onnxTypeName = "Unknown";
                return null;
        }
    }

    private static ModelDimensionContract ToDimensionContract(TensorShapeProto.Types.Dimension dimension)
    {
        if (dimension.HasDimValue)
        {
            return new ModelDimensionContract(
                $"{dimension.DimValue}L",
                null,
                false
            );
        }

        if (dimension.HasDimParam && !string.IsNullOrWhiteSpace(dimension.DimParam))
        {
            return new ModelDimensionContract(
                null,
                $"\"{Escape(dimension.DimParam)}\"",
                false
            );
        }

        return new ModelDimensionContract(
            null,
            null,
            true
        );
    }

    private static string ResolveNamespace(
        AdditionalText file,
        AnalyzerConfigOptionsProvider optionsProvider
    )
    {
        var fileOptions = optionsProvider.GetOptions(file);
        if (fileOptions.TryGetValue(ADDITIONAL_FILE_NAMESPACE_KEY, out var overrideNamespace) && !string.IsNullOrWhiteSpace(overrideNamespace))
        {
            return NormalizeNamespace(overrideNamespace);
        }

        if (optionsProvider.GlobalOptions.TryGetValue(GLOBAL_ROOT_NAMESPACE_KEY, out var rootNamespace) && !string.IsNullOrWhiteSpace(rootNamespace))
        {
            return NormalizeNamespace(rootNamespace);
        }

        if (optionsProvider.GlobalOptions.TryGetValue(GLOBAL_ASSEMBLY_NAME_KEY, out var assemblyName) && !string.IsNullOrWhiteSpace(assemblyName))
        {
            return NormalizeNamespace(assemblyName);
        }

        return "GeneratedOnnxModels";
    }

    private static string ResolveClassName(
        AdditionalText file,
        AnalyzerConfigOptionsProvider optionsProvider
    )
    {
        var fileOptions = optionsProvider.GetOptions(file);
        if (fileOptions.TryGetValue(ADDITIONAL_FILE_CLASS_NAME_KEY, out var overrideClassName) && !string.IsNullOrWhiteSpace(overrideClassName))
        {
            var sanitizedOverride = ToPascalIdentifier(overrideClassName, "Model");
            return sanitizedOverride.EndsWith("Model", StringComparison.Ordinal) ? sanitizedOverride : $"{sanitizedOverride}Model";
        }

        var baseName = ToPascalIdentifier(Path.GetFileNameWithoutExtension(file.Path), "Model");
        return baseName.EndsWith("Model", StringComparison.Ordinal) ? baseName : $"{baseName}Model";
    }

    private static string ResolveProjectRelativePath(
        string filePath,
        AnalyzerConfigOptionsProvider optionsProvider
    )
    {
        if (TryGetProjectDirectory(optionsProvider, out var projectDirectory) && !string.IsNullOrWhiteSpace(projectDirectory))
        {
            try
            {
                var relativePath = GetProjectRelativePath(projectDirectory, filePath);
                if (!relativePath.StartsWith("..", StringComparison.Ordinal) && !Path.IsPathRooted(relativePath))
                {
                    return NormalizeGeneratedPath(relativePath);
                }
            }
            catch
            {
            }
        }

        return NormalizeGeneratedPath(Path.GetFileName(filePath));
    }

    private static bool TryGetProjectDirectory(
        AnalyzerConfigOptionsProvider optionsProvider,
        out string projectDirectory
    )
    {
        if (optionsProvider.GlobalOptions.TryGetValue(GLOBAL_PROJECT_DIRECTORY_KEY, out var msbuildProjectDirectory) && !string.IsNullOrWhiteSpace(msbuildProjectDirectory))
        {
            projectDirectory = msbuildProjectDirectory;
            return true;
        }

        if (optionsProvider.GlobalOptions.TryGetValue(GLOBAL_PROJECT_DIR_KEY, out var projectDir) && !string.IsNullOrWhiteSpace(projectDir))
        {
            projectDirectory = projectDir;
            return true;
        }

        projectDirectory = string.Empty;
        return false;
    }

    private static string NormalizeNamespace(string namespaceName)
    {
        var segments = namespaceName
            .Split(['.'], StringSplitOptions.RemoveEmptyEntries)
            .Select(static segment => ToPascalIdentifier(segment, "Generated"))
            .ToArray();

        return segments.Length == 0
            ? "GeneratedOnnxModels"
            : string.Join(".", segments);
    }

    private static string ToPascalIdentifier(string? value, string fallback)
    {
        if (value is null || string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        var builder = new StringBuilder();
        var token = new StringBuilder();

        foreach (var character in value)
        {
            if (char.IsLetterOrDigit(character))
            {
                token.Append(character);
            }
            else
            {
                AppendToken(builder, token);
            }
        }

        AppendToken(builder, token);

        if (builder.Length == 0)
        {
            builder.Append(fallback);
        }

        var result = builder.ToString();
        if (result.Length > 0 && !char.IsLetter(result[0]) && result[0] != '_')
        {
            result = fallback + result;
        }

        if (SyntaxFacts.GetKeywordKind(result) != SyntaxKind.None)
        {
            result += "Value";
        }

        return result;
    }

    private static void AppendToken(StringBuilder builder, StringBuilder token)
    {
        if (token.Length == 0)
        {
            return;
        }

        builder.Append(char.ToUpperInvariant(token[0]));
        if (token.Length > 1)
        {
            builder.Append(token.ToString(1, token.Length - 1));
        }

        token.Clear();
    }

    private static string ToCamelIdentifier(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return "value";
        }

        var characters = value.ToCharArray();
        characters[0] = char.ToLowerInvariant(characters[0]);

        var result = new string(characters);
        if (SyntaxFacts.GetKeywordKind(result) != SyntaxKind.None)
        {
            result += "Value";
        }

        return result;
    }

    private static string MakeUniqueIdentifier(
        string baseName,
        HashSet<string> usedNames,
        string fallbackBaseName
    )
    {
        var candidate = string.IsNullOrWhiteSpace(baseName) ? fallbackBaseName : baseName;
        if (usedNames.Add(candidate))
        {
            return candidate;
        }

        var counter = 2;
        while (!usedNames.Add(candidate + counter.ToString()))
        {
            counter++;
        }

        return candidate + counter.ToString();
    }

    private static bool HasExternalData(TensorProto tensor)
    {
        return tensor.DataLocation == TensorProto.Types.DataLocation.External || tensor.ExternalData.Count > 0;
    }

    private static bool TryMapElementType(
        TensorProto.Types.DataType dataType,
        out string? clrTypeName
    )
    {
        clrTypeName = dataType switch
        {
            TensorProto.Types.DataType.Float => "float",
            TensorProto.Types.DataType.Double => "double",
            TensorProto.Types.DataType.Int64 => "long",
            TensorProto.Types.DataType.Uint64 => "ulong",
            TensorProto.Types.DataType.Int32 => "int",
            TensorProto.Types.DataType.Uint32 => "uint",
            TensorProto.Types.DataType.Int16 => "short",
            TensorProto.Types.DataType.Uint16 => "ushort",
            TensorProto.Types.DataType.Int8 => "sbyte",
            TensorProto.Types.DataType.Uint8 => "byte",
            TensorProto.Types.DataType.Bool => "bool",
            TensorProto.Types.DataType.String => "string",
            TensorProto.Types.DataType.Float16 => "Float16",
            TensorProto.Types.DataType.Bfloat16 => "BFloat16",
            _ => null,
        };

        return clrTypeName is not null;
    }

    private static string Escape(string value)
    {
        var builder = new StringBuilder(value.Length);
        foreach (var ch in value)
        {
            switch (ch)
            {
                case '\\':
                    builder.Append("\\\\");
                    break;
                case '\"':
                    builder.Append("\\\"");
                    break;
                case '\0':
                    builder.Append("\\0");
                    break;
                case '\a':
                    builder.Append("\\a");
                    break;
                case '\b':
                    builder.Append("\\b");
                    break;
                case '\f':
                    builder.Append("\\f");
                    break;
                case '\n':
                    builder.Append("\\n");
                    break;
                case '\r':
                    builder.Append("\\r");
                    break;
                case '\t':
                    builder.Append("\\t");
                    break;
                case '\v':
                    builder.Append("\\v");
                    break;
                default:
                    if (char.IsControl(ch))
                    {
                        builder.Append("\\u");
                        builder.Append(((int)ch).ToString("x4"));
                    }
                    else
                    {
                        builder.Append(ch);
                    }

                    break;
            }
        }

        return builder.ToString();
    }

    private static string ToVerbatimStringLiteral(string value)
    {
        return "@\"" + value.Replace("\"", "\"\"") + "\"";
    }

    private static string Indent(string text, int tabs)
    {
        var indent = new string(' ', tabs * 4);
        return text.Trim().Replace("\n", $"\n{indent}").Trim();
    }

    private static string GetProjectRelativePath(string projectDirectory, string filePath)
    {
        var normalizedProjectDirectory = Path.GetFullPath(projectDirectory);
        if (!normalizedProjectDirectory.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal))
        {
            normalizedProjectDirectory += Path.DirectorySeparatorChar;
        }

        var normalizedFilePath = Path.GetFullPath(filePath);
        if (normalizedFilePath.StartsWith(normalizedProjectDirectory, StringComparison.OrdinalIgnoreCase))
        {
            return normalizedFilePath.Substring(normalizedProjectDirectory.Length);
        }

        return filePath;
    }

    private static string NormalizeGeneratedPath(string path)
    {
        return path.Replace('/', '\\');
    }

    private static string BuildInputPropertyDocumentation(ModelTensorContract input)
    {
        return BuildTensorPropertyDocumentation(
            summary: input.IsRequired
                ? $"Gets or initializes the tensor supplied for model input '{input.OnnxName}'."
                : $"Gets or initializes the optional tensor supplied for model input '{input.OnnxName}'.",
            tensor: input
        );
    }

    private static string BuildOutputPropertyDocumentation(ModelTensorContract output)
    {
        return BuildTensorPropertyDocumentation(
            summary: $"Gets the tensor returned for model output '{output.OnnxName}'.",
            tensor: output
        );
    }

    private static string BuildGetTensorDocumentation()
    {
        return $$"""
        {{XmlSummary("Gets a typed tensor from the raw ONNX Runtime output collection by output name.")}}
        {{XmlParam("name", "The ONNX output name to resolve from the inference result.")}}
        {{XmlReturns("The tensor value for the requested output name.")}}
        """;
    }

    private static string BuildTensorParameterDocumentation(
        IEnumerable<ModelTensorContract> tensors
    )
    {
        var x = tensors.Select(static tensor =>
        {
            return XmlParamXml(
                tensor.MethodParameterName!,
                BuildTensorMethodParameterDescription(tensor)
            );
        });

        return string.Join("\n", x);
    }

    private static string XmlSummary(string text)
    {
        return $$"""
        /// <summary>
        /// {{EscapeXml(text)}}
        /// </summary>
        """;
    }

    private static string XmlParam(string name, string text)
    {
        return $"""/// <param name="{name}">{EscapeXml(text)}</param>""";
    }

    private static string XmlParamXml(string name, string xml)
    {
        return $"""/// <param name="{name}">{xml}</param>""";
    }

    private static string XmlReturns(string text)
    {
        return $"""/// <returns>{EscapeXml(text)}</returns>""";
    }

    private static string BuildTensorPropertyDocumentation(
        string summary,
        ModelTensorContract tensor
    )
    {
        return BuildXmlDocumentation(
            summary,
            BuildTensorDocumentationParagraphs(tensor)
        );
    }

    private static string BuildMetadataCollectionDocumentation(
        string kind,
        ImmutableArray<ModelTensorContract> tensors
    )
    {
        var summary = $"Describes the generated ONNX model {kind}s using Onnxify metadata objects.";

        return BuildXmlDocumentation(
            summary,
            tensors.Select(BuildMetadataParagraph)
        );
    }

    private static string BuildTensorCollectionTypeDocumentation(
        string summary,
        ImmutableArray<ModelTensorContract> tensors,
        string roleLabel
    )
    {
        return BuildXmlDocumentation(
            summary,
            tensors.Select(tensor => BuildTensorCollectionParagraph(tensor, roleLabel))
        );
    }

    private static IEnumerable<string> BuildTensorDocumentationParagraphs(
        ModelTensorContract tensor
    )
    {
        yield return $"Tensor type: {FormatCode($"Tensor<{tensor.ElementClrTypeName}>")}";
        yield return $"Element type: {FormatCode(tensor.ElementClrTypeName)}";
        yield return $"Shape: {FormatCode(FormatTensorShape(tensor.Shape))}";

        if (!string.IsNullOrWhiteSpace(tensor.Denotation))
        {
            yield return $"Denotation: {FormatCode(tensor.Denotation!)}";
        }
    }

    private static string BuildMetadataParagraph(ModelTensorContract tensor)
    {
        var builder = new StringBuilder();
        builder.Append($"{FormatCode(tensor.OnnxName)}: {FormatCode($"Tensor<{tensor.ElementClrTypeName}>")}, shape {FormatCode(FormatTensorShape(tensor.Shape))}");

        if (!string.IsNullOrWhiteSpace(tensor.Denotation))
        {
            builder.Append($", denotation {FormatCode(tensor.Denotation!)}");
        }

        return builder.ToString();
    }

    private static string BuildTensorCollectionParagraph(
        ModelTensorContract tensor,
        string roleLabel
    )
    {
        var builder = new StringBuilder();
        builder.Append($"{roleLabel} {FormatCode(tensor.PropertyName)} maps to ONNX name {FormatCode(tensor.OnnxName)}");
        builder.Append($"; tensor type {FormatCode($"Tensor<{tensor.ElementClrTypeName}>")}");
        builder.Append($"; shape {FormatCode(FormatTensorShape(tensor.Shape))}");

        if (!string.IsNullOrWhiteSpace(tensor.Denotation))
        {
            builder.Append($"; denotation {FormatCode(tensor.Denotation!)}");
        }

        return builder.ToString();
    }

    private static string BuildTensorMethodParameterDescription(ModelTensorContract tensor)
    {
        var builder = new StringBuilder();
        builder.Append($"{(tensor.IsRequired ? "Tensor" : "Optional tensor")} value for model input {FormatCode(tensor.OnnxName)}");
        builder.Append($"; parameter type {FormatCode(BuildInputTensorTypeName(tensor))}");
        builder.Append($"; shape {FormatCode(FormatTensorShape(tensor.Shape))}");

        if (!string.IsNullOrWhiteSpace(tensor.Denotation))
        {
            builder.Append($"; denotation {FormatCode(tensor.Denotation!)}");
        }

        if (!tensor.IsRequired)
        {
            builder.Append(tensor.HasDefaultInitializer
                ? "; pass null to omit this input and let the model use its initializer-backed default"
                : "; pass null to omit this optional ONNX input");
        }

        return builder.ToString();
    }

    private static string BuildTensorMethodParameterSignature(ModelTensorContract tensor)
    {
        return tensor.IsRequired
            ? $"Tensor<{tensor.ElementClrTypeName}> {tensor.MethodParameterName}"
            : $"Tensor<{tensor.ElementClrTypeName}>? {tensor.MethodParameterName} = null";
    }

    private static string BuildRunMethodSignatureWithRunOptions(
        IReadOnlyList<ModelTensorContract> orderedInputs
    )
    {
        var requiredInputs = orderedInputs
            .Where(static x => x.IsRequired)
            .Select(BuildTensorMethodParameterSignature);
        var optionalInputs = orderedInputs
            .Where(static x => !x.IsRequired)
            .Select(BuildTensorMethodParameterSignature);

        return string.Join(
            ", ",
            requiredInputs
                .Concat(["RunOptions? runOptions"])
                .Concat(optionalInputs)
        );
    }

    private static string BuildInputTensorTypeName(ModelTensorContract tensor)
    {
        return tensor.IsRequired
            ? $"Tensor<{tensor.ElementClrTypeName}>"
            : $"Tensor<{tensor.ElementClrTypeName}>?";
    }

    private static string BuildXmlDocumentation(
        string summary,
        IEnumerable<string> paragraphs
    )
    {
        var lines = new List<string>
        {
            "/// <summary>",
            $"/// {EscapeXml(summary)}",
        };

        foreach (var paragraph in paragraphs)
        {
            lines.Add($"/// <para>{paragraph}</para>");
        }

        lines.Add("/// </summary>");
        return string.Join("\n", lines);
    }

    private static string FormatTensorShape(ImmutableArray<ModelDimensionContract> shape)
    {
        if (shape.Length == 0)
        {
            return "[]";
        }

        var dimensions = shape.Select(static dimension =>
        {
            if (dimension.NumericValueLiteral is not null)
            {
                return dimension.NumericValueLiteral.EndsWith("L", StringComparison.Ordinal)
                    ? dimension.NumericValueLiteral.Substring(0, dimension.NumericValueLiteral.Length - 1)
                    : dimension.NumericValueLiteral;
            }

            if (dimension.SymbolicNameLiteral is not null)
            {
                return dimension.SymbolicNameLiteral.Trim('"');
            }

            return "?";
        });

        return $"[{string.Join(", ", dimensions)}]";
    }

    private static string FormatCode(string text)
    {
        return $"<c>{EscapeXml(text)}</c>";
    }

    private static string EscapeXml(string text)
    {
        return text
            .Replace("&", "&amp;")
            .Replace("<", "&lt;")
            .Replace(">", "&gt;");
    }

    private sealed record ModelAnalysisResult(
        ModelGenerationSpecification? Specification,
        ImmutableArray<Diagnostic> Diagnostics
    );

    private sealed record ModelGenerationSpecification(
        string FileName,
        string ProjectRelativePath,
        string NamespaceName,
        string ClassName,
        ImmutableArray<ModelTensorContract> Inputs,
        ImmutableArray<ModelTensorContract> Outputs
    )
    {
        public string FullyQualifiedClassName => $"{NamespaceName}.{ClassName}";
    }

    private sealed record ModelTensorContract(
        string OnnxName,
        string PropertyName,
        string? MethodParameterName,
        string ElementClrTypeName,
        string? Denotation,
        string DenotationLiteral,
        ImmutableArray<ModelDimensionContract> Shape,
        bool IsRequired,
        bool IsOptionalType,
        bool HasDefaultInitializer
    );

    private sealed record ModelDimensionContract(
        string? NumericValueLiteral,
        string? SymbolicNameLiteral,
        bool IsUnknown
    );
}
