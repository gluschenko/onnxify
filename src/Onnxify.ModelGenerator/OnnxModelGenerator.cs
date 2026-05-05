using System.Collections.Immutable;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Onnxify.ModelGenerator;

[Generator]
public sealed class OnnxModelGenerator : IIncrementalGenerator
{
    private const string GlobalProjectDirectoryKey = "build_property.MSBuildProjectDirectory";
    private const string GlobalProjectDirKey = "build_property.ProjectDir";
    private const string GlobalRootNamespaceKey = "build_property.RootNamespace";
    private const string GlobalAssemblyNameKey = "build_property.AssemblyName";
    private const string AdditionalFileClassNameKey = "build_metadata.AdditionalFiles.OnnxifyModelClassName";
    private const string AdditionalFileNamespaceKey = "build_metadata.AdditionalFiles.OnnxifyModelNamespace";

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
        messageFormat: "Multiple ONNX AdditionalFiles would generate the same wrapper type '{0}'. Override at least one file with the metadata OnnxifyModelClassName or OnnxifyModelNamespace.",
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
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var fileName = Path.GetFileName(file.Path);
        var diagnostics = new List<Diagnostic>();

        ParsedOnnxModel model;
        try
        {
            var bytes = File.ReadAllBytes(file.Path);
            model = OnnxModelMetadataReader.ReadModel(bytes);
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
            graph.Initializers.Select(static x => x.Name),
            StringComparer.Ordinal
        );

        var namespaceName = ResolveNamespace(file, optionsProvider);
        var className = ResolveClassName(file, optionsProvider);
        var projectRelativePath = ResolveProjectRelativePath(file.Path, optionsProvider);

        var inputPropertyNames = new HashSet<string>(StringComparer.Ordinal);
        var outputPropertyNames = new HashSet<string>(StringComparer.Ordinal);
        var inputMethodParameterNames = new HashSet<string>(StringComparer.Ordinal);

        var inputs = new List<ModelTensorContract>();
        foreach (var input in graph.Inputs.Where(x => !initializerNames.Contains(x.Name)))
        {
            if (!TryCreateTensorContract(
                    ownerFileName: fileName,
                    valueInfo: input,
                    kind: "input",
                    usedPropertyNames: inputPropertyNames,
                    usedMethodParameterNames: inputMethodParameterNames,
                    diagnostics: diagnostics,
                    contract: out var contract
            ))
            {
                return new ModelAnalysisResult(null, diagnostics.ToImmutableArray());
            }

            inputs.Add(contract!);
        }

        var outputs = new List<ModelTensorContract>();
        foreach (var output in graph.Outputs)
        {
            if (!TryCreateTensorContract(
                    ownerFileName: fileName,
                    valueInfo: output,
                    kind: "output",
                    usedPropertyNames: outputPropertyNames,
                    usedMethodParameterNames: null,
                    diagnostics: diagnostics,
                    contract: out var contract
            ))
            {
                return new ModelAnalysisResult(null, diagnostics.ToImmutableArray());
            }

            outputs.Add(contract!);
        }

        if (graph.Initializers.Any(static x => x.HasExternalData))
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
        ImmutableArray<ModelAnalysisResult> results)
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
        string inputTypeName)
    {
        var props = specification.Inputs
            .Select(static input => $"public Tensor<{input.ElementClrTypeName}>? {input.PropertyName} {{ get; set; }}")
            .ToArray();

        var code = string.Join("\n", props);

        return $$"""
            public sealed class {{inputTypeName}}
            {
                {{Indent(code, 1)}}
            }
            """;
    }

    private static string BuildOutputType(
        ModelGenerationSpecification specification,
        string outputTypeName)
    {
        var props = specification.Outputs
            .Select(static output => $"public Tensor<{output.ElementClrTypeName}> {output.PropertyName} => GetTensor<{output.ElementClrTypeName}>(\"{Escape(output.OnnxName)}\");")
            .ToArray();

        var code = string.Join("\n", props);

        return $$"""
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

            public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Raw => _results;
            
            {{Indent(code, 1)}}

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
        string outputTypeName)
    {
        var outputNames = string.Join(
            "\n",
            specification.Outputs.Select(static output => $"\"{Escape(output.OnnxName)}\","));

        var constructors = $$"""
        public {{specification.ClassName}}()
            : this(DefaultModelPath, null)
        {
        }

        public {{specification.ClassName}}(SessionOptions? sessionOptions)
            : this(DefaultModelPath, sessionOptions)
        {
        }

        public {{specification.ClassName}}(string modelPath)
            : this(modelPath, null)
        {
        }

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

        public {{specification.ClassName}}(byte[] modelBytes)
            : this(modelBytes, null)
        {
        }

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
        public sealed class {{specification.ClassName}} : IDisposable
        {
            public const string ModelProjectRelativePath = {{ToVerbatimStringLiteral(specification.ProjectRelativePath)}};

            public static string DefaultModelPath => GetDefaultModelPath();

            public static IReadOnlyList<Onnxify.OnnxValue> Inputs { get; } = CreateInputs();

            public static IReadOnlyList<Onnxify.OnnxValue> Outputs { get; } = CreateOutputs();

            public static IReadOnlyList<string> OutputNames { get; } = new string[]
            {
                {{Indent(outputNames, 2)}}
            };

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
                    ModelProjectRelativePath
                        .Replace('\\', Path.DirectorySeparatorChar)
                        .Replace('/', Path.DirectorySeparatorChar)
                );
            }

            {{Indent(constructors, 1)}}

            {{Indent(runMethods, 1)}}

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
        public {{outputTypeName}} Run()
        {
            return Run(null);
        }

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
        string outputTypeName)
    {
        var namedInputs = string.Join(
            "\n",
            specification.Inputs.Select(static input =>
                $"namedInputs.Add(NamedOnnxValue.CreateFromTensor(\"{Escape(input.OnnxName)}\", inputs.{input.PropertyName} ?? throw new InvalidOperationException(\"Model input '{Escape(input.OnnxName)}' must be provided.\")));"));

        var signature = string.Join(", ", specification.Inputs.Select(static x => $"Tensor<{x.ElementClrTypeName}> {x.MethodParameterName}"));
        var assignments = string.Join(
            "\n",
            specification.Inputs.Select(static x => $"{x.PropertyName} = {x.MethodParameterName},"));

        return $$"""
        public {{outputTypeName}} Run({{inputTypeName}} inputs)
        {
            return Run(inputs, null);
        }

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

        public {{outputTypeName}} Run({{signature}})
        {
            return Run(new {{inputTypeName}}
            {
                {{Indent(assignments, 1)}}
            }, null);
        }

        public {{outputTypeName}} Run({{signature}}, RunOptions? runOptions)
        {
            return Run(new {{inputTypeName}}
            {
                {{Indent(assignments, 1)}}
            }, runOptions);
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
        ImmutableArray<ModelTensorContract> tensors)
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
        ModelTensorContract tensor)
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
        ParsedOnnxValueInfo valueInfo,
        string kind,
        HashSet<string> usedPropertyNames,
        HashSet<string>? usedMethodParameterNames,
        List<Diagnostic> diagnostics,
        out ModelTensorContract? contract)
    {
        contract = null;

        if (valueInfo.Type.Kind != OnnxValueKind.Tensor || valueInfo.Type.TensorType is null)
        {
            diagnostics.Add(
                Diagnostic.Create(
                    _unsupportedTensorDescriptor,
                    location: Location.None,
                    ownerFileName,
                    kind,
                    valueInfo.Name,
                    valueInfo.Type.Kind.ToString()
                )
            );

            return false;
        }

        var tensorType = valueInfo.Type.TensorType;
        var elementType = tensorType.ElementType;
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

        var shape = tensorType.Shape
            .Select(ToDimensionContract)
            .ToImmutableArray();

        var denotationLiteral = string.IsNullOrWhiteSpace(valueInfo.Type.Denotation)
            ? "\"\""
            : $"\"{Escape(valueInfo.Type.Denotation)}\"";

        contract = new ModelTensorContract(
            valueInfo.Name,
            propertyName,
            methodParameterName,
            clrTypeName!,
            denotationLiteral,
            shape);

        return true;
    }

    private static ModelDimensionContract ToDimensionContract(ParsedOnnxDimension dimension)
    {
        if (dimension.NumericValue.HasValue)
        {
            return new ModelDimensionContract(
                $"{dimension.NumericValue.Value}L",
                null,
                false);
        }

        if (!string.IsNullOrWhiteSpace(dimension.SymbolicName))
        {
            return new ModelDimensionContract(
                null,
                $"\"{Escape(dimension.SymbolicName!)}\"",
                false);
        }

        return new ModelDimensionContract(
            null,
            null,
            true
        );
    }

    private static string ResolveNamespace(
        AdditionalText file,
        AnalyzerConfigOptionsProvider optionsProvider)
    {
        var fileOptions = optionsProvider.GetOptions(file);
        if (fileOptions.TryGetValue(AdditionalFileNamespaceKey, out var overrideNamespace) && !string.IsNullOrWhiteSpace(overrideNamespace))
        {
            return NormalizeNamespace(overrideNamespace);
        }

        if (optionsProvider.GlobalOptions.TryGetValue(GlobalRootNamespaceKey, out var rootNamespace) && !string.IsNullOrWhiteSpace(rootNamespace))
        {
            return NormalizeNamespace(rootNamespace);
        }

        if (optionsProvider.GlobalOptions.TryGetValue(GlobalAssemblyNameKey, out var assemblyName) && !string.IsNullOrWhiteSpace(assemblyName))
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
        if (fileOptions.TryGetValue(AdditionalFileClassNameKey, out var overrideClassName) && !string.IsNullOrWhiteSpace(overrideClassName))
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
        if (optionsProvider.GlobalOptions.TryGetValue(GlobalProjectDirectoryKey, out var msbuildProjectDirectory) && !string.IsNullOrWhiteSpace(msbuildProjectDirectory))
        {
            projectDirectory = msbuildProjectDirectory;
            return true;
        }

        if (optionsProvider.GlobalOptions.TryGetValue(GlobalProjectDirKey, out var projectDir) && !string.IsNullOrWhiteSpace(projectDir))
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
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        var builder = new StringBuilder();
        var token = new StringBuilder();

        var safeValue = value!;
        foreach (var character in safeValue)
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
        string fallbackBaseName)
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

    private static bool TryMapElementType(
        OnnxTensorDataType dataType,
        out string? clrTypeName)
    {
        clrTypeName = dataType switch
        {
            OnnxTensorDataType.Float => "float",
            OnnxTensorDataType.Double => "double",
            OnnxTensorDataType.Int64 => "long",
            OnnxTensorDataType.Uint64 => "ulong",
            OnnxTensorDataType.Int32 => "int",
            OnnxTensorDataType.Uint32 => "uint",
            OnnxTensorDataType.Int16 => "short",
            OnnxTensorDataType.Uint16 => "ushort",
            OnnxTensorDataType.Int8 => "sbyte",
            OnnxTensorDataType.Uint8 => "byte",
            OnnxTensorDataType.Bool => "bool",
            OnnxTensorDataType.String => "string",
            OnnxTensorDataType.Float16 => "Half",
            _ => null,
        };

        return clrTypeName is not null;
    }

    private static string Escape(string value)
    {
        return value
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"");
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
        string DenotationLiteral,
        ImmutableArray<ModelDimensionContract> Shape
    );

    private sealed record ModelDimensionContract(
        string? NumericValueLiteral,
        string? SymbolicNameLiteral,
        bool IsUnknown
    );
}
