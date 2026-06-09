using System.Collections.Immutable;
using System.Text;
using Google.Protobuf;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Onnx;
using Onnxify.ModelGenerator.Services;
using Onnxify.ModelGenerator.Services.TorchModuleOperators;
using static Onnxify.ModelGenerator.Helpers.TextHelper;

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
    private const string ADDITIONAL_FILE_CLASS_NAME_KEY = "build_metadata.additionalfiles.OnnxifyModelClassName";
    private const string ADDITIONAL_FILE_NAMESPACE_KEY = "build_metadata.additionalfiles.OnnxifyModelNamespace";
    private const string ADDITIONAL_FILE_IMPORT_TYPE_KEY = "build_metadata.additionalfiles.OnnxifyModelImportType";

    private static readonly ImmutableDictionary<string, TorchModuleOperator> _torchModuleOperators = TorchModuleOperatorRegistry.Create();

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

    private static readonly DiagnosticDescriptor _invalidImportTypeDescriptor = new(
        id: "OMG005",
        title: "Invalid ONNX model import type",
        messageFormat: "Model '{0}' uses invalid OnnxifyModelImportType value '{1}'. Supported values are OnnxRuntimeInference and TorchModule.",
        category: "Onnxify.ModelGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true
    );

    private static readonly DiagnosticDescriptor _unsupportedTorchModuleDescriptor = new(
        id: "OMG006",
        title: "Unsupported ONNX graph for TorchModule generation",
        messageFormat: "Model '{0}' cannot generate a TorchModule: {1}",
        category: "Onnxify.ModelGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
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
        var importTypes = ResolveImportTypes(file, optionsProvider, diagnostics);
        var projectRelativePath = ResolveProjectRelativePath(file.Path, optionsProvider);

        if (importTypes == ModelImportType.None)
        {
            return new ModelAnalysisResult(null, diagnostics.ToImmutableArray());
        }

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

        TorchModuleGenerationSpecification? torchModule = null;
        if (importTypes.HasFlag(ModelImportType.TorchModule))
        {
            torchModule = AnalyzeTorchModuleGraph(fileName, model, diagnostics);
            if (torchModule is null)
            {
                return new ModelAnalysisResult(null, diagnostics.ToImmutableArray());
            }
        }

        var specification = new ModelGenerationSpecification(
            fileName,
            projectRelativePath,
            namespaceName,
            className,
            importTypes,
            inputs.ToImmutableArray(),
            outputs.ToImmutableArray(),
            torchModule
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

            if (specification.ImportTypes.HasFlag(ModelImportType.OnnxRuntimeInference))
            {
                context.AddSource($"{specification.ClassName}.g.cs", new OnnxRuntimeInferencePrinter().GenerateSource(specification));
            }

            if (specification.ImportTypes.HasFlag(ModelImportType.TorchModule))
            {
                context.AddSource($"{specification.ClassName}TorchModule.g.cs", new TorchModulePrinter().GenerateSource(specification));
            }
        }
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

    private static ModelImportType ResolveImportTypes(
        AdditionalText file,
        AnalyzerConfigOptionsProvider optionsProvider,
        List<Diagnostic> diagnostics
    )
    {
        var fileOptions = optionsProvider.GetOptions(file);
        if (!fileOptions.TryGetValue(ADDITIONAL_FILE_IMPORT_TYPE_KEY, out var importTypeText) || string.IsNullOrWhiteSpace(importTypeText))
        {
            return ModelImportType.OnnxRuntimeInference;
        }

        var importTypes = ModelImportType.None;
        foreach (var part in importTypeText.Split(','))
        {
            var value = part.Trim();
            if (string.IsNullOrEmpty(value))
            {
                continue;
            }

            if (string.Equals(value, "OnnxRuntimeInference", StringComparison.OrdinalIgnoreCase))
            {
                importTypes |= ModelImportType.OnnxRuntimeInference;
                continue;
            }

            if (string.Equals(value, "TorchModule", StringComparison.OrdinalIgnoreCase))
            {
                importTypes |= ModelImportType.TorchModule;
                continue;
            }

            diagnostics.Add(
                Diagnostic.Create(
                    _invalidImportTypeDescriptor,
                    location: Location.None,
                    Path.GetFileName(file.Path),
                    importTypeText
                )
            );

            return ModelImportType.None;
        }

        return importTypes == ModelImportType.None
            ? ModelImportType.OnnxRuntimeInference
            : importTypes;
    }

    private static TorchModuleGenerationSpecification? AnalyzeTorchModuleGraph(
        string fileName,
        ModelProto model,
        List<Diagnostic> diagnostics
    )
    {
        var graph = model.Graph;
        if (graph is null)
        {
            ReportUnsupportedTorchModule(fileName, "the model does not contain a graph.", diagnostics);
            return null;
        }

        var initializerNames = new HashSet<string>(
            graph.Initializer.Select(static x => x.Name),
            StringComparer.Ordinal
        );

        var runtimeInputs = graph.Input
            .Where(input => !initializerNames.Contains(input.Name))
            .ToArray();

        if (runtimeInputs.Length != 1)
        {
            ReportUnsupportedTorchModule(fileName, "the MVP TorchModule backend supports exactly one non-initializer graph input.", diagnostics);
            return null;
        }

        if (graph.Output.Count != 1)
        {
            ReportUnsupportedTorchModule(fileName, "the MVP TorchModule backend supports exactly one graph output.", diagnostics);
            return null;
        }

        if (!TryGetTensorElementType(runtimeInputs[0], out var inputElementType) || inputElementType != TensorProto.Types.DataType.Float)
        {
            ReportUnsupportedTorchModule(fileName, "the MVP TorchModule backend supports only float32 runtime tensor inputs.", diagnostics);
            return null;
        }

        if (!TryGetTensorElementType(graph.Output[0], out var outputElementType) || outputElementType != TensorProto.Types.DataType.Float)
        {
            ReportUnsupportedTorchModule(fileName, "the MVP TorchModule backend supports only float32 runtime tensor outputs.", diagnostics);
            return null;
        }

        var initializerFieldNames = new HashSet<string>(StringComparer.Ordinal);
        var allInitializers = new Dictionary<string, TorchInitializerSpecification>(StringComparer.Ordinal);
        foreach (var initializer in graph.Initializer)
        {
            if (!TryCreateTorchInitializer(initializer, initializerFieldNames, out var specification, out var error))
            {
                ReportUnsupportedTorchModule(fileName, error, diagnostics);
                return null;
            }

            allInitializers[specification.OnnxName] = specification;
        }

        var supportedOps = new HashSet<string>(
            [
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Sigmoid",
                "Tanh",
                "Identity",
                "MatMul",
                "Reshape",
                "Flatten",
                "Transpose",
                "Shape",
                "Gather",
                "Unsqueeze",
                "Concat",
                "Constant",
            ],
            StringComparer.Ordinal
        );

        foreach (var opType in _torchModuleOperators.Keys)
        {
            supportedOps.Add(opType);
        }

        var nodes = new List<TorchNodeSpecification>();
        var moduleNodes = new List<TorchModuleNodeSpecification>();
        var consumedModuleInitializerNames = new HashSet<string>(StringComparer.Ordinal);
        var moduleFieldNames = new HashSet<string>(StringComparer.Ordinal);

        foreach (var node in graph.Node)
        {
            if (!string.IsNullOrWhiteSpace(node.Domain))
            {
                ReportUnsupportedTorchModule(fileName, $"node '{FormatNodeName(node)}' uses non-default ONNX domain '{node.Domain}'.", diagnostics);
                return null;
            }

            if (!supportedOps.Contains(node.OpType))
            {
                ReportUnsupportedTorchModule(fileName, $"operator '{node.OpType}' in node '{FormatNodeName(node)}' is not supported yet.", diagnostics);
                return null;
            }

            var outputs = node.Output.Where(static x => !string.IsNullOrWhiteSpace(x)).ToArray();
            if (outputs.Length != 1)
            {
                ReportUnsupportedTorchModule(fileName, $"operator '{node.OpType}' in node '{FormatNodeName(node)}' must have exactly one output for the MVP TorchModule backend.", diagnostics);
                return null;
            }

            Dictionary<string, object> attributes;
            try
            {
                attributes = node.Attribute.ToDictionary(static x => x.Name, ReadAttributeValue, StringComparer.Ordinal);
            }
            catch (NotSupportedException ex)
            {
                ReportUnsupportedTorchModule(fileName, ex.Message, diagnostics);
                return null;
            }

            var nodeSpecification = new TorchNodeSpecification(
                FormatNodeName(node),
                node.OpType,
                node.Input.Where(static x => !string.IsNullOrWhiteSpace(x)).ToImmutableArray(),
                outputs.ToImmutableArray(),
                attributes
            );
            nodes.Add(nodeSpecification);

            if (TryCreateTorchModuleNode(
                nodeSpecification,
                allInitializers,
                moduleFieldNames,
                out var moduleNode,
                out var consumedInitializers
            ))
            {
                moduleNodes.Add(moduleNode);
                foreach (var consumedInitializer in consumedInitializers)
                {
                    consumedModuleInitializerNames.Add(consumedInitializer);
                }
            }
        }

        var initializers = allInitializers
            .Values
            .Where(initializer => !consumedModuleInitializerNames.Contains(initializer.OnnxName))
            .ToImmutableArray();

        return new TorchModuleGenerationSpecification(
            runtimeInputs[0].Name,
            ToCamelIdentifier(runtimeInputs[0].Name, "input"),
            graph.Output[0].Name,
            initializers,
            moduleNodes.ToImmutableArray(),
            nodes.ToImmutableArray()
        );
    }

    private static bool TryCreateTorchModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        if (!_torchModuleOperators.TryGetValue(node.OpType, out var @operator))
        {
            module = null!;
            consumedInitializers = [];
            return false;
        }

        return @operator.TryCreateModuleNode(node, initializers, usedFieldNames, out module, out consumedInitializers);
    }

    private static bool TryCreateTorchInitializer(
        TensorProto tensor,
        HashSet<string> usedFieldNames,
        out TorchInitializerSpecification specification,
        out string error
    )
    {
        var dataType = (TensorProto.Types.DataType)tensor.DataType;
        var mapping = dataType switch
        {
            TensorProto.Types.DataType.Float => ("float", "ScalarType.Float32", true),
            TensorProto.Types.DataType.Int64 => ("long", "ScalarType.Int64", false),
            _ => default,
        };

        if (mapping == default)
        {
            specification = null!;
            error = $"initializer '{tensor.Name}' uses unsupported tensor data type '{dataType}'. The MVP TorchModule backend supports float32 and int64 initializers.";
            return false;
        }

        var baseName = "_" + ToCamelIdentifier(tensor.Name, "initializer");
        var fieldName = MakeUniqueIdentifier(baseName, usedFieldNames, "_initializer");
        var stateName = SanitizeStateName(tensor.Name);
        specification = new TorchInitializerSpecification(
            tensor.Name,
            stateName,
            fieldName,
            tensor.Dims.ToImmutableArray(),
            TryReadScalarFloatValue(tensor),
            FormatLongArray(tensor.Dims.ToArray()),
            mapping.Item1,
            mapping.Item2,
            mapping.Item3
        );
        error = string.Empty;
        return true;
    }

    private static float? TryReadScalarFloatValue(TensorProto tensor)
    {
        if ((TensorProto.Types.DataType)tensor.DataType != TensorProto.Types.DataType.Float
            || tensor.Dims.Aggregate(1L, static (product, dim) => product * dim) > 1)
        {
            return null;
        }

        var values = ReadFloatTensorValues(tensor);
        return values.Length == 1 ? values[0] : null;
    }

    private static bool TryGetTensorElementType(
        ValueInfoProto value,
        out TensorProto.Types.DataType dataType
    )
    {
        if (value.Type.ValueCase == TypeProto.ValueOneofCase.TensorType)
        {
            dataType = (TensorProto.Types.DataType)value.Type.TensorType.ElemType;
            return true;
        }

        dataType = default;
        return false;
    }

    private static void ReportUnsupportedTorchModule(
        string fileName,
        string message,
        List<Diagnostic> diagnostics
    )
    {
        diagnostics.Add(
            Diagnostic.Create(
                _unsupportedTorchModuleDescriptor,
                location: Location.None,
                fileName,
                message
            )
        );
    }

    private static object ReadAttributeValue(AttributeProto attribute)
    {
        return attribute.Type switch
        {
            AttributeProto.Types.AttributeType.Float => attribute.F,
            AttributeProto.Types.AttributeType.Int => attribute.I,
            AttributeProto.Types.AttributeType.Floats => attribute.Floats.ToArray(),
            AttributeProto.Types.AttributeType.Ints => attribute.Ints.ToArray(),
            AttributeProto.Types.AttributeType.Tensor => attribute.T,
            _ => throw new NotSupportedException($"Unsupported TorchModule attribute type '{attribute.Type}' for attribute '{attribute.Name}'."),
        };
    }

    internal static long GetLongAttribute(
        TorchNodeSpecification node,
        string name,
        long defaultValue
    )
    {
        if (!node.Attributes.TryGetValue(name, out var value))
        {
            return defaultValue;
        }

        return value switch
        {
            long longValue => longValue,
            int intValue => intValue,
            _ => Convert.ToInt64(value),
        };
    }

    internal static float GetFloatAttribute(
        TorchNodeSpecification node,
        string name,
        float defaultValue
    )
    {
        if (!node.Attributes.TryGetValue(name, out var value))
        {
            return defaultValue;
        }

        return value switch
        {
            float floatValue => floatValue,
            double doubleValue => (float)doubleValue,
            _ => Convert.ToSingle(value),
        };
    }

    internal static long[] GetLongArrayAttribute(
        TorchNodeSpecification node,
        string name,
        long[] defaultValue
    )
    {
        if (!node.Attributes.TryGetValue(name, out var value))
        {
            return defaultValue;
        }

        return value switch
        {
            long[] longValues => longValues,
            int[] intValues => intValues.Select(static x => (long)x).ToArray(),
            IEnumerable<long> longValues => longValues.ToArray(),
            _ => defaultValue,
        };
    }

    internal static string FormatLongArray(IEnumerable<long> values)
    {
        var array = values.ToArray();
        return array.Length == 0
            ? "Array.Empty<long>()"
            : $"new long[] {{ {string.Join(", ", array.Select(static x => $"{x}L"))} }}";
    }

    internal static string FormatModuleArgument(IEnumerable<long> values)
    {
        var array = values.ToArray();
        return array.Length > 0 && array.All(x => x == array[0])
            ? $"{array[0]}L"
            : FormatLongArray(array);
    }

    internal static string FormatBool(bool value)
    {
        return value ? "true" : "false";
    }

    internal static string FormatFloat(float value)
    {
        return value.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
    }

    internal static string FormatAttributeScalar(object value)
    {
        return value switch
        {
            float floatValue => $"{FormatFloat(floatValue)}f",
            double doubleValue => $"{FormatFloat((float)doubleValue)}f",
            long longValue => $"{longValue}L",
            int intValue => $"{intValue}",
            _ => throw new NotSupportedException($"Unsupported TorchModule scalar attribute value '{value}'."),
        };
    }

    internal static string FormatTensorProtoExpression(TensorProto tensor)
    {
        var dataType = (TensorProto.Types.DataType)tensor.DataType;
        var shapeExpression = FormatLongArray(tensor.Dims.ToArray());

        return dataType switch
        {
            TensorProto.Types.DataType.Float => $"torch.tensor(new float[] {{ {string.Join(", ", ReadFloatTensorValues(tensor).Select(static x => $"{FormatFloat(x)}f"))} }}, {shapeExpression}, dtype: ScalarType.Float32)",
            TensorProto.Types.DataType.Int64 => $"torch.tensor(new long[] {{ {string.Join(", ", ReadLongTensorValues(tensor).Select(static x => $"{x}L"))} }}, {shapeExpression}, dtype: ScalarType.Int64)",
            _ => throw new NotSupportedException($"Unsupported Constant tensor data type '{dataType}'."),
        };
    }

    internal static float[] ReadFloatTensorValues(TensorProto tensor)
    {
        if (tensor.FloatData.Count > 0)
        {
            return tensor.FloatData.ToArray();
        }

        if (!tensor.RawData.IsEmpty)
        {
            var bytes = tensor.RawData.ToByteArray();
            var values = new float[bytes.Length / sizeof(float)];
            for (var index = 0; index < values.Length; index++)
            {
                values[index] = BitConverter.ToSingle(bytes, index * sizeof(float));
            }

            return values;
        }

        return [];
    }

    internal static long[] ReadLongTensorValues(TensorProto tensor)
    {
        if (tensor.Int64Data.Count > 0)
        {
            return tensor.Int64Data.ToArray();
        }

        if (!tensor.RawData.IsEmpty)
        {
            var bytes = tensor.RawData.ToByteArray();
            var values = new long[bytes.Length / sizeof(long)];
            for (var index = 0; index < values.Length; index++)
            {
                values[index] = BitConverter.ToInt64(bytes, index * sizeof(long));
            }

            return values;
        }

        return [];
    }

    internal static string FormatNodeName(NodeProto node)
    {
        return string.IsNullOrWhiteSpace(node.Name)
            ? node.OpType
            : node.Name;
    }

    internal static string SanitizeStateName(string value)
    {
        var builder = new StringBuilder(value.Length);
        foreach (var character in value)
        {
            builder.Append(char.IsLetterOrDigit(character) || character == '_' || character == '.'
                ? character
                : '_');
        }

        return builder.Length == 0 ? "initializer" : builder.ToString();
    }

    internal static string ResolveProjectRelativePath(
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

    internal static bool TryGetProjectDirectory(
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

    internal static string NormalizeNamespace(string namespaceName)
    {
        var segments = namespaceName
            .Split(['.'], StringSplitOptions.RemoveEmptyEntries)
            .Select(static segment => ToPascalIdentifier(segment, "Generated"))
            .ToArray();

        return segments.Length == 0
            ? "GeneratedOnnxModels"
            : string.Join(".", segments);
    }

    internal static string ToPascalIdentifier(string? value, string fallback)
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

    internal static void AppendToken(StringBuilder builder, StringBuilder token)
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

    internal static string ToCamelIdentifier(string value)
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

    internal static string ToCamelIdentifier(string value, string fallback)
    {
        return ToCamelIdentifier(ToPascalIdentifier(value, fallback));
    }

    internal static string MakeUniqueIdentifier(
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

    internal static bool HasExternalData(TensorProto tensor)
    {
        return tensor.DataLocation == TensorProto.Types.DataLocation.External || tensor.ExternalData.Count > 0;
    }

    internal static bool TryMapElementType(
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

    internal static string GetProjectRelativePath(string projectDirectory, string filePath)
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

    internal static string NormalizeGeneratedPath(string path)
    {
        return path.Replace('/', '\\');
    }


    internal sealed record ModelAnalysisResult(
        ModelGenerationSpecification? Specification,
        ImmutableArray<Diagnostic> Diagnostics
    );

    internal sealed record ModelGenerationSpecification(
        string FileName,
        string ProjectRelativePath,
        string NamespaceName,
        string ClassName,
        ModelImportType ImportTypes,
        ImmutableArray<ModelTensorContract> Inputs,
        ImmutableArray<ModelTensorContract> Outputs,
        TorchModuleGenerationSpecification? TorchModule
    )
    {
        public string FullyQualifiedClassName => $"{NamespaceName}.{ClassName}";
    }

    [Flags]
    internal enum ModelImportType
    {
        None = 0,
        OnnxRuntimeInference = 1,
        TorchModule = 2,
    }

    internal sealed record TorchModuleGenerationSpecification(
        string InputOnnxName,
        string InputParameterName,
        string OutputOnnxName,
        ImmutableArray<TorchInitializerSpecification> Initializers,
        ImmutableArray<TorchModuleNodeSpecification> ModuleNodes,
        ImmutableArray<TorchNodeSpecification> Nodes
    );

    internal sealed record TorchInitializerSpecification(
        string OnnxName,
        string StateName,
        string FieldName,
        ImmutableArray<long> Shape,
        float? ScalarFloatValue,
        string ShapeExpression,
        string ClrTypeName,
        string ScalarTypeExpression,
        bool IsParameter
    );

    internal sealed record TorchModuleNodeSpecification(
        string NodeName,
        TorchModuleNodeKind Kind,
        string FieldName,
        string FieldTypeName,
        string ConstructorExpression,
        bool TransposeInput,
        ImmutableArray<string> LoadStatements
    );

    internal sealed record TorchForwardGroupSpecification(
        string Name,
        string FieldName,
        string TypeName,
        TorchForwardGroupKind Kind,
        ImmutableArray<TorchNodeSpecification> Nodes,
        TorchNodeSpecification? ResidualAddNode,
        string OutputOnnxName
    );

    internal enum TorchForwardGroupKind
    {
        Sequential = 1,
        Residual = 2,
    }

    internal enum TorchModuleNodeKind
    {
        Conv2d = 1,
        BatchNorm2d = 2,
        Linear = 3,
        AdaptiveAvgPool2d = 4,
        ReLU = 5,
        ReLU6 = 6,
    }

    internal sealed record TorchNodeSpecification(
        string Name,
        string OpType,
        ImmutableArray<string> Inputs,
        ImmutableArray<string> Outputs,
        IReadOnlyDictionary<string, object> Attributes
    );

    internal sealed record ModelTensorContract(
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

    internal sealed record ModelDimensionContract(
        string? NumericValueLiteral,
        string? SymbolicNameLiteral,
        bool IsUnknown
    );
}
