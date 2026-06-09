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
    private const string ADDITIONAL_FILE_CLASS_NAME_KEY = "build_metadata.additionalfiles.OnnxifyModelClassName";
    private const string ADDITIONAL_FILE_NAMESPACE_KEY = "build_metadata.additionalfiles.OnnxifyModelNamespace";
    private const string ADDITIONAL_FILE_IMPORT_TYPE_KEY = "build_metadata.additionalfiles.OnnxifyModelImportType";

    private static readonly ImmutableDictionary<string, TorchModuleLowering> _torchModuleLowerings = CreateTorchModuleLowerings();

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
                context.AddSource($"{specification.ClassName}.g.cs", GenerateSource(specification));
            }

            if (specification.ImportTypes.HasFlag(ModelImportType.TorchModule))
            {
                context.AddSource($"{specification.ClassName}TorchModule.g.cs", GenerateTorchModuleSource(specification));
            }
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

    private static string GenerateTorchModuleSource(ModelGenerationSpecification specification)
    {
        var torchModule = specification.TorchModule
            ?? throw new InvalidOperationException("TorchModule generation metadata is missing.");
        var torchClassName = $"{specification.ClassName}TorchModule";

        return $$"""
        // <auto-generated/>
        #nullable enable

        using System;
        using System.Collections.Generic;
        using System.Linq;
        using TorchSharp;
        using static TorchSharp.torch;
        using static TorchSharp.torch.nn;
        using TorchModules = TorchSharp.Modules;

        namespace {{specification.NamespaceName}}
        {
            {{Indent(BuildTorchModuleType(specification, torchModule, torchClassName), 1)}}
        }

        #nullable restore
        """;
    }

    private static string BuildTorchModuleType(
        ModelGenerationSpecification specification,
        TorchModuleGenerationSpecification torchModule,
        string torchClassName
    )
    {
        var forwardGroups = BuildForwardGroups(torchModule);
        var groupedModuleNodeNames = GetGroupedModuleNodeNames(forwardGroups);
        var fields = string.Join(
            "\n",
            forwardGroups
                .Select(static group => $"private readonly {group.TypeName} {group.FieldName};")
                .Concat(torchModule.ModuleNodes
                    .Where(module => !groupedModuleNodeNames.Contains(module.NodeName))
                .Select(static module => $"private readonly {module.FieldTypeName} {module.FieldName};")
                )
                .Concat(torchModule.Initializers.Select(static initializer =>
                    $"private Tensor {initializer.FieldName};")));

        var fieldInitializers = string.Join(
            "\n",
            forwardGroups
                .Select(static group => $"{group.FieldName} = new {group.TypeName}();")
                .Concat(torchModule.ModuleNodes
                    .Where(module => !groupedModuleNodeNames.Contains(module.NodeName))
                .Select(static module => $"{module.FieldName} = {module.ConstructorExpression};")
                )
                .Concat(torchModule.Initializers.Select(static initializer =>
                    initializer.IsParameter
                        ? $$"""
                        var {{initializer.FieldName.Substring(1)}}Parameter = new global::TorchSharp.Modules.Parameter(torch.empty({{initializer.ShapeExpression}}, dtype: {{initializer.ScalarTypeExpression}}));
                        register_parameter("{{Escape(initializer.StateName)}}", {{initializer.FieldName.Substring(1)}}Parameter);
                        {{initializer.FieldName}} = {{initializer.FieldName.Substring(1)}}Parameter;
                        """
                        : $$"""
                        var {{initializer.FieldName.Substring(1)}}Buffer = torch.empty({{initializer.ShapeExpression}}, dtype: {{initializer.ScalarTypeExpression}});
                        register_buffer("{{Escape(initializer.StateName)}}", {{initializer.FieldName.Substring(1)}}Buffer);
                        {{initializer.FieldName}} = {{initializer.FieldName.Substring(1)}}Buffer;
                        """))
                .Concat(["RegisterComponents();"]));

        var loadCalls = string.Join(
            "\n",
            forwardGroups
                .Select(static group => $"{group.FieldName}.LoadWeights(tensors);")
                .Concat(torchModule.ModuleNodes
                    .Where(module => !groupedModuleNodeNames.Contains(module.NodeName))
                .SelectMany(static module => module.LoadStatements)
                )
                .Concat(torchModule.Initializers.Select(static initializer =>
                    initializer.ClrTypeName == "float"
                        ? $"LoadFloatTensor(tensors, \"{Escape(initializer.OnnxName)}\", {initializer.FieldName});"
                        : $"LoadLongTensor(tensors, \"{Escape(initializer.OnnxName)}\", {initializer.FieldName});")));

        var forwardBody = BuildTorchForwardBody(specification, torchModule, forwardGroups);
        var forwardGroupTypes = BuildForwardGroupTypes(forwardGroups, torchModule);

        return $$"""
        {{XmlSummary($"Reconstructs the ONNX graph from '{specification.FileName}' as a TorchSharp module.")}}
        public sealed class {{torchClassName}} : torch.nn.Module<Tensor, Tensor>
        {
            public const string MODEL_PROJECT_RELATIVE_PATH = {{ToVerbatimStringLiteral(specification.ProjectRelativePath)}};

            {{Indent(fields, 1)}}

            public {{torchClassName}}() : base(nameof({{torchClassName}}))
            {
                {{Indent(fieldInitializers, 2)}}
            }

            public void LoadWeightsFromOnnx(string modelPath)
            {
                if (string.IsNullOrWhiteSpace(modelPath))
                {
                    throw new ArgumentException("Model path must be provided.", nameof(modelPath));
                }

                var model = Onnxify.OnnxModel.FromFile(
                    modelPath,
                    new Onnxify.OnnxModelBaseOptions
                    {
                        NodeTypeResolutionStrategy = Onnxify.NodeTypeResolutionStrategy.IgnoreIncompatible,
                    }
                );

                var tensors = model.Graph.Initializers.ToDictionary(static x => x.Name, StringComparer.Ordinal);
                {{Indent(loadCalls, 2)}}
            }

            public override Tensor forward(Tensor {{torchModule.InputParameterName}})
            {
                {{Indent(forwardBody, 2)}}
            }

            {{Indent(forwardGroupTypes, 1)}}

            private static Tensor CreateShapeTensor(Tensor value)
            {
                return torch.tensor(value.shape, dtype: ScalarType.Int64, device: value.device);
            }

            private static Tensor GatherTensor(Tensor input, Tensor indices, long axis)
            {
                using var flatIndices = indices.reshape(-1);
                return input.index_select(axis, flatIndices).squeeze();
            }

            private static Tensor UnsqueezeTensor(Tensor input, Tensor axes)
            {
                return UnsqueezeTensor(input, axes.data<long>().ToArray());
            }

            private static Tensor UnsqueezeTensor(Tensor input, long[] axes)
            {
                var result = input;
                foreach (var axis in axes.OrderBy(static x => x))
                {
                    result = result.unsqueeze(axis);
                }

                return result;
            }

            private static Tensor ConcatTensors(Tensor[] tensors, long axis)
            {
                if (tensors.Length == 0)
                {
                    throw new ArgumentException("At least one tensor is required.", nameof(tensors));
                }

                var device = tensors[0].device;
                var aligned = tensors.Select(x => x.device == device ? x : x.to(device)).ToArray();
                return torch.cat(aligned, axis);
            }

            private static void LoadFloatTensor(
                IReadOnlyDictionary<string, Onnxify.OnnxTensor> tensors,
                string name,
                Tensor target
            )
            {
                if (!tensors.TryGetValue(name, out var tensor))
                {
                    throw new KeyNotFoundException($"The ONNX model does not contain initializer '{name}'.");
                }

                if (tensor is not Onnxify.OnnxTensor<float> typedTensor)
                {
                    throw new InvalidOperationException($"Initializer '{name}' is not a float32 tensor.");
                }

                using var source = torch.tensor(typedTensor.Value.ToArray(), typedTensor.Shape, dtype: ScalarType.Float32, device: target.device);
                target.detach().copy_(source);
            }

            private static void LoadLongTensor(
                IReadOnlyDictionary<string, Onnxify.OnnxTensor> tensors,
                string name,
                Tensor target
            )
            {
                if (!tensors.TryGetValue(name, out var tensor))
                {
                    throw new KeyNotFoundException($"The ONNX model does not contain initializer '{name}'.");
                }

                if (tensor is not Onnxify.OnnxTensor<long> typedTensor)
                {
                    throw new InvalidOperationException($"Initializer '{name}' is not an int64 tensor.");
                }

                using var source = torch.tensor(typedTensor.Value.ToArray(), typedTensor.Shape, dtype: ScalarType.Int64, device: target.device);
                target.detach().copy_(source);
            }

            private static void LoadFloatTensorTransposed2D(
                IReadOnlyDictionary<string, Onnxify.OnnxTensor> tensors,
                string name,
                Tensor target
            )
            {
                if (!tensors.TryGetValue(name, out var tensor))
                {
                    throw new KeyNotFoundException($"The ONNX model does not contain initializer '{name}'.");
                }

                if (tensor is not Onnxify.OnnxTensor<float> typedTensor || typedTensor.Shape.Length != 2)
                {
                    throw new InvalidOperationException($"Initializer '{name}' is not a 2D float32 tensor.");
                }

                using var source = torch.tensor(typedTensor.Value.ToArray(), typedTensor.Shape, dtype: ScalarType.Float32, device: target.device).transpose(0, 1).contiguous();
                target.detach().copy_(source);
            }
        }
        """;
    }

    private static string BuildTorchForwardBody(
        ModelGenerationSpecification specification,
        TorchModuleGenerationSpecification torchModule,
        ImmutableArray<TorchForwardGroupSpecification> forwardGroups
    )
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        values[torchModule.InputOnnxName] = torchModule.InputParameterName;
        foreach (var initializer in torchModule.Initializers)
        {
            values[initializer.OnnxName] = initializer.FieldName;
        }

        var statements = new List<string>();
        var modulesByNodeName = torchModule.ModuleNodes.ToDictionary(static x => x.NodeName, StringComparer.Ordinal);
        var groupsByStartNodeName = forwardGroups.ToDictionary(static x => x.Nodes[0].Name, StringComparer.Ordinal);
        var groupedNodeNames = GetGroupedTorchNodeNames(forwardGroups);

        foreach (var node in torchModule.Nodes)
        {
            if (groupedNodeNames.Contains(node.Name))
            {
                continue;
            }

            if (groupsByStartNodeName.TryGetValue(node.Name, out var group))
            {
                var input = values[group.Nodes[0].Inputs[0]];
                var groupOutput = group.OutputOnnxName;
                var groupLocalName = ToCamelIdentifier(groupOutput, "value");
                values[groupOutput] = groupLocalName;
                statements.Add($"var {groupLocalName} = {group.FieldName}.forward({input});");
                continue;
            }

            var expression = modulesByNodeName.TryGetValue(node.Name, out var module)
                ? EmitTorchModuleNode(module, node, values)
                : EmitTorchNode(node, values);
            var output = node.Outputs[0];
            var localName = ToCamelIdentifier(output, "value");
            values[output] = localName;
            statements.Add($"var {localName} = {expression};");
        }

        if (!values.TryGetValue(torchModule.OutputOnnxName, out var outputExpression))
        {
            outputExpression = ToCamelIdentifier(torchModule.OutputOnnxName, "output");
        }

        statements.Add($"return {outputExpression};");
        return string.Join("\n", statements);
    }

    private static HashSet<string> GetGroupedModuleNodeNames(
        ImmutableArray<TorchForwardGroupSpecification> forwardGroups
    )
    {
        return new HashSet<string>(
            forwardGroups
                .SelectMany(static group => group.Nodes)
                .Select(static node => node.Name),
            StringComparer.Ordinal
        );
    }

    private static HashSet<string> GetGroupedTorchNodeNames(
        ImmutableArray<TorchForwardGroupSpecification> forwardGroups
    )
    {
        return new HashSet<string>(
            forwardGroups
                .SelectMany(static group =>
                    group.ResidualAddNode is null
                        ? group.Nodes.Skip(1).Select(static node => node.Name)
                        : group.Nodes.Skip(1).Select(static node => node.Name).Concat([group.ResidualAddNode.Name])),
            StringComparer.Ordinal
        );
    }

    private static ImmutableArray<TorchForwardGroupSpecification> BuildForwardGroups(
        TorchModuleGenerationSpecification torchModule
    )
    {
        var moduleNodeNames = new HashSet<string>(
            torchModule.ModuleNodes.Select(static x => x.NodeName),
            StringComparer.Ordinal
        );
        var inputUseCounts = torchModule.Nodes
            .SelectMany(static x => x.Inputs)
            .GroupBy(static x => x, StringComparer.Ordinal)
            .ToDictionary(static x => x.Key, static x => x.Count(), StringComparer.Ordinal);
        var groups = ImmutableArray.CreateBuilder<TorchForwardGroupSpecification>();
        var index = 0;

        while (index < torchModule.Nodes.Length)
        {
            var node = torchModule.Nodes[index];
            if (!moduleNodeNames.Contains(node.Name))
            {
                index++;
                continue;
            }

            var groupNodes = ImmutableArray.CreateBuilder<TorchNodeSpecification>();
            groupNodes.Add(node);
            var cursor = index;
            while (cursor + 1 < torchModule.Nodes.Length)
            {
                var current = torchModule.Nodes[cursor];
                var next = torchModule.Nodes[cursor + 1];
                if (!moduleNodeNames.Contains(next.Name)
                    || next.Inputs.Length == 0
                    || current.Outputs.Length != 1
                    || !string.Equals(next.Inputs[0], current.Outputs[0], StringComparison.Ordinal)
                    || !inputUseCounts.TryGetValue(current.Outputs[0], out var useCount)
                    || useCount != 1)
                {
                    break;
                }

                groupNodes.Add(next);
                cursor++;
            }

            if (groupNodes.Count >= 2)
            {
                var immutableGroupNodes = groupNodes.ToImmutable();
                var residualAddNode = TryGetResidualAddNode(
                    torchModule.Nodes,
                    cursor + 1,
                    immutableGroupNodes[0].Inputs[0],
                    immutableGroupNodes[immutableGroupNodes.Length - 1].Outputs[0]
                );
                var kind = residualAddNode is null
                    ? TorchForwardGroupKind.Sequential
                    : TorchForwardGroupKind.Residual;
                var name = kind == TorchForwardGroupKind.Sequential
                    ? $"ForwardBlock{groups.Count}"
                    : $"ResidualBlock{groups.Count}";
                groups.Add(
                    new TorchForwardGroupSpecification(
                        name,
                        "_" + ToCamelIdentifier(name, "block"),
                        $"{name}Module",
                        kind,
                        immutableGroupNodes,
                        residualAddNode,
                        residualAddNode?.Outputs[0] ?? immutableGroupNodes[immutableGroupNodes.Length - 1].Outputs[0]
                    )
                );
                index = residualAddNode is null ? cursor + 1 : cursor + 2;
                continue;
            }

            index++;
        }

        return groups.ToImmutable();
    }

    private static TorchNodeSpecification? TryGetResidualAddNode(
        ImmutableArray<TorchNodeSpecification> nodes,
        int index,
        string inputName,
        string branchOutputName
    )
    {
        if (index >= nodes.Length)
        {
            return null;
        }

        var node = nodes[index];
        if (node.OpType != "Add" || node.Inputs.Length != 2)
        {
            return null;
        }

        var leftIsResidual = string.Equals(node.Inputs[0], inputName, StringComparison.Ordinal)
            && string.Equals(node.Inputs[1], branchOutputName, StringComparison.Ordinal);
        var rightIsResidual = string.Equals(node.Inputs[1], inputName, StringComparison.Ordinal)
            && string.Equals(node.Inputs[0], branchOutputName, StringComparison.Ordinal);

        return leftIsResidual || rightIsResidual ? node : null;
    }

    private static string BuildForwardGroupTypes(
        ImmutableArray<TorchForwardGroupSpecification> forwardGroups,
        TorchModuleGenerationSpecification torchModule
    )
    {
        if (forwardGroups.Length == 0)
        {
            return string.Empty;
        }

        var modulesByNodeName = torchModule.ModuleNodes.ToDictionary(static x => x.NodeName, StringComparer.Ordinal);
        return string.Join(
            "\n\n",
            forwardGroups.Select(group =>
            {
                var modules = group.Nodes.Select(node => modulesByNodeName[node.Name]).ToImmutableArray();
                var fields = string.Join(
                    "\n",
                    modules.Select(static module => $"private readonly {module.FieldTypeName} {module.FieldName};")
                );
                var initializers = string.Join(
                    "\n",
                    modules.Select(static module => $"{module.FieldName} = {module.ConstructorExpression};")
                        .Concat(["RegisterComponents();"])
                );
                var loadStatements = string.Join(
                    "\n",
                    modules.SelectMany(static module => module.LoadStatements)
                );
                var forwardStatements = new List<string>();
                var current = "input";
                foreach (var node in group.Nodes)
                {
                    var module = modulesByNodeName[node.Name];
                    var localName = ToCamelIdentifier(node.Outputs[0], "value");
                    forwardStatements.Add($"var {localName} = {module.FieldName}.forward({current});");
                    current = localName;
                }

                forwardStatements.Add(group.Kind == TorchForwardGroupKind.Residual
                    ? $"return input + {current};"
                    : $"return {current};");

                return $$"""
                private sealed class {{group.TypeName}} : torch.nn.Module<Tensor, Tensor>
                {
                    {{Indent(fields, 1)}}

                    public {{group.TypeName}}()
                        : base(nameof({{group.TypeName}}))
                    {
                        {{Indent(initializers, 2)}}
                    }

                    public void LoadWeights(IReadOnlyDictionary<string, Onnxify.OnnxTensor> tensors)
                    {
                        {{Indent(loadStatements, 2)}}
                    }

                    public override Tensor forward(Tensor input)
                    {
                        {{Indent(string.Join("\n", forwardStatements), 2)}}
                    }
                }
                """;
            })
        );
    }

    private static string EmitTorchModuleNode(
        TorchModuleNodeSpecification module,
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, string> values
    )
    {
        var input = values[node.Inputs[0]];
        return module.Kind == TorchModuleNodeKind.Linear && module.TransposeInput
            ? $"{module.FieldName}.forward({input}.transpose(0, 1))"
            : $"{module.FieldName}.forward({input})";
    }

    private static string EmitTorchNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, string> values
    )
    {
        string Input(int index)
        {
            if (index >= node.Inputs.Length)
            {
                throw new InvalidOperationException($"Node '{node.Name}' does not have input {index}.");
            }

            return values[node.Inputs[index]];
        }

        return node.OpType switch
        {
            "Add" => $"{Input(0)} + {Input(1)}",
            "Sub" => $"{Input(0)} - {Input(1)}",
            "Mul" => $"{Input(0)} * {Input(1)}",
            "Div" => $"{Input(0)} / {Input(1)}",
            "Relu" => $"{Input(0)}.relu()",
            "Sigmoid" => $"{Input(0)}.sigmoid()",
            "Tanh" => $"{Input(0)}.tanh()",
            "Identity" => Input(0),
            "MatMul" => $"torch.matmul({Input(0)}, {Input(1)})",
            "Gemm" => EmitTorchGemm(node, Input(0), Input(1), node.Inputs.Length > 2 ? Input(2) : null),
            "Reshape" => $"{Input(0)}.reshape({Input(1)}.data<long>().ToArray())",
            "Flatten" => $"{Input(0)}.flatten({GetLongAttribute(node, "axis", 1L)})",
            "Transpose" => $"{Input(0)}.permute({FormatLongArray(GetLongArrayAttribute(node, "perm", []))})",
            "Clip" => EmitTorchClip(node, Input(0), node.Inputs.Length > 1 ? Input(1) : null, node.Inputs.Length > 2 ? Input(2) : null),
            "Conv" => EmitTorchConv(node, Input(0), Input(1), node.Inputs.Length > 2 ? Input(2) : "null"),
            "BatchNormalization" => EmitTorchBatchNormalization(node, Input(0), Input(3), Input(4), Input(1), Input(2)),
            "GlobalAveragePool" => $"torch.nn.functional.adaptive_avg_pool2d({Input(0)}, new long[] {{ 1L, 1L }})",
            "Shape" => $"CreateShapeTensor({Input(0)})",
            "Gather" => $"GatherTensor({Input(0)}, {Input(1)}, {GetLongAttribute(node, "axis", 0L)}L)",
            "Unsqueeze" => node.Inputs.Length > 1
                ? $"UnsqueezeTensor({Input(0)}, {Input(1)})"
                : $"UnsqueezeTensor({Input(0)}, {FormatLongArray(GetLongArrayAttribute(node, "axes", []))})",
            "Concat" => $"ConcatTensors(new Tensor[] {{ {string.Join(", ", node.Inputs.Select(x => values[x]))} }}, {GetLongAttribute(node, "axis", 0L)}L)",
            "Constant" => EmitTorchConstant(node),
            _ => throw new NotSupportedException($"Unsupported TorchModule node op '{node.OpType}'."),
        };
    }

    private static string EmitTorchGemm(
        TorchNodeSpecification node,
        string a,
        string b,
        string? c
    )
    {
        var alpha = GetFloatAttribute(node, "alpha", 1f);
        var beta = GetFloatAttribute(node, "beta", 1f);
        var transA = GetLongAttribute(node, "transA", 0L) != 0;
        var transB = GetLongAttribute(node, "transB", 0L) != 0;
        var left = transA ? $"{a}.transpose(0, 1)" : a;
        var right = transB ? $"{b}.transpose(0, 1)" : b;
        var expression = $"torch.matmul({left}, {right})";

        if (alpha != 1f)
        {
            expression = $"({FormatFloat(alpha)}f * {expression})";
        }

        if (c is not null)
        {
            var bias = beta == 1f ? c : $"({FormatFloat(beta)}f * {c})";
            expression = $"({expression} + {bias})";
        }

        return expression;
    }

    private static string EmitTorchClip(
        TorchNodeSpecification node,
        string input,
        string? min,
        string? max
    )
    {
        if (min is null && node.Attributes.TryGetValue("min", out var minAttribute))
        {
            min = FormatAttributeScalar(minAttribute);
        }

        if (max is null && node.Attributes.TryGetValue("max", out var maxAttribute))
        {
            max = FormatAttributeScalar(maxAttribute);
        }

        return $"{input}.clamp({min ?? "null"}, {max ?? "null"})";
    }

    private static bool IsRelu6Clip(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers
    )
    {
        if (node.OpType != "Clip")
        {
            return false;
        }

        float? min = null;
        float? max = null;
        if (node.Attributes.TryGetValue("min", out var minAttribute))
        {
            min = Convert.ToSingle(minAttribute);
        }

        if (node.Attributes.TryGetValue("max", out var maxAttribute))
        {
            max = Convert.ToSingle(maxAttribute);
        }

        if (node.Inputs.Length > 1
            && initializers.TryGetValue(node.Inputs[1], out var minInitializer))
        {
            min = minInitializer.ScalarFloatValue;
        }

        if (node.Inputs.Length > 2
            && initializers.TryGetValue(node.Inputs[2], out var maxInitializer))
        {
            max = maxInitializer.ScalarFloatValue;
        }

        return min == 0f && max == 6f;
    }

    private static string EmitTorchConv(
        TorchNodeSpecification node,
        string input,
        string weight,
        string bias
    )
    {
        var strides = GetLongArrayAttribute(node, "strides", [1L, 1L]);
        var pads = GetLongArrayAttribute(node, "pads", [0L, 0L, 0L, 0L]);
        var dilations = GetLongArrayAttribute(node, "dilations", [1L, 1L]);
        var group = GetLongAttribute(node, "group", 1L);
        var padding = pads.Length >= 2 ? pads.Take(pads.Length / 2).ToArray() : pads;

        return $"torch.nn.functional.conv2d({input}, {weight}, {bias}, {FormatLongArray(strides)}, {FormatLongArray(padding)}, {FormatLongArray(dilations)}, {group}L)";
    }

    private static string EmitTorchBatchNormalization(
        TorchNodeSpecification node,
        string input,
        string runningMean,
        string runningVar,
        string weight,
        string bias
    )
    {
        var epsilon = GetFloatAttribute(node, "epsilon", 1e-5f);
        var momentum = GetFloatAttribute(node, "momentum", 0.9f);

        return $"torch.nn.functional.batch_norm({input}, {runningMean}, {runningVar}, {weight}, {bias}, training: false, momentum: {FormatFloat(momentum)}f, eps: {FormatFloat(epsilon)}f)";
    }

    private static string EmitTorchConstant(TorchNodeSpecification node)
    {
        if (!node.Attributes.TryGetValue("value", out var value) || value is not TensorProto tensor)
        {
            throw new NotSupportedException($"Constant node '{node.Name}' does not contain a tensor value attribute.");
        }

        return FormatTensorProtoExpression(tensor);
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

        foreach (var opType in _torchModuleLowerings.Keys)
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
        if (!_torchModuleLowerings.TryGetValue(node.OpType, out var lowering))
        {
            module = null!;
            consumedInitializers = [];
            return false;
        }

        return lowering.TryLower(node, initializers, usedFieldNames, out module, out consumedInitializers);
    }

    private static ImmutableDictionary<string, TorchModuleLowering> CreateTorchModuleLowerings()
    {
        var builder = ImmutableDictionary.CreateBuilder<string, TorchModuleLowering>(StringComparer.Ordinal);
        Add("Conv", TryCreateConv2dModuleNode);
        Add("BatchNormalization", TryCreateBatchNorm2dModuleNode);
        Add("Gemm", TryCreateLinearModuleNode);
        Add("GlobalAveragePool", TryCreateAdaptiveAvgPool2dModuleNode);
        Add("Relu", TryCreateReluModuleNode);
        Add("Clip", TryCreateRelu6ModuleNode);
        return builder.ToImmutable();

        void Add(string opType, TorchModuleLoweringDelegate lowering)
        {
            builder[opType] = new TorchModuleLowering(opType, lowering);
        }
    }

    private static bool TryCreateConv2dModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        module = null!;
        consumedInitializers = [];
        if (node.Inputs.Length < 2
            || !initializers.TryGetValue(node.Inputs[1], out var weight)
            || weight.ClrTypeName != "float"
            || weight.Shape.Length != 4)
        {
            return false;
        }

        TorchInitializerSpecification? bias = null;
        if (node.Inputs.Length > 2
            && (!initializers.TryGetValue(node.Inputs[2], out bias) || bias.ClrTypeName != "float"))
        {
            return false;
        }

        var pads = GetLongArrayAttribute(node, "pads", [0L, 0L, 0L, 0L]);
        if (pads.Length != 4 || pads[0] != pads[2] || pads[1] != pads[3])
        {
            return false;
        }

        var strides = GetLongArrayAttribute(node, "strides", [1L, 1L]);
        var dilations = GetLongArrayAttribute(node, "dilations", [1L, 1L]);
        var group = GetLongAttribute(node, "group", 1L);
        var fieldName = MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "conv"), usedFieldNames, "_module");
        var constructor = $"Conv2d({weight.Shape[1] * group}, {weight.Shape[0]}, kernel_size: {FormatModuleArgument(weight.Shape.Skip(2))}, stride: {FormatModuleArgument(strides)}, padding: {FormatModuleArgument(pads.Take(2))}, dilation: {FormatModuleArgument(dilations)}, groups: {group}L, bias: {FormatBool(bias is not null)})";

        module = new TorchModuleNodeSpecification(
            node.Name,
            TorchModuleNodeKind.Conv2d,
            fieldName,
            "TorchModules.Conv2d",
            constructor,
            TransposeInput: false,
            [
                $"LoadFloatTensor(tensors, \"{Escape(weight.OnnxName)}\", {fieldName}.weight);",
                .. bias is null
                    ? []
                    : new[] { $"LoadFloatTensor(tensors, \"{Escape(bias.OnnxName)}\", {fieldName}.bias!);" },
            ]
        );
        consumedInitializers = bias is null ? [weight.OnnxName] : [weight.OnnxName, bias.OnnxName];
        return true;
    }

    private static bool TryCreateBatchNorm2dModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        module = null!;
        consumedInitializers = [];
        if (node.Inputs.Length < 5
            || !initializers.TryGetValue(node.Inputs[1], out var scale)
            || !initializers.TryGetValue(node.Inputs[2], out var bias)
            || !initializers.TryGetValue(node.Inputs[3], out var mean)
            || !initializers.TryGetValue(node.Inputs[4], out var variance)
            || scale.Shape.Length != 1
            || scale.ClrTypeName != "float"
            || bias.ClrTypeName != "float"
            || mean.ClrTypeName != "float"
            || variance.ClrTypeName != "float")
        {
            return false;
        }

        var fieldName = MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "batchNorm"), usedFieldNames, "_module");
        module = new TorchModuleNodeSpecification(
            node.Name,
            TorchModuleNodeKind.BatchNorm2d,
            fieldName,
            "TorchModules.BatchNorm2d",
            $"BatchNorm2d({scale.Shape[0]})",
            TransposeInput: false,
            [
                $"LoadFloatTensor(tensors, \"{Escape(scale.OnnxName)}\", {fieldName}.weight!);",
                $"LoadFloatTensor(tensors, \"{Escape(bias.OnnxName)}\", {fieldName}.bias!);",
                $"LoadFloatTensor(tensors, \"{Escape(mean.OnnxName)}\", {fieldName}.running_mean);",
                $"LoadFloatTensor(tensors, \"{Escape(variance.OnnxName)}\", {fieldName}.running_var);",
            ]
        );
        consumedInitializers = [scale.OnnxName, bias.OnnxName, mean.OnnxName, variance.OnnxName];
        return true;
    }

    private static bool TryCreateLinearModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        module = null!;
        consumedInitializers = [];
        if (node.Inputs.Length < 2
            || !initializers.TryGetValue(node.Inputs[1], out var weight)
            || weight.Shape.Length != 2
            || weight.ClrTypeName != "float"
            || GetFloatAttribute(node, "alpha", 1f) != 1f
            || GetFloatAttribute(node, "beta", 1f) != 1f)
        {
            return false;
        }

        var transA = GetLongAttribute(node, "transA", 0L) != 0;
        var transB = GetLongAttribute(node, "transB", 0L) != 0;
        if (transA)
        {
            return false;
        }

        TorchInitializerSpecification? bias = null;
        if (node.Inputs.Length > 2
            && (!initializers.TryGetValue(node.Inputs[2], out bias) || bias.ClrTypeName != "float" || bias.Shape.Length != 1))
        {
            return false;
        }

        var inputFeatures = transB ? weight.Shape[1] : weight.Shape[0];
        var outputFeatures = transB ? weight.Shape[0] : weight.Shape[1];
        var fieldName = MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "linear"), usedFieldNames, "_module");
        module = new TorchModuleNodeSpecification(
            node.Name,
            TorchModuleNodeKind.Linear,
            fieldName,
            "TorchModules.Linear",
            $"Linear({inputFeatures}, {outputFeatures}, hasBias: {FormatBool(bias is not null)})",
            TransposeInput: false,
            [
                transB
                    ? $"LoadFloatTensor(tensors, \"{Escape(weight.OnnxName)}\", {fieldName}.weight);"
                    : $"LoadFloatTensorTransposed2D(tensors, \"{Escape(weight.OnnxName)}\", {fieldName}.weight);",
                .. bias is null
                    ? []
                    : new[] { $"LoadFloatTensor(tensors, \"{Escape(bias.OnnxName)}\", {fieldName}.bias!);" },
            ]
        );
        consumedInitializers = bias is null ? [weight.OnnxName] : [weight.OnnxName, bias.OnnxName];
        return true;
    }

    private static bool TryCreateAdaptiveAvgPool2dModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        consumedInitializers = [];
        return TryCreateStatelessModuleNode(
            node,
            TorchModuleNodeKind.AdaptiveAvgPool2d,
            "TorchModules.AdaptiveAvgPool2d",
            "AdaptiveAvgPool2d(1)",
            usedFieldNames,
            out module
        );
    }

    private static bool TryCreateReluModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        consumedInitializers = [];
        return TryCreateStatelessModuleNode(
            node,
            TorchModuleNodeKind.ReLU,
            "TorchModules.ReLU",
            "ReLU()",
            usedFieldNames,
            out module
        );
    }

    private static bool TryCreateRelu6ModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        module = null!;
        consumedInitializers = [];
        if (!IsRelu6Clip(node, initializers))
        {
            return false;
        }

        var consumed = ImmutableArray.CreateBuilder<string>();
        if (node.Inputs.Length > 1 && initializers.ContainsKey(node.Inputs[1]))
        {
            consumed.Add(node.Inputs[1]);
        }

        if (node.Inputs.Length > 2 && initializers.ContainsKey(node.Inputs[2]))
        {
            consumed.Add(node.Inputs[2]);
        }

        consumedInitializers = consumed.ToImmutable();
        return TryCreateStatelessModuleNode(
            node,
            TorchModuleNodeKind.ReLU6,
            "TorchModules.ReLU6",
            "ReLU6()",
            usedFieldNames,
            out module
        );
    }

    private static bool TryCreateStatelessModuleNode(
        TorchNodeSpecification node,
        TorchModuleNodeKind kind,
        string fieldTypeName,
        string constructorExpression,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module
    )
    {
        module = new TorchModuleNodeSpecification(
            node.Name,
            kind,
            MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "module"), usedFieldNames, "_module"),
            fieldTypeName,
            constructorExpression,
            TransposeInput: false,
            []
        );
        return true;
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

    private static long GetLongAttribute(
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

    private static float GetFloatAttribute(
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

    private static long[] GetLongArrayAttribute(
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

    private static string FormatLongArray(IEnumerable<long> values)
    {
        var array = values.ToArray();
        return array.Length == 0
            ? "Array.Empty<long>()"
            : $"new long[] {{ {string.Join(", ", array.Select(static x => $"{x}L"))} }}";
    }

    private static string FormatModuleArgument(IEnumerable<long> values)
    {
        var array = values.ToArray();
        return array.Length > 0 && array.All(x => x == array[0])
            ? $"{array[0]}L"
            : FormatLongArray(array);
    }

    private static string FormatBool(bool value)
    {
        return value ? "true" : "false";
    }

    private static string FormatFloat(float value)
    {
        return value.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
    }

    private static string FormatAttributeScalar(object value)
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

    private static string FormatTensorProtoExpression(TensorProto tensor)
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

    private static float[] ReadFloatTensorValues(TensorProto tensor)
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

    private static long[] ReadLongTensorValues(TensorProto tensor)
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

    private static string FormatNodeName(NodeProto node)
    {
        return string.IsNullOrWhiteSpace(node.Name)
            ? node.OpType
            : node.Name;
    }

    private static string SanitizeStateName(string value)
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

    private static string ToCamelIdentifier(string value, string fallback)
    {
        return ToCamelIdentifier(ToPascalIdentifier(value, fallback));
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
        ModelImportType ImportTypes,
        ImmutableArray<ModelTensorContract> Inputs,
        ImmutableArray<ModelTensorContract> Outputs,
        TorchModuleGenerationSpecification? TorchModule
    )
    {
        public string FullyQualifiedClassName => $"{NamespaceName}.{ClassName}";
    }

    [Flags]
    private enum ModelImportType
    {
        None = 0,
        OnnxRuntimeInference = 1,
        TorchModule = 2,
    }

    private sealed record TorchModuleGenerationSpecification(
        string InputOnnxName,
        string InputParameterName,
        string OutputOnnxName,
        ImmutableArray<TorchInitializerSpecification> Initializers,
        ImmutableArray<TorchModuleNodeSpecification> ModuleNodes,
        ImmutableArray<TorchNodeSpecification> Nodes
    );

    private sealed record TorchInitializerSpecification(
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

    private sealed record TorchModuleNodeSpecification(
        string NodeName,
        TorchModuleNodeKind Kind,
        string FieldName,
        string FieldTypeName,
        string ConstructorExpression,
        bool TransposeInput,
        ImmutableArray<string> LoadStatements
    );

    private sealed record TorchForwardGroupSpecification(
        string Name,
        string FieldName,
        string TypeName,
        TorchForwardGroupKind Kind,
        ImmutableArray<TorchNodeSpecification> Nodes,
        TorchNodeSpecification? ResidualAddNode,
        string OutputOnnxName
    );

    private enum TorchForwardGroupKind
    {
        Sequential = 1,
        Residual = 2,
    }

    private sealed record TorchModuleLowering(
        string OpType,
        TorchModuleLoweringDelegate TryLower
    );

    private delegate bool TorchModuleLoweringDelegate(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    );

    private enum TorchModuleNodeKind
    {
        Conv2d = 1,
        BatchNorm2d = 2,
        Linear = 3,
        AdaptiveAvgPool2d = 4,
        ReLU = 5,
        ReLU6 = 6,
    }

    private sealed record TorchNodeSpecification(
        string Name,
        string OpType,
        ImmutableArray<string> Inputs,
        ImmutableArray<string> Outputs,
        IReadOnlyDictionary<string, object> Attributes
    );

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
