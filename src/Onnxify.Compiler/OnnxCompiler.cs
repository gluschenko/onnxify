using System.Collections;
using System.Reflection;
using Onnxify;
using Onnxify.Data.Numerics;

namespace Onnxify.Compiler;

/// <summary>
/// Public compiler session for the ONNX frontend and backend.
/// </summary>
public sealed class OnnxCompilerSession : ICompilerSession
{
    /// <inheritdoc />
    public CompilerResult<ICompilerTree> CreateTree(ICompilerSource source)
    {
        CompilerStructural.RequireNotNull(source, nameof(source));

        if (source is not OnnxCompilerSource onnxSource)
        {
            return CompilerResult<ICompilerTree>.Failure(
            [
                new CompilerDiagnostic(
                    CompilerDiagnosticCodes.Unsupported,
                    $"The ONNX compiler session cannot import source kind '{source.Kind}'.",
                    CompilerDiagnosticStage.Parse,
                    CompilerDiagnosticSeverity.Error),
            ]);
        }

        var result = OnnxCompilerFrontend.Import(
            onnxSource.Model,
            onnxSource.Document);

        return result.IsSuccess
            ? CompilerResult<ICompilerTree>.Success(result.Value!, result.Diagnostics)
            : CompilerResult<ICompilerTree>.Failure(result.Diagnostics);
    }

    /// <inheritdoc />
    public CompilerResult<TOutput> Generate<TOutput>(
        ICompilerTree tree,
        ICompilerSink<TOutput> sink
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        CompilerStructural.RequireNotNull(sink, nameof(sink));

        if (tree is not CompilerComputationTree computationTree)
        {
            return CompilerResult<TOutput>.Failure(
            [
                new CompilerDiagnostic(
                    CompilerDiagnosticCodes.Unsupported,
                    "The ONNX backend requires a CompilerComputationTree.",
                    CompilerDiagnosticStage.Emit,
                    CompilerDiagnosticSeverity.Error),
            ]);
        }

        if (sink is OnnxModelCompilerSink && typeof(TOutput) == typeof(OnnxModel))
        {
            var result = OnnxCompilerBackend.EmitModel(computationTree);
            return result.IsSuccess
                ? CompilerResult<TOutput>.Success((TOutput)(object)result.Value!, result.Diagnostics)
                : CompilerResult<TOutput>.Failure(result.Diagnostics);
        }

        if (sink is OnnxCompilerSink && typeof(TOutput) == typeof(OnnxGraph))
        {
            var result = OnnxCompilerBackend.EmitModel(computationTree);
            if (!result.IsSuccess)
            {
                return CompilerResult<TOutput>.Failure(result.Diagnostics);
            }

            return CompilerResult<TOutput>.Success(
                (TOutput)(object)result.Value!.Graph,
                result.Diagnostics);
        }

        return CompilerResult<TOutput>.Failure(
        [
            new CompilerDiagnostic(
                CompilerDiagnosticCodes.Unsupported,
                $"The ONNX compiler session cannot emit target '{sink.Kind}'.",
                CompilerDiagnosticStage.Emit,
                CompilerDiagnosticSeverity.Error),
        ]);
    }
}

/// <summary>Convenience entry points for ONNX compiler operations.</summary>
public static class Compiler
{
    /// <summary>Imports an in-memory ONNX model into the shared compiler tree.</summary>
    public static CompilerResult<CompilerComputationTree> CreateTreeFromOnnx(
        OnnxModel model,
        string? document = null
    )
    {
        return new OnnxCompilerSession().CreateTreeTyped(
            new OnnxCompilerSource(model, document));
    }

    /// <summary>Loads and imports an ONNX model from a file.</summary>
    public static CompilerResult<CompilerComputationTree> CreateTreeFromOnnx(
        string path,
        OnnxModelBaseOptions? options = null
    )
    {
        try
        {
            return CreateTreeFromOnnx(
                OnnxModel.FromFile(path, EnsureUntyped(options)),
                Path.GetFullPath(path));
        }
        catch (Exception exception)
        {
            return OnnxCompilerFrontend.Failure<CompilerComputationTree>(
                CompilerDiagnosticCodes.InvalidSource,
                $"The ONNX file could not be loaded: {exception.Message}",
                CompilerDiagnosticStage.Parse,
                Path.GetFullPath(path));
        }
    }

    /// <summary>Loads and imports an ONNX model from a stream.</summary>
    public static CompilerResult<CompilerComputationTree> CreateTreeFromOnnx(
        Stream stream,
        OnnxModelBaseOptions? options = null,
        string? document = null
    )
    {
        try
        {
            return CreateTreeFromOnnx(
                OnnxModel.FromStream(stream, EnsureUntyped(options)),
                document ?? "<memory>");
        }
        catch (Exception exception)
        {
            return OnnxCompilerFrontend.Failure<CompilerComputationTree>(
                CompilerDiagnosticCodes.InvalidSource,
                $"The ONNX stream could not be loaded: {exception.Message}",
                CompilerDiagnosticStage.Parse,
                document ?? "<memory>");
        }
    }

    /// <summary>Asynchronously loads and imports an ONNX model from a file.</summary>
    public static async Task<CompilerResult<CompilerComputationTree>> CreateTreeFromOnnxAsync(
        string path,
        OnnxModelBaseOptions? options = null,
        CancellationToken cancellationToken = default
    )
    {
        try
        {
            var model = await OnnxModel.FromFileAsync(
                path,
                EnsureUntyped(options),
                cancellationToken);
            return CreateTreeFromOnnx(model, Path.GetFullPath(path));
        }
        catch (Exception exception)
        {
            return OnnxCompilerFrontend.Failure<CompilerComputationTree>(
                CompilerDiagnosticCodes.InvalidSource,
                $"The ONNX file could not be loaded: {exception.Message}",
                CompilerDiagnosticStage.Parse,
                Path.GetFullPath(path));
        }
    }

    /// <summary>Asynchronously loads and imports an ONNX model from a stream.</summary>
    public static async Task<CompilerResult<CompilerComputationTree>> CreateTreeFromOnnxAsync(
        Stream stream,
        OnnxModelBaseOptions? options = null,
        string? document = null,
        CancellationToken cancellationToken = default
    )
    {
        try
        {
            var model = await OnnxModel.FromStreamAsync(
                stream,
                EnsureUntyped(options),
                cancellationToken);
            return CreateTreeFromOnnx(model, document ?? "<memory>");
        }
        catch (Exception exception)
        {
            return OnnxCompilerFrontend.Failure<CompilerComputationTree>(
                CompilerDiagnosticCodes.InvalidSource,
                $"The ONNX stream could not be loaded: {exception.Message}",
                CompilerDiagnosticStage.Parse,
                document ?? "<memory>");
        }
    }

    /// <summary>Emits a complete ONNX model from the shared compiler tree.</summary>
    public static CompilerResult<OnnxModel> GenerateOnnx(
        CompilerComputationTree tree,
        OnnxModelCreationOptions? options = null
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        return OnnxCompilerBackend.EmitModel(tree, options);
    }

    /// <summary>Emits only the graph from the shared compiler tree.</summary>
    public static CompilerResult<OnnxGraph> GenerateOnnxGraph(
        CompilerComputationTree tree,
        OnnxModelCreationOptions? options = null
    )
    {
        var model = GenerateOnnx(tree, options);
        return model.IsSuccess
            ? CompilerResult<OnnxGraph>.Success(model.Value!.Graph, model.Diagnostics)
            : CompilerResult<OnnxGraph>.Failure(model.Diagnostics);
    }

    private static OnnxModelBaseOptions EnsureUntyped(OnnxModelBaseOptions? options)
    {
        options ??= new OnnxModelBaseOptions();
        options.NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.PreserveUntyped;
        return options;
    }
}

internal static class OnnxCompilerSessionExtensions
{
    public static CompilerResult<CompilerComputationTree> CreateTreeTyped(
        this OnnxCompilerSession session,
        OnnxCompilerSource source
    )
    {
        var result = session.CreateTree(source);
        return result.IsSuccess
            ? CompilerResult<CompilerComputationTree>.Success(
                (CompilerComputationTree)result.Value!,
                result.Diagnostics)
            : CompilerResult<CompilerComputationTree>.Failure(result.Diagnostics);
    }
}

internal sealed class CompilerConversionException : Exception
{
    public CompilerConversionException(string code, string message)
        : base(message)
    {
        Code = code;
    }

    public string Code { get; }
}

internal static class OnnxCompilerFrontend
{
    public static CompilerResult<CompilerComputationTree> Import(
        OnnxModel model,
        string? document
    )
    {
        CompilerStructural.RequireNotNull(model, nameof(model));

        var diagnostics = new List<CompilerDiagnostic>();
        try
        {
            var tree = ImportGraph(
                model.Graph,
                document ?? "<memory>",
                diagnostics,
                outerValues: null,
                caller: null,
                includeEnvelope: true,
                model: model);

            return new CompilerResult<CompilerComputationTree>(tree, diagnostics);
        }
        catch (CompilerConversionException exception)
        {
            diagnostics.Add(new CompilerDiagnostic(
                exception.Code,
                exception.Message,
                CompilerDiagnosticStage.Normalize,
                CompilerDiagnosticSeverity.Error));
            return CompilerResult<CompilerComputationTree>.Failure(diagnostics);
        }
        catch (Exception exception)
        {
            diagnostics.Add(new CompilerDiagnostic(
                CompilerDiagnosticCodes.InvalidSource,
                $"The ONNX graph could not be converted: {exception.Message}",
                CompilerDiagnosticStage.Normalize,
                CompilerDiagnosticSeverity.Error));
            return CompilerResult<CompilerComputationTree>.Failure(diagnostics);
        }
    }

    public static CompilerResult<T> Failure<T>(
        string code,
        string message,
        CompilerDiagnosticStage stage,
        string document
    )
    {
        return CompilerResult<T>.Failure(
        [
            new CompilerDiagnostic(
                code,
                message,
                stage,
                CompilerDiagnosticSeverity.Error,
                new CompilerSourceSpan(
                    CompilerSourceSpanKind.Onnx,
                    document,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0)),
        ]);
    }

    private static CompilerComputationTree ImportGraph(
        OnnxGraph graph,
        string document,
        List<CompilerDiagnostic> diagnostics,
        IReadOnlyDictionary<string, CompilerType>? outerValues,
        string? caller,
        bool includeEnvelope,
        OnnxModel? model
    )
    {
        var values = new Dictionary<string, CompilerValue>(StringComparer.Ordinal);
        var stateTypes = new Dictionary<string, CompilerType>(StringComparer.Ordinal);
        var builder = new CompilerComputationTreeBuilder(graph.Name);
        builder.SetDocument(graph.Document);

        if (includeEnvelope && model is not null)
        {
            builder.SetModelEnvelope(new CompilerModelEnvelope(
                producerName: model.ProducerName,
                producerVersion: model.ProducerVersion,
                modelVersion: model.ModelVersion,
                irVersion: model.IrVersion,
                document: model.Document,
                domain: model.Domain,
                metadata: model.MetadataProps,
                opsetImports: model.OpsetImport.Select(x => new CompilerOpsetImport(x.Domain, x.Version))));
        }

        foreach (var metadata in graph.MetadataProps)
        {
            builder.AddMetadata(metadata.Key, metadata.Value);
        }

        foreach (var annotation in graph.QuantizationAnnotations)
        {
            builder.AddQuantizationAnnotation(new CompilerQuantizationAnnotation(
                annotation.TensorName,
                annotation.QuantParameterTensorNames));
        }

        foreach (var tensor in graph.Initializers)
        {
            var type = new CompilerTensorType(
                CompilerElementTypeMap.FromSystemType(tensor.DataType),
                tensor.Shape.Select(static x => (CompilerDimension)new CompilerFixedDimension(x)));
            var literal = ImportTensorLiteral(tensor, diagnostics, document, caller);
            builder.AddStateMember(new CompilerStateMember(
                tensor.Name,
                CompilerStateMemberKind.Initializer,
                type,
                literal,
                Span(document, 0, tensor.Name, null)));
            stateTypes[tensor.Name] = type;
        }

        foreach (var sparse in graph.SparseInitializers)
        {
            var type = new CompilerSparseTensorType(
                CompilerElementTypeMap.FromSystemType(sparse.Value.DataType),
                sparse.Shape.Select(static x => (CompilerDimension)new CompilerFixedDimension(x)));
            var literal = ImportSparseTensorLiteral(sparse, diagnostics, document, caller);
            builder.AddStateMember(new CompilerStateMember(
                sparse.Name,
                CompilerStateMemberKind.Initializer,
                type,
                literal,
                Span(document, 0, sparse.Name, null)));
            stateTypes[sparse.Name] = type;
        }

        foreach (var value in graph.Inputs)
        {
            AddValue(builder, values, value, isInput: true, isOutput: false);
        }

        foreach (var value in graph.Outputs)
        {
            AddValue(builder, values, value, isInput: false, isOutput: true);
        }

        foreach (var value in graph.IntermediateValues)
        {
            AddValue(builder, values, value, isInput: false, isOutput: false);
        }

        var allEdges = graph.Nodes
            .SelectMany(node => node.Inputs.Concat(node.Outputs))
            .Select(edge => edge.Name)
            .Where(static name => !string.IsNullOrWhiteSpace(name));

        foreach (var edgeName in allEdges.Distinct(StringComparer.Ordinal))
        {
            if (values.ContainsKey(edgeName) || stateTypes.ContainsKey(edgeName))
            {
                continue;
            }

            if (outerValues is not null && outerValues.TryGetValue(edgeName, out var outerType))
            {
                builder.AddCapture(new CompilerValue(edgeName, outerType));
                values[edgeName] = new CompilerValue(edgeName, outerType);
            }
            else
            {
                var unknown = new CompilerValue(
                    edgeName,
                    new CompilerOpaqueType("onnx", "unknown-value"));
                builder.AddIntermediateValue(unknown);
                values[edgeName] = unknown;
            }
        }

        var knownTypes = values.ToDictionary(x => x.Key, x => x.Value.Type, StringComparer.Ordinal);
        foreach (var stateType in stateTypes)
        {
            knownTypes[stateType.Key] = stateType.Value;
        }

        foreach (var pair in graph.Nodes.Select((node, index) => (node, index)))
        {
            var node = pair.node;
            var nodeName = string.IsNullOrWhiteSpace(node.Name)
                ? $"__onnx_node_{pair.index}"
                : node.Name;
            var span = Span(document, pair.index, nodeName, node.OpType);

            if (string.IsNullOrWhiteSpace(node.Name))
            {
                diagnostics.Add(new CompilerDiagnostic(
                    CompilerDiagnosticCodes.Lossy,
                    $"ONNX node at ordinal {pair.index} had no name; '{nodeName}' was synthesized.",
                    CompilerDiagnosticStage.Normalize,
                    CompilerDiagnosticSeverity.Warning,
                    span,
                    new CompilerDiagnosticContext(caller, node.OpType)));
            }

            var attributes = new List<CompilerAttribute>();
            foreach (var attribute in node.Attributes.OrderBy(x => x.Name, StringComparer.Ordinal))
            {
                try
                {
                    attributes.Add(new CompilerAttribute(
                        attribute.Name,
                        ImportAttributeLiteral(
                            attribute.GetValue(),
                            document,
                            pair.index,
                            nodeName,
                            node.OpType,
                            knownTypes,
                            diagnostics)));
                }
                catch (CompilerConversionException exception)
                {
                    diagnostics.Add(new CompilerDiagnostic(
                        exception.Code,
                        exception.Message,
                        CompilerDiagnosticStage.Normalize,
                        CompilerDiagnosticSeverity.Error,
                        span,
                        new CompilerDiagnosticContext(caller, node.OpType)));
                }
            }

            var descriptor = new CompilerOperatorDescriptor(
                node.OpType,
                node.Domain,
                CompilerOperationCapability.Unsupported,
                [string.IsNullOrEmpty(node.Domain) ? "ai.onnx" : node.Domain]);

            diagnostics.Add(new CompilerDiagnostic(
                CompilerDiagnosticCodes.Unsupported,
                $"No semantic compiler mapping is registered for ONNX operator '{node.Domain}::{node.OpType}'; the generic operation is preserved.",
                CompilerDiagnosticStage.Analyze,
                CompilerDiagnosticSeverity.Warning,
                span,
                new CompilerDiagnosticContext(
                    caller ?? (string.IsNullOrEmpty(graph.Name) ? "<graph>" : graph.Name),
                    node.OpType)));

            builder.AddOperation(new CompilerOperation(
                nodeName,
                descriptor,
                node.Inputs.Select(ToReference),
                node.Outputs.Select(ToReference),
                attributes,
                span));
        }

        return builder.Build();
    }

    private static void AddValue(
        CompilerComputationTreeBuilder builder,
        Dictionary<string, CompilerValue> values,
        OnnxValue value,
        bool isInput,
        bool isOutput
    )
    {
        var compilerValue = new CompilerValue(
            value.Name,
            OnnxCompilerTypeMap.FromOnnxType(value.Type));

        if (values.TryGetValue(value.Name, out var existing))
        {
            if (!existing.Type.Equals(compilerValue.Type))
            {
                throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Ambiguous,
                    $"ONNX value '{value.Name}' has conflicting type metadata.");
            }
        }
        else
        {
            values.Add(value.Name, compilerValue);
            if (isInput)
            {
                builder.AddInput(compilerValue);
            }
            else if (isOutput)
            {
                builder.AddOutput(compilerValue);
            }
            else
            {
                builder.AddIntermediateValue(compilerValue);
            }

            return;
        }

        if (isInput && !builder.HasInput(value.Name))
        {
            builder.AddInput(compilerValue);
        }

        if (isOutput && !builder.HasOutput(value.Name))
        {
            builder.AddOutput(compilerValue);
        }
    }

    private static CompilerValueReference ToReference(IOnnxGraphEdge edge)
    {
        return string.IsNullOrEmpty(edge.Name)
            ? new CompilerValueReference(string.Empty, isEmptyOptional: true)
            : new CompilerValueReference(edge.Name);
    }

    private static CompilerLiteral ImportAttributeLiteral(
        object value,
        string document,
        int ordinal,
        string nodeName,
        string operatorName,
        IReadOnlyDictionary<string, CompilerType> outerValues,
        List<CompilerDiagnostic> diagnostics
    )
    {
        if (value is null)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Attribute on node '{nodeName}' has a null payload.");
        }

        if (value is OnnxTensor tensor)
        {
            return ImportTensorLiteral(tensor, diagnostics, document, nodeName);
        }

        if (value is OnnxSparseTensor sparse)
        {
            return ImportSparseTensorLiteral(sparse, diagnostics, document, nodeName);
        }

        if (value is OnnxGraph graph)
        {
            var nested = ImportGraph(
                graph,
                document,
                diagnostics,
                outerValues,
                caller: $"{nodeName}::{operatorName}",
                includeEnvelope: false,
                model: null);
            return new CompilerGraphLiteral(nested);
        }

        if (value is OnnxValueType onnxType)
        {
            return new CompilerTypeLiteral(OnnxCompilerTypeMap.FromOnnxType(onnxType));
        }

        if (value is Array array)
        {
            var items = new List<CompilerLiteral>(array.Length);
            foreach (var item in array)
            {
                items.Add(ImportAttributeLiteral(
                    item!,
                    document,
                    ordinal,
                    nodeName,
                    operatorName,
                    outerValues,
                    diagnostics));
            }

            return new CompilerArrayLiteral(items, array.GetType().GetElementType()?.FullName);
        }

        return CompilerElementTypeMap.ToLiteral(value);
    }

    private static CompilerTensorLiteral ImportTensorLiteral(
        OnnxTensor tensor,
        List<CompilerDiagnostic> diagnostics,
        string document,
        string? caller
    )
    {
        var elementType = CompilerElementTypeMap.FromSystemType(tensor.DataType);
        var dimensions = tensor.Shape.Select(static x => (CompilerDimension)new CompilerFixedDimension(x));
        var values = tensor.Values.Select(value => value is null
            ? throw new CompilerConversionException(
                CompilerDiagnosticCodes.InvalidSource,
                $"Tensor '{tensor.Name}' contains a null payload value.")
            : CompilerElementTypeMap.ToScalarLiteral(elementType, value));

        if (tensor.DataLocation == OnnxTensor.TensorDataLocation.External)
        {
            diagnostics.Add(new CompilerDiagnostic(
                CompilerDiagnosticCodes.Lossy,
                $"Tensor '{tensor.Name}' used ONNX external data; the compiler preserves its values but not the external storage placement.",
                CompilerDiagnosticStage.Normalize,
                CompilerDiagnosticSeverity.Warning,
                Span(document, 0, caller, null)));
        }

        return new CompilerTensorLiteral(elementType, dimensions, values);
    }

    private static CompilerSparseTensorLiteral ImportSparseTensorLiteral(
        OnnxSparseTensor tensor,
        List<CompilerDiagnostic> diagnostics,
        string document,
        string? caller
    )
    {
        var dimensions = tensor.Shape.Select(static x => (CompilerDimension)new CompilerFixedDimension(x));
        return new CompilerSparseTensorLiteral(
            dimensions,
            ImportTensorLiteral(tensor.Value, diagnostics, document, caller),
            ImportTensorLiteral(tensor.Indices, diagnostics, document, caller));
    }

    private static CompilerSourceSpan Span(
        string document,
        int ordinal,
        string? nodeName,
        string? operatorName
    )
    {
        return new CompilerSourceSpan(
            CompilerSourceSpanKind.Onnx,
            document,
            ordinal,
            1,
            ordinal,
            0,
            ordinal,
            1,
            nodeName,
            operatorName);
    }
}

internal static class OnnxCompilerBackend
{
    public static CompilerResult<OnnxModel> EmitModel(
        CompilerComputationTree tree,
        OnnxModelCreationOptions? creationOptions = null
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        var diagnostics = new List<CompilerDiagnostic>();

        try
        {
            var envelope = tree.ModelEnvelope;
            var options = creationOptions ?? new OnnxModelCreationOptions();
            var model = OnnxModel.Create(options);

            if (envelope is not null)
            {
                model.ProducerName = envelope.ProducerName;
                model.ProducerVersion = envelope.ProducerVersion;
                model.ModelVersion = envelope.ModelVersion;
                model.IrVersion = envelope.IrVersion;
                model.Document = envelope.Document;
                model.Domain = envelope.Domain;
                model.ClearOpsetImports();
                foreach (var opset in envelope.OpsetImports)
                {
                    model.SetOpsetImport(opset.Domain, opset.Version);
                }

                foreach (var metadata in envelope.Metadata)
                {
                    model.AddMetadataProps(metadata.Key, metadata.Value);
                }
            }

            EmitGraph(tree, model.Graph, diagnostics, caller: null);

            if (diagnostics.Any(x => x.Severity == CompilerDiagnosticSeverity.Error))
            {
                return CompilerResult<OnnxModel>.Failure(diagnostics);
            }

            using var validationStream = new MemoryStream();
            model.Save(validationStream);
            validationStream.Position = 0;
            _ = OnnxModel.FromStream(
                validationStream,
                new OnnxModelBaseOptions
                {
                    NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.PreserveUntyped,
                });

            return new CompilerResult<OnnxModel>(model, diagnostics);
        }
        catch (CompilerConversionException exception)
        {
            diagnostics.Add(new CompilerDiagnostic(
                exception.Code,
                exception.Message,
                CompilerDiagnosticStage.Emit,
                CompilerDiagnosticSeverity.Error));
            return CompilerResult<OnnxModel>.Failure(diagnostics);
        }
        catch (Exception exception)
        {
            diagnostics.Add(new CompilerDiagnostic(
                CompilerDiagnosticCodes.InvalidSource,
                $"The compiler could not emit a valid ONNX model: {exception.Message}",
                CompilerDiagnosticStage.Validate,
                CompilerDiagnosticSeverity.Error));
            return CompilerResult<OnnxModel>.Failure(diagnostics);
        }
    }

    private static void EmitGraph(
        CompilerComputationTree tree,
        OnnxGraph graph,
        List<CompilerDiagnostic> diagnostics,
        string? caller
    )
    {
        graph.Name = tree.Name;
        graph.Document = tree.Document;

        foreach (var metadata in tree.Metadata)
        {
            graph.AddMetadataProps(metadata.Key, metadata.Value);
        }

        foreach (var annotation in tree.QuantizationAnnotations)
        {
            var target = graph.AddQuantizationAnnotation(new OnnxQuantizationAnnotation(annotation.TensorName));
            foreach (var parameter in annotation.ParameterTensorNames)
            {
                target.SetQuantParameterTensorName(parameter.Key, parameter.Value);
            }
        }

        var definitions = tree.Inputs
            .Concat(tree.Outputs)
            .Concat(tree.IntermediateValues)
            .GroupBy(x => x.Name, StringComparer.Ordinal)
            .Select(x => x.First())
            .ToDictionary(x => x.Name, x => x, StringComparer.Ordinal);

        var onnxValues = new Dictionary<string, OnnxValue>(StringComparer.Ordinal);
        foreach (var value in definitions.Values)
        {
            var onnxType = OnnxCompilerTypeMap.ToOnnxType(value.Type);
            var onnxValue = CreateOnnxValue(value.Name, onnxType);
            graph.AddValue(onnxValue);
            onnxValues.Add(value.Name, onnxValue);
        }

        foreach (var input in tree.Inputs)
        {
            graph.AddInput(onnxValues.TryGetValue(input.Name, out var inputValue)
                ? inputValue
                : throw new CompilerConversionException(
                    CompilerDiagnosticCodes.MissingReference,
                    $"Input '{input.Name}' was not registered in the emitted graph."));
        }

        foreach (var output in tree.Outputs)
        {
            graph.AddOutput(onnxValues.TryGetValue(output.Name, out var outputValue)
                ? outputValue
                : throw new CompilerConversionException(
                    CompilerDiagnosticCodes.MissingReference,
                    $"Output '{output.Name}' was not registered in the emitted graph."));
        }

        foreach (var initializer in tree.Initializers)
        {
            AddInitializer(graph, initializer, diagnostics, caller);
        }

        if (tree.Parameters.Count > 0 || tree.Buffers.Count > 0)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                "ONNX graphs cannot represent compiler parameters or buffers separately from initializers.");
        }

        foreach (var operation in tree.Operations)
        {
            switch (operation)
            {
                case CompilerOperation compilerOperation:
                    EmitOperation(graph, compilerOperation, diagnostics, caller);
                    break;
                case CompilerModuleCall moduleCall:
                    throw new CompilerConversionException(
                        CompilerDiagnosticCodes.Unsupported,
                        $"Module call '{moduleCall.Name}' has no direct ONNX representation.");
                default:
                    throw new CompilerConversionException(
                        CompilerDiagnosticCodes.Unsupported,
                        $"Computation step '{operation.Name}' has no ONNX representation.");
            }
        }
    }

    private static void AddInitializer(
        OnnxGraph graph,
        CompilerStateMember initializer,
        List<CompilerDiagnostic> diagnostics,
        string? caller
    )
    {
        switch (initializer.Value)
        {
            case CompilerTensorLiteral tensor:
                AddTensor(graph, initializer.Name, tensor);
                break;
            case CompilerSparseTensorLiteral sparse:
                AddSparseTensor(graph, initializer.Name, sparse);
                break;
            default:
                throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Unsupported,
                    $"Initializer '{initializer.Name}' is not a tensor or sparse tensor payload.");
        }
    }

    private static OnnxValue CreateOnnxValue(string name, OnnxValueType type)
    {
        var valueType = typeof(OnnxValue<>).MakeGenericType(type.GetType());
        return (OnnxValue)Activator.CreateInstance(valueType, name, type)!;
    }

    private static void EmitOperation(
        OnnxGraph graph,
        CompilerOperation operation,
        List<CompilerDiagnostic> diagnostics,
        string? caller
    )
    {
        if (operation.Descriptor.Capability == CompilerOperationCapability.Unsupported)
        {
            diagnostics.Add(new CompilerDiagnostic(
                CompilerDiagnosticCodes.Unsupported,
                $"Operation '{operation.Name}' is emitted through generic ONNX passthrough because no semantic mapping is registered.",
                CompilerDiagnosticStage.Emit,
                CompilerDiagnosticSeverity.Warning,
                operation.Span,
                new CompilerDiagnosticContext(caller, operation.Descriptor.Name)));
        }

        var inputs = operation.Inputs.Select(x => (IOnnxGraphEdge)new OnnxEdge(x.IsEmptyOptional ? string.Empty : x.Name));
        var outputs = operation.Outputs.Select(x => (IOnnxGraphEdge)new OnnxEdge(x.IsEmptyOptional ? string.Empty : x.Name));
        var attributes = operation.Attributes
            .Select(attribute => CreateOnnxAttribute(attribute, diagnostics, caller))
            .ToArray();

        graph.AddNode(
            operation.Name,
            operation.Descriptor.Name,
            operation.Descriptor.Domain,
            string.Empty,
            inputs,
            outputs,
            attributes);
    }

    private static OnnxAttribute CreateOnnxAttribute(
        CompilerAttribute attribute,
        List<CompilerDiagnostic> diagnostics,
        string? caller
    )
    {
        var value = ToOnnxAttributeValue(attribute.Value, diagnostics, caller);
        var valueType = value.GetType();
        var attributeType = typeof(OnnxAttribute<>).MakeGenericType(valueType);
        return (OnnxAttribute)Activator.CreateInstance(
            attributeType,
            attribute.Name,
            value)!;
    }

    private static object ToOnnxAttributeValue(
        CompilerLiteral literal,
        List<CompilerDiagnostic> diagnostics,
        string? caller
    )
    {
        switch (literal)
        {
            case CompilerGraphLiteral graphLiteral:
                var nestedModelResult = EmitModel(graphLiteral.Graph);
                if (!nestedModelResult.IsSuccess)
                {
                    throw new CompilerConversionException(
                        CompilerDiagnosticCodes.Unsupported,
                        $"Nested graph '{graphLiteral.Graph.Name}' could not be emitted.");
                }

                diagnostics.AddRange(nestedModelResult.Diagnostics);
                return nestedModelResult.Value!.Graph;
            case CompilerTensorLiteral tensorLiteral:
                return CreateTensor(tensorLiteral);
            case CompilerSparseTensorLiteral sparseLiteral:
                return CreateSparseTensor(sparseLiteral);
            case CompilerTypeLiteral typeLiteral:
                return OnnxCompilerTypeMap.ToOnnxType(typeLiteral.Value);
            case CompilerArrayLiteral arrayLiteral:
                return CreateArrayLiteral(arrayLiteral, diagnostics, caller);
            case CompilerTupleLiteral:
                throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Unsupported,
                    "Tuple literals do not have an ONNX attribute representation.");
            case CompilerScalarLiteral scalarLiteral:
                return CompilerElementTypeMap.ToOnnxValue(scalarLiteral);
            default:
                throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Unsupported,
                    $"Literal type '{literal.GetType().Name}' cannot be emitted as an ONNX attribute.");
        }
    }

    private static object CreateArrayLiteral(
        CompilerArrayLiteral literal,
        List<CompilerDiagnostic> diagnostics,
        string? caller
    )
    {
        if (literal.Items.Count == 0)
        {
            return CreateEmptyArray(literal.ItemType);
        }

        var items = literal.Items
            .Select(x => ToOnnxAttributeValue(x, diagnostics, caller))
            .ToArray();
        var itemType = items[0].GetType();
        if (items.Any(x => x.GetType() != itemType))
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Ambiguous,
                "An ONNX attribute array contains values with different runtime types.");
        }

        var array = Array.CreateInstance(itemType, items.Length);
        for (var index = 0; index < items.Length; index++)
        {
            array.SetValue(items[index], index);
        }

        return array;
    }

    private static Array CreateEmptyArray(string? itemType)
    {
        var elementType = itemType switch
        {
            "System.Single" => typeof(float),
            "System.Int64" => typeof(long),
            "System.String" => typeof(string),
            "Onnxify.OnnxTensor" => typeof(OnnxTensor),
            "Onnxify.OnnxGraph" => typeof(OnnxGraph),
            "Onnxify.OnnxSparseTensor" => typeof(OnnxSparseTensor),
            "Onnxify.OnnxValueType" => typeof(OnnxValueType),
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Ambiguous,
                $"An empty ONNX attribute array has unsupported element type '{itemType}'."),
        };

        return Array.CreateInstance(elementType, 0);
    }

    private static void AddTensor(
        OnnxGraph graph,
        string name,
        CompilerTensorLiteral tensor
    )
    {
        var array = CompilerElementTypeMap.ToArray(tensor);
        var method = typeof(OnnxGraph)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Single(x => x.Name == nameof(OnnxGraph.AddTensor) && x.IsGenericMethodDefinition);
        method.MakeGenericMethod(array.GetType().GetElementType()!)
            .Invoke(graph, [name, tensor.Dimensions.Select(ToFixedDimension).ToArray(), array]);
    }

    private static void AddSparseTensor(
        OnnxGraph graph,
        string name,
        CompilerSparseTensorLiteral sparse
    )
    {
        var values = CompilerElementTypeMap.ToArray(sparse.Values);
        var indices = CompilerElementTypeMap.ToArray(sparse.Indices);
        var method = typeof(OnnxGraph)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Single(x => x.Name == nameof(OnnxGraph.AddSparseTensor) && x.IsGenericMethodDefinition);
        method.MakeGenericMethod(
                values.GetType().GetElementType()!,
                indices.GetType().GetElementType()!)
            .Invoke(
                graph,
                [
                    name,
                    sparse.Dimensions.Select(ToFixedDimension).ToArray(),
                    sparse.Values.Dimensions.Select(ToFixedDimension).ToArray(),
                    values,
                    sparse.Indices.Dimensions.Select(ToFixedDimension).ToArray(),
                    indices,
                ]);
    }

    private static OnnxSparseTensor CreateSparseTensor(CompilerSparseTensorLiteral sparse)
    {
        var model = OnnxModel.Create();
        AddSparseTensor(model.Graph, "attribute", sparse);
        return model.Graph.SparseInitializers.Single();
    }

    private static OnnxTensor CreateTensor(CompilerTensorLiteral tensor)
    {
        var model = OnnxModel.Create();
        AddTensor(model.Graph, "attribute", tensor);
        return model.Graph.Initializers.Single();
    }

    private static long ToFixedDimension(CompilerDimension dimension)
    {
        return dimension is CompilerFixedDimension fixedDimension
            ? fixedDimension.Value
            : throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                "ONNX tensor payload dimensions must be fixed.");
    }
}

internal static class OnnxCompilerTypeMap
{
    public static CompilerType FromOnnxType(OnnxValueType type)
    {
        return type switch
        {
            OnnxTensorType tensor => new CompilerTensorType(
                CompilerElementTypeMap.FromSystemType(tensor.Type),
                tensor.Shape?.Dimensions.Select(ToCompilerDimension),
                tensor.Denotation),
            OnnxSparseTensorType sparse => new CompilerSparseTensorType(
                CompilerElementTypeMap.FromSystemType(sparse.Type),
                sparse.Shape?.Dimensions.Select(ToCompilerDimension),
                sparse.Denotation),
            OnnxSequenceType sequence => new CompilerSequenceType(
                FromOnnxType(sequence.ElementType),
                sequence.Denotation),
            OnnxOptionalType optional => new CompilerOptionalType(
                FromOnnxType(optional.ElementType),
                optional.Denotation),
            OnnxMapType map => new CompilerMapType(
                CompilerElementTypeMap.FromSystemType(map.KeyType),
                FromOnnxType(map.ValueType),
                map.Denotation),
            OnnxOpaqueType opaque => new CompilerOpaqueType(
                opaque.Domain,
                opaque.Name,
                opaque.Denotation),
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"ONNX value type '{type.GetType().Name}' is not supported by the compiler IR."),
        };
    }

    public static OnnxValueType ToOnnxType(CompilerType type)
    {
        return type switch
        {
            CompilerTensorType tensor => new OnnxTensorType(
                CompilerElementTypeMap.ToSystemType(tensor.ElementType),
                tensor.Dimensions is null
                    ? null
                    : OnnxTensorShape.Create(tensor.Dimensions.Select(ToOnnxDimension)),
                tensor.Denotation),
            CompilerSparseTensorType sparse => new OnnxSparseTensorType(
                CompilerElementTypeMap.ToSystemType(sparse.ElementType),
                sparse.Dimensions is null
                    ? null
                    : OnnxTensorShape.Create(sparse.Dimensions.Select(ToOnnxDimension)),
                sparse.Denotation),
            CompilerSequenceType sequence => new OnnxSequenceType(
                ToOnnxType(sequence.ElementType),
                sequence.Denotation),
            CompilerOptionalType optional => new OnnxOptionalType(
                ToOnnxType(optional.ElementType),
                optional.Denotation),
            CompilerMapType map => new OnnxMapType(
                CompilerElementTypeMap.ToSystemType(map.KeyType),
                ToOnnxType(map.ValueType),
                map.Denotation),
            CompilerOpaqueType opaque => new OnnxOpaqueType(
                opaque.Domain,
                opaque.Name,
                opaque.Denotation),
            CompilerScalarType => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                "ONNX value-info does not support scalar values without a tensor container."),
            CompilerTupleType => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                "ONNX value-info does not support tuple values."),
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Compiler type '{type.GetType().Name}' cannot be emitted to ONNX."),
        };
    }

    private static CompilerDimension ToCompilerDimension(OnnxDimension dimension)
    {
        return dimension.GetValue() switch
        {
            long value => new CompilerFixedDimension(value, dimension.Denotation),
            string value => new CompilerSymbolicDimension(value, dimension.Denotation),
            _ => new CompilerUnknownDimension(dimension.Denotation),
        };
    }

    private static OnnxDimension ToOnnxDimension(CompilerDimension dimension)
    {
        return dimension switch
        {
            CompilerFixedDimension fixedDimension => new OnnxDimension<long>(
                fixedDimension.Value,
                fixedDimension.Denotation),
            CompilerSymbolicDimension symbolicDimension => new OnnxDimension<string>(
                symbolicDimension.Name,
                symbolicDimension.Denotation),
            CompilerUnknownDimension unknownDimension => new OnnxDimensionNone(unknownDimension.Denotation),
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Dimension '{dimension.GetType().Name}' cannot be emitted to ONNX."),
        };
    }
}

internal static class CompilerElementTypeMap
{
    public static CompilerElementType FromSystemType(Type type)
    {
        var fullName = type.FullName ?? type.Name;
        return fullName switch
        {
            "System.Boolean" => CompilerElementType.Boolean,
            "System.SByte" => CompilerElementType.Int8,
            "System.Byte" => CompilerElementType.UInt8,
            "System.Int16" => CompilerElementType.Int16,
            "System.UInt16" => CompilerElementType.UInt16,
            "System.Int32" => CompilerElementType.Int32,
            "System.UInt32" => CompilerElementType.UInt32,
            "System.Int64" => CompilerElementType.Int64,
            "System.UInt64" => CompilerElementType.UInt64,
            "System.Half" => CompilerElementType.Float16,
            "System.Single" => CompilerElementType.Float32,
            "System.Double" => CompilerElementType.Float64,
            "System.String" => CompilerElementType.String,
            "Onnxify.Data.Numerics.Complex64" => CompilerElementType.Complex64,
            "Onnxify.Data.Numerics.Complex128" => CompilerElementType.Complex128,
            "Onnxify.Data.Numerics.BFloat16" => CompilerElementType.BFloat16,
            "Onnxify.Data.Numerics.Float8E4M3FN" => CompilerElementType.Float8E4M3FN,
            "Onnxify.Data.Numerics.Float8E4M3FNUZ" => CompilerElementType.Float8E4M3FNUZ,
            "Onnxify.Data.Numerics.Float8E5M2" => CompilerElementType.Float8E5M2,
            "Onnxify.Data.Numerics.Float8E5M2FNUZ" => CompilerElementType.Float8E5M2FNUZ,
            "Onnxify.Data.Numerics.Float4E2M1" => CompilerElementType.Float4E2M1,
            "Onnxify.Data.Numerics.Float8E8M0" => CompilerElementType.Float8E8M0,
            "Onnxify.Data.Numerics.UInt4" => CompilerElementType.UInt4,
            "Onnxify.Data.Numerics.Int4" => CompilerElementType.Int4,
            "Onnxify.Data.Numerics.UInt2" => CompilerElementType.UInt2,
            "Onnxify.Data.Numerics.Int2" => CompilerElementType.Int2,
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"CLR tensor element type '{type.FullName}' is not supported by the compiler IR."),
        };
    }

    public static Type ToSystemType(CompilerElementType elementType)
    {
        return elementType switch
        {
            CompilerElementType.Boolean => typeof(bool),
            CompilerElementType.Int8 => typeof(sbyte),
            CompilerElementType.UInt8 => typeof(byte),
            CompilerElementType.Int16 => typeof(short),
            CompilerElementType.UInt16 => typeof(ushort),
            CompilerElementType.Int32 => typeof(int),
            CompilerElementType.UInt32 => typeof(uint),
            CompilerElementType.Int64 => typeof(long),
            CompilerElementType.UInt64 => typeof(ulong),
            CompilerElementType.Float32 => typeof(float),
            CompilerElementType.Float64 => typeof(double),
            CompilerElementType.String => typeof(string),
            CompilerElementType.Float16 => FindType("System.Half"),
            CompilerElementType.Complex64 => typeof(Complex64),
            CompilerElementType.Complex128 => typeof(Complex128),
            CompilerElementType.BFloat16 => typeof(BFloat16),
            CompilerElementType.Float8E4M3FN => typeof(Float8E4M3FN),
            CompilerElementType.Float8E4M3FNUZ => typeof(Float8E4M3FNUZ),
            CompilerElementType.Float8E5M2 => typeof(Float8E5M2),
            CompilerElementType.Float8E5M2FNUZ => typeof(Float8E5M2FNUZ),
            CompilerElementType.Float4E2M1 => typeof(Float4E2M1),
            CompilerElementType.Float8E8M0 => typeof(Float8E8M0),
            CompilerElementType.UInt4 => typeof(UInt4),
            CompilerElementType.Int4 => typeof(Int4),
            CompilerElementType.UInt2 => typeof(UInt2),
            CompilerElementType.Int2 => typeof(Int2),
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Compiler element type '{elementType}' is not supported by Onnxify."),
        };
    }

    public static CompilerLiteral ToLiteral(object value)
    {
        return value switch
        {
            bool boolean => new CompilerBooleanLiteral(boolean),
            sbyte signedByte => new CompilerSignedIntegerLiteral(CompilerElementType.Int8, signedByte),
            byte unsignedByte => new CompilerUnsignedIntegerLiteral(CompilerElementType.UInt8, unsignedByte),
            short signedShort => new CompilerSignedIntegerLiteral(CompilerElementType.Int16, signedShort),
            ushort unsignedShort => new CompilerUnsignedIntegerLiteral(CompilerElementType.UInt16, unsignedShort),
            int signedInt => new CompilerSignedIntegerLiteral(CompilerElementType.Int32, signedInt),
            uint unsignedInt => new CompilerUnsignedIntegerLiteral(CompilerElementType.UInt32, unsignedInt),
            long signedLong => new CompilerSignedIntegerLiteral(CompilerElementType.Int64, signedLong),
            ulong unsignedLong => new CompilerUnsignedIntegerLiteral(CompilerElementType.UInt64, unsignedLong),
            float single => new CompilerFloatingPointLiteral(CompilerElementType.Float32, single),
            double doubleValue => new CompilerFloatingPointLiteral(CompilerElementType.Float64, doubleValue),
            string text => new CompilerStringLiteral(text),
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Attribute scalar type '{value.GetType().FullName}' is not supported."),
        };
    }

    public static CompilerScalarLiteral ToScalarLiteral(
        CompilerElementType elementType,
        object value
    )
    {
        if (elementType is CompilerElementType.Complex64 or CompilerElementType.Complex128)
        {
            var real = Convert.ToDouble(value.GetType().GetProperty("Real")?.GetValue(value));
            var imaginary = Convert.ToDouble(value.GetType().GetProperty("Imaginary")?.GetValue(value));
            return new CompilerComplexLiteral(elementType, real, imaginary);
        }

        if (elementType is CompilerElementType.BFloat16
            or CompilerElementType.Float8E4M3FN
            or CompilerElementType.Float8E4M3FNUZ
            or CompilerElementType.Float8E5M2
            or CompilerElementType.Float8E5M2FNUZ
            or CompilerElementType.Float4E2M1
            or CompilerElementType.Float8E8M0
            or CompilerElementType.UInt4
            or CompilerElementType.Int4
            or CompilerElementType.UInt2
            or CompilerElementType.Int2)
        {
            var raw = value.GetType().GetProperty("Value")?.GetValue(value);
            return new CompilerPackedScalarLiteral(elementType, Convert.ToUInt64(raw));
        }

        return elementType switch
        {
            CompilerElementType.Boolean => new CompilerBooleanLiteral(Convert.ToBoolean(value)),
            CompilerElementType.Int8 or CompilerElementType.Int16 or CompilerElementType.Int32 or CompilerElementType.Int64
                => new CompilerSignedIntegerLiteral(elementType, Convert.ToInt64(value)),
            CompilerElementType.UInt8 or CompilerElementType.UInt16 or CompilerElementType.UInt32 or CompilerElementType.UInt64
                => new CompilerUnsignedIntegerLiteral(elementType, Convert.ToUInt64(value)),
            CompilerElementType.Float16 or CompilerElementType.Float32 or CompilerElementType.Float64
                => new CompilerFloatingPointLiteral(elementType, Convert.ToDouble(value)),
            CompilerElementType.String => new CompilerStringLiteral(Convert.ToString(value) ?? string.Empty),
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Tensor element type '{elementType}' is not supported."),
        };
    }

    public static object ToOnnxValue(CompilerScalarLiteral literal)
    {
        return literal switch
        {
            CompilerBooleanLiteral boolean => boolean.Value,
            CompilerSignedIntegerLiteral signed => Convert.ChangeType(signed.Value, ToSystemType(signed.ElementType))!,
            CompilerUnsignedIntegerLiteral unsigned => Convert.ChangeType(unsigned.Value, ToSystemType(unsigned.ElementType))!,
            CompilerFloatingPointLiteral floating => ConvertFloating(floating),
            CompilerComplexLiteral complex => CreateComplex(complex),
            CompilerStringLiteral text => text.Value,
            CompilerPackedScalarLiteral packed => CreatePacked(packed),
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Scalar literal '{literal.GetType().Name}' cannot be emitted to ONNX."),
        };
    }

    public static Array ToArray(CompilerTensorLiteral tensor)
    {
        var type = ToSystemType(tensor.ElementType);
        var array = Array.CreateInstance(type, tensor.Values.Count);
        for (var index = 0; index < tensor.Values.Count; index++)
        {
            array.SetValue(ToOnnxValue(tensor.Values[index]), index);
        }

        return array;
    }

    private static object ConvertFloating(CompilerFloatingPointLiteral literal)
    {
        return literal.ElementType switch
        {
            CompilerElementType.Float16 => Convert.ChangeType(literal.Value, ToSystemType(literal.ElementType))!,
            CompilerElementType.Float32 => (float)literal.Value,
            CompilerElementType.Float64 => literal.Value,
            _ => throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Floating literal element type '{literal.ElementType}' is not supported."),
        };
    }

    private static object CreateComplex(CompilerComplexLiteral literal)
    {
        var type = ToSystemType(literal.ElementType);
        return Activator.CreateInstance(
            type,
            literal.ElementType == CompilerElementType.Complex64
                ? (object)(float)literal.Real
                : literal.Real,
            literal.ElementType == CompilerElementType.Complex64
                ? (object)(float)literal.Imaginary
                : literal.Imaginary)!;
    }

    private static object CreatePacked(CompilerPackedScalarLiteral literal)
    {
        var type = ToSystemType(literal.ElementType);
        var encodedFactory = type.GetMethod(
            "FromEncoded",
            BindingFlags.Public | BindingFlags.Static);
        if (encodedFactory is not null)
        {
            var encodedType = encodedFactory.GetParameters().Single().ParameterType;
            var encoded = Convert.ChangeType(literal.EncodedValue, encodedType);
            return encodedFactory.Invoke(null, [encoded])!;
        }

        var constructorArgument = literal.ElementType switch
        {
            CompilerElementType.Int4 or CompilerElementType.Int2 => (object)(sbyte)literal.EncodedValue,
            _ => (object)(byte)literal.EncodedValue,
        };

        var constructor = type.GetConstructor([constructorArgument.GetType()]);
        if (constructor is null)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Lossy,
                $"The encoded payload for '{literal.ElementType}' cannot be reconstructed by the current Onnxify numeric wrapper.");
        }

        return constructor.Invoke([constructorArgument]);
    }

    private static Type FindType(string name)
    {
        return Type.GetType(name)
            ?? throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Runtime type '{name}' is unavailable on this target framework.");
    }
}
