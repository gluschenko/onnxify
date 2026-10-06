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
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: $"The ONNX compiler session cannot import source kind '{source.Kind}'.",
                    stage: CompilerDiagnosticStage.Parse,
                    severity: CompilerDiagnosticSeverity.Error),
            ]);
        }

        var importResult = OnnxCompilerFrontend.Import(
            onnxSource.Model,
            onnxSource.Document);
        var resultTree = CompilerResultMapper.Map(
            source: importResult,
            projection: static computationTree => (ICompilerTree)computationTree,
            missingValueStage: CompilerDiagnosticStage.Normalize,
            missingValueMessage: "The ONNX frontend reported success without producing a computation tree.");
        return resultTree;
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
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: "The ONNX backend requires a CompilerComputationTree.",
                    stage: CompilerDiagnosticStage.Emit,
                    severity: CompilerDiagnosticSeverity.Error),
            ]);
        }

        if (sink is OnnxModelCompilerSink && typeof(TOutput) == typeof(OnnxModel))
        {
            var emittedModel = OnnxCompilerBackend.EmitModel(computationTree);
            var resultModel = CompilerResultMapper.Map(
                source: emittedModel,
                projection: static model => (TOutput)(object)model,
                missingValueStage: CompilerDiagnosticStage.Emit,
                missingValueMessage: "The ONNX backend reported success without producing a model.");
            return resultModel;
        }

        if (sink is OnnxCompilerSink && typeof(TOutput) == typeof(OnnxGraph))
        {
            var emittedModel = OnnxCompilerBackend.EmitModel(computationTree);
            var resultGraph = CompilerResultMapper.Map(
                source: emittedModel,
                projection: static model => (TOutput)(object)model.Graph,
                missingValueStage: CompilerDiagnosticStage.Emit,
                missingValueMessage: "The ONNX backend reported success without producing a model.");
            return resultGraph;
        }

        return CompilerResult<TOutput>.Failure(
        [
            new CompilerDiagnostic(
                code: CompilerDiagnosticCodes.Unsupported,
                message: $"The ONNX compiler session cannot emit target '{sink.Kind}'.",
                stage: CompilerDiagnosticStage.Emit,
                severity: CompilerDiagnosticSeverity.Error),
        ]);
    }
}

/// <summary>Convenience entry points for ONNX compiler operations.</summary>
public static class Compiler
{
    /// <summary>Exports an opaque consumer module through registered module exporters.</summary>
    public static CompilerResult<CompilerComputationTree> CreateTreeFromModule(
        object module,
        CompilerOptions options
    )
    {
        CompilerStructural.RequireNotNull(module, nameof(module));
        CompilerStructural.RequireNotNull(options, nameof(options));
        var handlers = options.Extensions.Get<ICompilerModuleExporter>();
        CompilerResult<CompilerComputationTree>? result = null;
        var delegatedWithOptions = false;
        if (handlers.Count > 0 && handlers.Count(x => x.Priority == handlers[0].Priority) > 1)
        {
            result = ExtensionAmbiguity(options, $"Multiple {nameof(ICompilerModuleExporter)} handlers have priority {handlers[0].Priority}.");
        }
        else
        {
            foreach (var handler in handlers)
            {
                try
                {
                    if (handler.TryExport(module, out var source) && source is not null)
                    {
                        result = source switch
                        {
                            CSharpTorchSharpSource csharp => DelegateTorchSharp(csharp),
                            OnnxCompilerSource onnx => DelegateOnnx(onnx),
                            _ => CompilerResult<CompilerComputationTree>.Failure([new CompilerDiagnostic(
                                CompilerDiagnosticCodes.Unsupported,
                                $"Module exporter returned unsupported source kind '{source.Kind}'.",
                                CompilerDiagnosticStage.Parse,
                                CompilerDiagnosticSeverity.Error)]),
                        };
                        break;
                    }
                }
                catch (Exception exception)
                {
                    var diagnostic = ExtensionFailure(options, exception);
                    if (options.ErrorMode == CompilerErrorMode.Strict)
                    {
                        result = CompilerResult<CompilerComputationTree>.Failure([diagnostic]);
                        break;
                    }
                    options.Report([diagnostic]);
                }
            }
        }

        result ??= CompilerResult<CompilerComputationTree>.Failure([new CompilerDiagnostic(
            CompilerDiagnosticCodes.Unsupported,
            "No registered module exporter accepted the supplied module.",
            CompilerDiagnosticStage.Parse,
            CompilerDiagnosticSeverity.Error)]);
        return delegatedWithOptions ? result : options.Observe(result);

        CompilerResult<CompilerComputationTree> DelegateTorchSharp(CSharpTorchSharpSource source)
        {
            delegatedWithOptions = true;
            return CreateTreeFromTorchSharp(source, options);
        }

        CompilerResult<CompilerComputationTree> DelegateOnnx(OnnxCompilerSource source)
        {
            delegatedWithOptions = true;
            return CreateTreeFromOnnx(source.Model, source.Document, options);
        }
    }

    /// <summary>Collects metadata using registered compiler metadata providers in priority order.</summary>
    public static IReadOnlyDictionary<string, string> GetMetadata(
        ICompilerSource source,
        CompilerOptions options
    )
    {
        CompilerStructural.RequireNotNull(source, nameof(source));
        CompilerStructural.RequireNotNull(options, nameof(options));
        var handlers = options.Extensions.Get<ICompilerMetadataProvider>();
        if (handlers.Count > 0 && handlers.Count(x => x.Priority == handlers[0].Priority) > 1)
        {
            var diagnostic = new CompilerDiagnostic(CompilerDiagnosticCodes.Ambiguous,
                "Multiple metadata providers have the same highest priority.", CompilerDiagnosticStage.Analyze,
                options.ErrorMode == CompilerErrorMode.Strict ? CompilerDiagnosticSeverity.Error : CompilerDiagnosticSeverity.Warning);
            options.Report([diagnostic]);
            return new Dictionary<string, string>();
        }

        foreach (var handler in handlers)
        {
            try
            {
                if (handler.TryProvideMetadata(source, out var metadata))
                {
                    return metadata.ToDictionary(
                        static pair => pair.Key,
                        static pair => pair.Value,
                        StringComparer.Ordinal);
                }
            }
            catch (Exception exception)
            {
                options.Report([ExtensionFailure(options, exception)]);
                if (options.ErrorMode == CompilerErrorMode.Strict)
                {
                    break;
                }
            }
        }

        return new Dictionary<string, string>();
    }

    /// <summary>Imports C# TorchSharp source into the shared compiler tree.</summary>
    public static CompilerResult<CompilerComputationTree> CreateTreeFromTorchSharp(
        CSharpTorchSharpSource source
    )
    {
        CompilerStructural.RequireNotNull(source, nameof(source));
        var result = new CSharpCompilerSession().CreateTree(source);
        if (!result.IsSuccess)
        {
            return CompilerResult<CompilerComputationTree>.Failure(result.Diagnostics);
        }

        if (result.Value is not CompilerComputationTree computationTree)
        {
            return CompilerResult<CompilerComputationTree>.Failure(
                result.Diagnostics.Append(new CompilerDiagnostic(
                    code: CompilerDiagnosticCodes.InvalidSource,
                    message: "The C# frontend reported success without producing a computation tree.",
                    stage: CompilerDiagnosticStage.Analyze,
                    severity: CompilerDiagnosticSeverity.Error)));
        }

        var treeResult = CompilerResult<CompilerComputationTree>.Success(computationTree, result.Diagnostics);
        return treeResult;
    }

    /// <summary>Imports TorchSharp source using registered compiler extensions and observation options.</summary>
    public static CompilerResult<CompilerComputationTree> CreateTreeFromTorchSharp(
        CSharpTorchSharpSource source,
        CompilerOptions options
    )
    {
        CompilerStructural.RequireNotNull(source, nameof(source));
        CompilerStructural.RequireNotNull(options, nameof(options));
        var result = TryExtensionTree<ICompilerSourceScanner>(source, options, static (extension, input) => extension.Scan(input))
            ?? TryExtensionTree<ICompilerMethodLowering>(source, options, static (extension, input) => extension.Lower(input))
            ?? (source.Module is not null && !options.EnableRuntimeReflection
                ? CompilerResult<CompilerComputationTree>.Failure([new CompilerDiagnostic(
                    CompilerDiagnosticCodes.Unsupported,
                    "Runtime metadata reflection is disabled, but this source requires a compiled module descriptor.",
                    CompilerDiagnosticStage.Parse,
                    CompilerDiagnosticSeverity.Error)])
                : CreateTreeFromTorchSharp(source));
        return options.Observe(result);
    }

    /// <summary>Generates compiler-owned C# TorchSharp module source.</summary>
    public static CompilerResult<string> GenerateCSharp(
        CompilerComputationTree tree,
        CompilerCSharpGenerationOptions? options = null
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        var result = CSharpCompilerBackend.Generate(tree, options);
        return result;
    }

    /// <summary>Generates C# using registered compiler extensions and observation options.</summary>
    public static CompilerResult<string> GenerateCSharp(
        CompilerComputationTree tree,
        CompilerCSharpGenerationOptions? generationOptions,
        CompilerOptions compilerOptions
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        CompilerStructural.RequireNotNull(compilerOptions, nameof(compilerOptions));
        var extensionResult = TryExtensionPrint(tree, CompilerTargetKind.CSharp, compilerOptions);
        var result = extensionResult is CompilerResult<object> printed && printed.Value is string source
            ? CompilerResult<string>.Success(source, printed.Diagnostics)
            : extensionResult is CompilerResult<object> failed && !failed.IsSuccess
                ? CompilerResult<string>.Failure(failed.Diagnostics)
                : GenerateCSharp(tree, generationOptions);
        return compilerOptions.Observe(result);
    }

    /// <summary>Imports an in-memory ONNX model into the shared compiler tree.</summary>
    public static CompilerResult<CompilerComputationTree> CreateTreeFromOnnx(
        OnnxModel model,
        string? document = null
    )
    {
        return new OnnxCompilerSession().CreateTreeTyped(
            new OnnxCompilerSource(model, document));
    }

    /// <summary>Imports an ONNX model using compiler extensions and observation options.</summary>
    public static CompilerResult<CompilerComputationTree> CreateTreeFromOnnx(
        OnnxModel model,
        string? document,
        CompilerOptions options
    )
    {
        CompilerStructural.RequireNotNull(model, nameof(model));
        CompilerStructural.RequireNotNull(options, nameof(options));
        var source = new OnnxCompilerSource(model, document);
        var result = TryExtensionTree<ICompilerSourceScanner>(source, options, static (extension, input) => extension.Scan(input))
            ?? TryExtensionTree<ICompilerMethodLowering>(source, options, static (extension, input) => extension.Lower(input))
            ?? CreateTreeFromOnnx(model, document);
        return options.Observe(result);
    }

    private static CompilerResult<CompilerComputationTree>? TryExtensionTree<TExtension>(
        ICompilerSource source,
        CompilerOptions options,
        Func<TExtension, ICompilerSource, CompilerResult<CompilerComputationTree>?> invoke
    ) where TExtension : class, ICompilerExtension
    {
        var handlers = options.Extensions.Get<TExtension>();
        if (handlers.Count == 0)
        {
            return null;
        }

        var highestPriority = handlers[0].Priority;
        var candidates = handlers.Where(handler => handler.Priority == highestPriority).ToArray();
        if (candidates.Length > 1)
        {
            return ExtensionAmbiguity(options, $"Multiple {typeof(TExtension).Name} handlers have priority {highestPriority}.");
        }

        foreach (var handler in handlers)
        {
            try
            {
                var result = invoke(handler, source);
                if (result is not null)
                {
                    return result;
                }
            }
            catch (Exception exception)
            {
                var diagnostic = ExtensionFailure(options, exception);
                if (options.ErrorMode == CompilerErrorMode.Strict)
                {
                    return CompilerResult<CompilerComputationTree>.Failure([diagnostic]);
                }
                options.Report([diagnostic]);
            }
        }

        return null;
    }

    private static CompilerResult<object>? TryExtensionPrint(
        CompilerComputationTree tree,
        CompilerTargetKind target,
        CompilerOptions options
    )
    {
        var handlers = options.Extensions.Get<ICompilerOutputPrinter>();
        if (handlers.Count == 0)
        {
            return null;
        }

        if (handlers.Count(x => x.Priority == handlers[0].Priority) > 1)
        {
            var diagnostic = new CompilerDiagnostic(CompilerDiagnosticCodes.Ambiguous,
                "Multiple output printers have the same highest priority; the built-in printer will be used.",
                CompilerDiagnosticStage.Emit,
                options.ErrorMode == CompilerErrorMode.Strict ? CompilerDiagnosticSeverity.Error : CompilerDiagnosticSeverity.Warning);
            if (options.ErrorMode == CompilerErrorMode.Strict)
            {
                return CompilerResult<object>.Failure([diagnostic]);
            }
            options.Report([diagnostic]);
            return null;
        }

        foreach (var handler in handlers)
        {
            try
            {
                var result = handler.Print(tree, target);
                if (result is not null)
                {
                    return result;
                }
            }
            catch (Exception exception)
            {
                var diagnostic = ExtensionFailure(options, exception);
                if (options.ErrorMode == CompilerErrorMode.Strict)
                {
                    return CompilerResult<object>.Failure([diagnostic]);
                }
                options.Report([diagnostic]);
            }
        }

        return null;
    }

    private static CompilerResult<CompilerComputationTree>? ExtensionAmbiguity(CompilerOptions options, string message)
    {
        var diagnostic = new CompilerDiagnostic(CompilerDiagnosticCodes.Ambiguous, message,
            CompilerDiagnosticStage.Analyze,
            options.ErrorMode == CompilerErrorMode.Strict ? CompilerDiagnosticSeverity.Error : CompilerDiagnosticSeverity.Warning);
        if (options.ErrorMode == CompilerErrorMode.Strict)
        {
            return CompilerResult<CompilerComputationTree>.Failure([diagnostic]);
        }
        options.Report([diagnostic]);
        return null;
    }

    private static CompilerDiagnostic ExtensionFailure(CompilerOptions options, Exception exception)
    {
        return new CompilerDiagnostic(CompilerDiagnosticCodes.Unsupported,
            $"Compiler extension failed: {exception.Message}", CompilerDiagnosticStage.Analyze,
            options.ErrorMode == CompilerErrorMode.Strict ? CompilerDiagnosticSeverity.Error : CompilerDiagnosticSeverity.Warning);
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
            var result = CreateTreeFromOnnx(model, document ?? "<memory>");
            return result;
        }
        catch (Exception exception)
        {
            var result = OnnxCompilerFrontend.Failure<CompilerComputationTree>(
                CompilerDiagnosticCodes.InvalidSource,
                $"The ONNX stream could not be loaded: {exception.Message}",
                CompilerDiagnosticStage.Parse,
                document ?? "<memory>");
            return result;
        }
    }

    /// <summary>Emits a complete ONNX model from the shared compiler tree.</summary>
    public static CompilerResult<OnnxModel> GenerateOnnx(
        CompilerComputationTree tree,
        OnnxModelCreationOptions? options = null
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        var result = OnnxCompilerBackend.EmitModel(tree, options);
        return result;
    }

    /// <summary>Emits ONNX through registered compiler printers and observation options.</summary>
    public static CompilerResult<OnnxModel> GenerateOnnx(
        CompilerComputationTree tree,
        OnnxModelCreationOptions? generationOptions,
        CompilerOptions compilerOptions
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        CompilerStructural.RequireNotNull(compilerOptions, nameof(compilerOptions));
        var extensionResult = TryExtensionPrint(tree, CompilerTargetKind.OnnxModel, compilerOptions);
        CompilerResult<OnnxModel> result;
        if (extensionResult is CompilerResult<object> printed && printed.Value is OnnxModel model)
        {
            result = CompilerResult<OnnxModel>.Success(model, printed.Diagnostics);
        }
        else if (extensionResult is CompilerResult<object> failed && !failed.IsSuccess)
        {
            result = CompilerResult<OnnxModel>.Failure(failed.Diagnostics);
        }
        else
        {
            result = GenerateOnnx(tree, generationOptions);
        }

        return compilerOptions.Observe(result);
    }

    /// <summary>Emits only the graph from the shared compiler tree.</summary>
    public static CompilerResult<OnnxGraph> GenerateOnnxGraph(
        CompilerComputationTree tree,
        OnnxModelCreationOptions? options = null
    )
    {
        var model = GenerateOnnx(tree, options);
        var result = CompilerResultMapper.Map(
            source: model,
            projection: static onnxModel => onnxModel.Graph,
            missingValueStage: CompilerDiagnosticStage.Emit,
            missingValueMessage: "The ONNX backend reported success without producing a model.");
        return result;
    }

    /// <summary>Emits an ONNX graph through registered compiler printers and observation options.</summary>
    public static CompilerResult<OnnxGraph> GenerateOnnxGraph(
        CompilerComputationTree tree,
        OnnxModelCreationOptions? generationOptions,
        CompilerOptions compilerOptions
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        CompilerStructural.RequireNotNull(compilerOptions, nameof(compilerOptions));
        var extensionResult = TryExtensionPrint(tree, CompilerTargetKind.OnnxGraph, compilerOptions);
        CompilerResult<OnnxGraph> result;
        if (extensionResult is CompilerResult<object> printed && printed.Value is OnnxGraph graph)
        {
            result = CompilerResult<OnnxGraph>.Success(graph, printed.Diagnostics);
        }
        else if (extensionResult is CompilerResult<object> failed && !failed.IsSuccess)
        {
            result = CompilerResult<OnnxGraph>.Failure(failed.Diagnostics);
        }
        else
        {
            var model = GenerateOnnx(tree, generationOptions);
            result = CompilerResultMapper.Map(
                source: model,
                projection: static onnxModel => onnxModel.Graph,
                missingValueStage: CompilerDiagnosticStage.Emit,
                missingValueMessage: "The ONNX backend reported success without producing a model.");
        }

        return compilerOptions.Observe(result);
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
        var sourceResult = session.CreateTree(source);
        var treeResult = CompilerResultMapper.Map(
            source: sourceResult,
            projection: static compilerTree => (CompilerComputationTree)compilerTree,
            missingValueStage: CompilerDiagnosticStage.Normalize,
            missingValueMessage: "The ONNX frontend reported success without producing a computation tree.");
        return treeResult;
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
    /// <summary>
    /// Преобразует ONNX model envelope и граф в immutable промежуточное представление compiler.
    /// Ошибки нормализации собираются в diagnostics и не оставляют частично построенное дерево успешным результатом.
    /// </summary>
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
                graph: model.Graph,
                document: document ?? "<memory>",
                diagnostics: diagnostics,
                outerValues: null,
                caller: null,
                includeEnvelope: true,
                model: model);

            var result = new CompilerResult<CompilerComputationTree>(tree, diagnostics);
            return result;
        }
        catch (CompilerConversionException exception)
        {
            diagnostics.Add(new CompilerDiagnostic(
                code: exception.Code,
                message: exception.Message,
                stage: CompilerDiagnosticStage.Normalize,
                severity: CompilerDiagnosticSeverity.Error));
            return CompilerResult<CompilerComputationTree>.Failure(diagnostics);
        }
        catch (Exception exception)
        {
            diagnostics.Add(new CompilerDiagnostic(
                code: CompilerDiagnosticCodes.InvalidSource,
                message: $"The ONNX graph could not be converted: {exception.Message}",
                stage: CompilerDiagnosticStage.Normalize,
                severity: CompilerDiagnosticSeverity.Error));
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
                code: code,
                message: message,
                stage: stage,
                severity: CompilerDiagnosticSeverity.Error,
                span: new CompilerSourceSpan(
                    kind: CompilerSourceSpanKind.Onnx,
                    document: document,
                    start: 0,
                    length: 0,
                    startLine: 0,
                    startColumn: 0,
                    endLine: 0,
                    endColumn: 0)),
        ]);
    }

    /// <summary>
    /// Преобразует граф ONNX в единое дерево промежуточного представления, включая captured values и рекурсивные атрибуты-графы.
    /// Сначала регистрируются типы и state, затем узлы, чтобы ссылки оставались корректными при любом порядке wire metadata.
    /// </summary>
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
                intermediateRepresentationVersion: model.IrVersion,
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
            var literal = ImportTensorLiteral(
                tensor: tensor,
                diagnostics: diagnostics,
                document: document,
                caller: caller);
            builder.AddStateMember(new CompilerStateMember(
                name: tensor.Name,
                kind: CompilerStateMemberKind.Initializer,
                type: type,
                value: literal,
                span: Span(document: document, ordinal: 0, nodeName: tensor.Name, operatorName: null)));
            stateTypes[tensor.Name] = type;
        }

        foreach (var sparse in graph.SparseInitializers)
        {
            var type = new CompilerSparseTensorType(
                CompilerElementTypeMap.FromSystemType(sparse.Value.DataType),
                sparse.Shape.Select(static x => (CompilerDimension)new CompilerFixedDimension(x)));
            var literal = ImportSparseTensorLiteral(
                tensor: sparse,
                diagnostics: diagnostics,
                document: document,
                caller: caller);
            builder.AddStateMember(new CompilerStateMember(
                name: sparse.Name,
                kind: CompilerStateMemberKind.Initializer,
                type: type,
                value: literal,
                span: Span(document: document, ordinal: 0, nodeName: sparse.Name, operatorName: null)));
            stateTypes[sparse.Name] = type;
        }

        foreach (var value in graph.Inputs)
        {
            AddValue(
                builder: builder,
                values: values,
                value: value,
                isInput: true,
                isOutput: false);
        }

        foreach (var value in graph.Outputs)
        {
            AddValue(
                builder: builder,
                values: values,
                value: value,
                isInput: false,
                isOutput: true);
        }

        foreach (var value in graph.IntermediateValues)
        {
            AddValue(
                builder: builder,
                values: values,
                value: value,
                isInput: false,
                isOutput: false);
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

        ImportNodes(
            graph: graph,
            builder: builder,
            knownTypes: knownTypes,
            document: document,
            caller: caller,
            diagnostics: diagnostics);

        var result = builder.Build();
        return result;
    }

    /// <summary>
    /// Импортирует узлы в исходном порядке и нормализует атрибуты по имени для стабильного промежуточного представления.
    /// Неизвестные ONNX operators остаются generic operations и получают warning для последующих backend этапов.
    /// </summary>
    private static void ImportNodes(
        OnnxGraph graph,
        CompilerComputationTreeBuilder builder,
        IReadOnlyDictionary<string, CompilerType> knownTypes,
        string document,
        string? caller,
        List<CompilerDiagnostic> diagnostics
    )
    {
        foreach (var pair in graph.Nodes.Select((node, index) => (node, index)))
        {
            var node = pair.node;
            var nodeName = string.IsNullOrWhiteSpace(node.Name)
                ? $"__onnx_node_{pair.index}"
                : node.Name;
            var span = Span(
                document: document,
                ordinal: pair.index,
                nodeName: nodeName,
                operatorName: node.OpType);

            if (string.IsNullOrWhiteSpace(node.Name))
            {
                diagnostics.Add(new CompilerDiagnostic(
                    code: CompilerDiagnosticCodes.Lossy,
                    message: $"ONNX node at ordinal {pair.index} had no name; '{nodeName}' was synthesized.",
                    stage: CompilerDiagnosticStage.Normalize,
                    severity: CompilerDiagnosticSeverity.Warning,
                    span: span,
                    context: new CompilerDiagnosticContext(caller, node.OpType)));
            }

            var attributes = new List<CompilerAttribute>();
            foreach (var attribute in node.Attributes.OrderBy(x => x.Name, StringComparer.Ordinal))
            {
                try
                {
                    attributes.Add(new CompilerAttribute(
                        attribute.Name,
                        ImportAttributeLiteral(
                            value: attribute.GetValue(),
                            document: document,
                            ordinal: pair.index,
                            nodeName: nodeName,
                            operatorName: node.OpType,
                            outerValues: knownTypes,
                            diagnostics: diagnostics)));
                }
                catch (CompilerConversionException exception)
                {
                    diagnostics.Add(new CompilerDiagnostic(
                        code: exception.Code,
                        message: exception.Message,
                        stage: CompilerDiagnosticStage.Normalize,
                        severity: CompilerDiagnosticSeverity.Error,
                        span: span,
                        context: new CompilerDiagnosticContext(caller, node.OpType)));
                }
            }

            CompilerOperatorMapping? mapping = null;
            var hasMapping = CompilerOperatorMappingRegistry.TryGetOnnx(
                    node.Domain,
                    node.OpType,
                    out mapping)
                && mapping is not null
                && mapping.Accepts(node);
            if (hasMapping
                && mapping!.SupportsMultidirectionalBroadcast
                && HasKnownIncompatibleBroadcast(node, knownTypes))
            {
                diagnostics.Add(new CompilerDiagnostic(
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: $"ONNX operator '{node.OpType}' has input shapes that cannot be broadcast together.",
                    stage: CompilerDiagnosticStage.Analyze,
                    severity: CompilerDiagnosticSeverity.Error,
                    span: span,
                    context: new CompilerDiagnosticContext(caller, node.OpType)));
            }

            if (hasMapping
                && mapping!.OnnxName is "MatMul" or "Gemm"
                && TryGetMatrixSemanticError(node, mapping.OnnxName, attributes, knownTypes, out var matrixError))
            {
                diagnostics.Add(new CompilerDiagnostic(
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: matrixError,
                    stage: CompilerDiagnosticStage.Analyze,
                    severity: CompilerDiagnosticSeverity.Error,
                    span: span,
                    context: new CompilerDiagnosticContext(caller, node.OpType)));
            }

            var descriptor = mapping is not null && hasMapping
                ? mapping.Descriptor
                : new CompilerOperatorDescriptor(
                    name: node.OpType,
                    domain: node.Domain,
                    capability: CompilerOperationCapability.Unsupported,
                    constraints: [string.IsNullOrEmpty(node.Domain) ? "ai.onnx" : node.Domain]);

            if (!hasMapping)
            {
                diagnostics.Add(new CompilerDiagnostic(
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: $"No semantic compiler mapping is registered for ONNX operator '{node.Domain}::{node.OpType}'; the generic operation is preserved.",
                    stage: CompilerDiagnosticStage.Analyze,
                    severity: CompilerDiagnosticSeverity.Warning,
                    span: span,
                    context: new CompilerDiagnosticContext(
                        caller: caller ?? (string.IsNullOrEmpty(graph.Name) ? "<graph>" : graph.Name),
                        callee: node.OpType)));
            }

            var operation = new CompilerOperation(
                name: nodeName,
                descriptor: descriptor,
                inputs: node.Inputs.Select(ToReference),
                outputs: node.Outputs.Select(ToReference),
                attributes: attributes,
                span: span);
            builder.AddOperation(operation);
        }
    }

    private static bool TryGetMatrixSemanticError(
        OnnxNode node,
        string operatorName,
        IReadOnlyList<CompilerAttribute> attributes,
        IReadOnlyDictionary<string, CompilerType> knownTypes,
        out string message
    )
    {
        message = string.Empty;
        if (operatorName == "Gemm")
        {
            foreach (var flagName in new[] { "transA", "transB" })
            {
                var flag = GetMatrixIntegerAttribute(attributes, flagName, 0);
                if (flag is not (0 or 1))
                {
                    message = $"ONNX Gemm attribute '{flagName}' must be 0 or 1.";
                    return true;
                }
            }

            foreach (var scaleName in new[] { "alpha", "beta" })
            {
                var scale = attributes.FirstOrDefault(attribute => attribute.Name == scaleName)?.Value;
                if (scale is not null && scale is not (CompilerFloatingPointLiteral or CompilerSignedIntegerLiteral or CompilerUnsignedIntegerLiteral))
                {
                    message = $"ONNX Gemm attribute '{scaleName}' must be numeric.";
                    return true;
                }
            }
        }

        var tensors = node.Inputs
            .Where(static input => !string.IsNullOrEmpty(input.Name))
            .Select(input => knownTypes.TryGetValue(input.Name, out var type) ? type as CompilerTensorType : null)
            .ToArray();
        if (tensors.Length < 2 || tensors.Take(2).Any(static tensor => tensor is null))
        {
            return false;
        }

        var left = tensors[0]!;
        var right = tensors[1]!;
        if (left.ElementType != right.ElementType)
        {
            message = $"ONNX {operatorName} requires both matrix operands to have the same element type.";
            return true;
        }

        var isGemm = operatorName == "Gemm";
        if (!IsSupportedMatrixElementType(left.ElementType, isGemm))
        {
            message = $"ONNX {operatorName} does not have a registered runtime-verified mapping for element type '{left.ElementType}'.";
            return true;
        }

        if (isGemm && tensors.Length > 2 && tensors[2] is { } biasType && biasType.ElementType != left.ElementType)
        {
            message = "ONNX Gemm requires bias input C to have the same element type as A and B.";
            return true;
        }

        if (left.Dimensions is null || right.Dimensions is null
            || left.Dimensions.Any(static dimension => dimension is not CompilerFixedDimension)
            || right.Dimensions.Any(static dimension => dimension is not CompilerFixedDimension))
        {
            return false;
        }

        var leftDimensions = left.Dimensions.Cast<CompilerFixedDimension>().Select(static dimension => dimension.Value).ToArray();
        var rightDimensions = right.Dimensions.Cast<CompilerFixedDimension>().Select(static dimension => dimension.Value).ToArray();
        if (isGemm)
        {
            if (leftDimensions.Length != 2 || rightDimensions.Length != 2)
            {
                message = "ONNX Gemm requires rank-2 A and B inputs.";
                return true;
            }

            var transA = GetMatrixIntegerAttribute(attributes, "transA", 0) == 1;
            var transB = GetMatrixIntegerAttribute(attributes, "transB", 0) == 1;
            var aRows = leftDimensions[transA ? 1 : 0];
            var aColumns = leftDimensions[transA ? 0 : 1];
            var bRows = rightDimensions[transB ? 1 : 0];
            var bColumns = rightDimensions[transB ? 0 : 1];
            if (aColumns != bRows)
            {
                message = $"ONNX Gemm inner dimensions are incompatible ({aColumns} and {bRows}).";
                return true;
            }

            if (tensors.Length > 2 && tensors[2]?.Dimensions is { } biasDimensions
                && biasDimensions.All(static dimension => dimension is CompilerFixedDimension))
            {
                if (biasDimensions.Count > 2)
                {
                    message = "ONNX Gemm bias input C must have rank at most 2.";
                    return true;
                }

                var biasShape = biasDimensions.Cast<CompilerFixedDimension>().Select(static dimension => dimension.Value).ToArray();
                if (!CanBroadcastTo(biasShape, [aRows, bColumns]))
                {
                    message = "ONNX Gemm bias input C cannot be broadcast to the matrix output shape.";
                    return true;
                }
            }

            return false;
        }

        if (leftDimensions.Length == 0 || rightDimensions.Length == 0)
        {
            message = "ONNX MatMul requires both inputs to have rank at least 1.";
            return true;
        }

        var leftContract = leftDimensions[leftDimensions.Length - 1];
        var rightContract = rightDimensions.Length == 1 ? rightDimensions[0] : rightDimensions[rightDimensions.Length - 2];
        if (leftContract != rightContract)
        {
            message = $"ONNX MatMul inner dimensions are incompatible ({leftContract} and {rightContract}).";
            return true;
        }

        var leftBatch = leftDimensions.Take(Math.Max(0, leftDimensions.Length - 2)).ToArray();
        var rightBatch = rightDimensions.Take(Math.Max(0, rightDimensions.Length - 2)).ToArray();
        if (leftDimensions.Length == 1)
        {
            leftBatch = [];
        }

        if (rightDimensions.Length == 1)
        {
            rightBatch = [];
        }

        if (!CanMultidirectionallyBroadcast(leftBatch, rightBatch))
        {
            message = "ONNX MatMul batch dimensions cannot be broadcast together.";
            return true;
        }

        return false;
    }

    private static bool IsSupportedMatrixElementType(CompilerElementType elementType, bool isGemm)
    {
        return isGemm
            ? elementType is CompilerElementType.Float32 or CompilerElementType.Float64
            : elementType is CompilerElementType.Float32 or CompilerElementType.Float64
                or CompilerElementType.Int32 or CompilerElementType.Int64;
    }

    private static long GetMatrixIntegerAttribute(IReadOnlyList<CompilerAttribute> attributes, string name, long defaultValue)
    {
        return attributes.FirstOrDefault(attribute => attribute.Name == name)?.Value switch
        {
            null => defaultValue,
            CompilerSignedIntegerLiteral signed => signed.Value,
            CompilerUnsignedIntegerLiteral unsigned => checked((long)unsigned.Value),
            _ => defaultValue,
        };
    }

    private static bool CanMultidirectionallyBroadcast(long[] left, long[] right)
    {
        var rank = Math.Max(left.Length, right.Length);
        for (var offset = 1; offset <= rank; offset++)
        {
            var leftDimension = offset <= left.Length ? left[left.Length - offset] : 1;
            var rightDimension = offset <= right.Length ? right[right.Length - offset] : 1;
            if (leftDimension != rightDimension && leftDimension != 1 && rightDimension != 1)
            {
                return false;
            }
        }

        return true;
    }

    private static bool CanBroadcastTo(long[] source, long[] target)
    {
        var rank = Math.Max(source.Length, target.Length);
        for (var offset = 1; offset <= rank; offset++)
        {
            var sourceDimension = offset <= source.Length ? source[source.Length - offset] : 1;
            var targetDimension = offset <= target.Length ? target[target.Length - offset] : 1;
            if (sourceDimension != targetDimension && sourceDimension != 1)
            {
                return false;
            }
        }

        return true;
    }

    private static bool HasKnownIncompatibleBroadcast(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes
    )
    {
        var shapes = node.Inputs
            .Select(input => input.Name)
            .Select(name => knownTypes.TryGetValue(name, out var type) ? type as CompilerTensorType : null)
            .ToArray();
        if (shapes.Length < 2
            || shapes.Any(shape => shape?.Dimensions is null
                || shape.Dimensions.Any(dimension => dimension is not CompilerFixedDimension)))
        {
            return false;
        }

        var dimensions = shapes
            .Select(shape => shape!.Dimensions!.Cast<CompilerFixedDimension>().Select(dimension => dimension.Value).ToArray())
            .ToArray();
        var rank = dimensions.Max(shape => shape.Length);
        for (var offset = 1; offset <= rank; offset++)
        {
            var alignedDimensions = dimensions
                .Select(shape => offset <= shape.Length ? shape[shape.Length - offset] : 1)
                .Where(dimension => dimension != 1)
                .Distinct()
                .Take(2)
                .Count();
            if (alignedDimensions > 1)
            {
                return true;
            }
        }

        return false;
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
            return ImportTensorLiteral(
                tensor: tensor,
                diagnostics: diagnostics,
                document: document,
                caller: nodeName);
        }

        if (value is OnnxSparseTensor sparse)
        {
            return ImportSparseTensorLiteral(
                tensor: sparse,
                diagnostics: diagnostics,
                document: document,
                caller: nodeName);
        }

        if (value is OnnxGraph graph)
        {
            var nested = ImportGraph(
                graph: graph,
                document: document,
                diagnostics: diagnostics,
                outerValues: outerValues,
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
            for (var index = 0; index < array.Length; index++)
            {
                var item = array.GetValue(index);
                if (item is null)
                {
                    throw new CompilerConversionException(
                        CompilerDiagnosticCodes.InvalidSource,
                        $"ONNX attribute array for node '{nodeName}' contains a null value at index {index}.");
                }

                items.Add(ImportAttributeLiteral(
                    value: item,
                    document: document,
                    ordinal: ordinal,
                    nodeName: nodeName,
                    operatorName: operatorName,
                    outerValues: outerValues,
                    diagnostics: diagnostics));
            }

            var arrayLiteral = new CompilerArrayLiteral(items, array.GetType().GetElementType()?.FullName);
            return arrayLiteral;
        }

        var result = CompilerElementTypeMap.ToLiteral(value);
        return result;
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
                code: CompilerDiagnosticCodes.Lossy,
                message: $"Tensor '{tensor.Name}' used ONNX external data; the compiler preserves its values but not the external storage placement.",
                stage: CompilerDiagnosticStage.Normalize,
                severity: CompilerDiagnosticSeverity.Warning,
                span: Span(document: document, ordinal: 0, nodeName: caller, operatorName: null)));
        }

        var result = new CompilerTensorLiteral(elementType, dimensions, values);
        return result;
    }

    private static CompilerSparseTensorLiteral ImportSparseTensorLiteral(
        OnnxSparseTensor tensor,
        List<CompilerDiagnostic> diagnostics,
        string document,
        string? caller
    )
    {
        var dimensions = tensor.Shape.Select(static x => (CompilerDimension)new CompilerFixedDimension(x));
        var result = new CompilerSparseTensorLiteral(
            dimensions,
            ImportTensorLiteral(
                tensor: tensor.Value,
                diagnostics: diagnostics,
                document: document,
                caller: caller),
            ImportTensorLiteral(
                tensor: tensor.Indices,
                diagnostics: diagnostics,
                document: document,
                caller: caller));
        return result;
    }

    private static CompilerSourceSpan Span(
        string document,
        int ordinal,
        string? nodeName,
        string? operatorName
    )
    {
        return new CompilerSourceSpan(
            kind: CompilerSourceSpanKind.Onnx,
            document: document,
            start: ordinal,
            length: 1,
            startLine: ordinal,
            startColumn: 0,
            endLine: ordinal,
            endColumn: 1,
            nodeName: nodeName,
            operatorName: operatorName);
    }
}

internal static class OnnxCompilerBackend
{
    /// <summary>
    /// Эмитирует полную ONNX-модель из промежуточного представления compiler и проверяет её повторной загрузкой после сериализации.
    /// Валидация охватывает не только структуру промежуточного представления, но и фактическую protobuf границу core API.
    /// </summary>
    public static CompilerResult<OnnxModel> EmitModel(
        CompilerComputationTree tree,
        OnnxModelCreationOptions? creationOptions = null
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        var diagnostics = new List<CompilerDiagnostic>();

        if (tree.SyntaxBody is not null)
        {
            return CompilerResult<OnnxModel>.Failure(
            [
                new CompilerDiagnostic(
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: "The ONNX backend cannot emit an unlowered C# syntax body.",
                    stage: CompilerDiagnosticStage.Emit,
                    severity: CompilerDiagnosticSeverity.Error),
            ]);
        }

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
                model.IrVersion = envelope.IntermediateRepresentationVersion;
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

            EmitGraph(
                tree: tree,
                graph: model.Graph,
                diagnostics: diagnostics,
                caller: null);

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

            var result = new CompilerResult<OnnxModel>(model, diagnostics);
            return result;
        }
        catch (CompilerConversionException exception)
        {
            diagnostics.Add(new CompilerDiagnostic(
                code: exception.Code,
                message: exception.Message,
                stage: CompilerDiagnosticStage.Emit,
                severity: CompilerDiagnosticSeverity.Error));
            return CompilerResult<OnnxModel>.Failure(diagnostics);
        }
        catch (Exception exception)
        {
            diagnostics.Add(new CompilerDiagnostic(
                code: CompilerDiagnosticCodes.InvalidSource,
                message: $"The compiler could not emit a valid ONNX model: {exception.Message}",
                stage: CompilerDiagnosticStage.Validate,
                severity: CompilerDiagnosticSeverity.Error));
            return CompilerResult<OnnxModel>.Failure(diagnostics);
        }
    }

    /// <summary>
    /// Создаёт graph values, state и узлы в порядке промежуточного представления перед сериализацией модели.
    /// Отдельный graph-проход сохраняет ONNX порядок и даёт вложенным graph literals тот же backend путь.
    /// </summary>
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
            AddInitializer(
                graph: graph,
                initializer: initializer,
                diagnostics: diagnostics,
                caller: caller);
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
                    EmitOperation(
                        graph: graph,
                        operation: compilerOperation,
                        diagnostics: diagnostics,
                        caller: caller);
                    break;
                case CompilerModuleCall moduleCall:
                    EmitModuleCall(
                        graph: graph,
                        moduleCall: moduleCall,
                        blocks: tree.Blocks.ToDictionary(block => block.Name, StringComparer.Ordinal),
                        diagnostics: diagnostics,
                        caller: caller,
                        callStack: []);
                    break;
                default:
                    throw new CompilerConversionException(
                        CompilerDiagnosticCodes.Unsupported,
                        $"Computation step '{operation.Name}' has no ONNX representation.");
            }
        }
    }

    private static void EmitModuleCall(
        OnnxGraph graph,
        CompilerModuleCall moduleCall,
        IReadOnlyDictionary<string, CompilerComputationBlock> blocks,
        List<CompilerDiagnostic> diagnostics,
        string? caller,
        IReadOnlyList<string> callStack
    )
    {
        if (callStack.Contains(moduleCall.TargetBlock, StringComparer.Ordinal))
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Recursive helper/module call '{moduleCall.TargetBlock}' cannot be inlined into an ONNX graph.");
        }

        if (!blocks.TryGetValue(moduleCall.TargetBlock, out var block))
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.MissingReference,
                $"Module call '{moduleCall.Name}' references missing block '{moduleCall.TargetBlock}'.");
        }

        if (block.Inputs.Count != moduleCall.Arguments.Count || block.Outputs.Count != moduleCall.Outputs.Count)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Module call '{moduleCall.Name}' has input or output bindings that do not match block '{block.Name}'.");
        }

        var bindings = block.Inputs
            .Select((input, index) => new KeyValuePair<string, CompilerExpression>(input.Name, moduleCall.Arguments[index]))
            .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        var returns = FindReturns(block.Body, bindings);
        if (returns.Count != block.Outputs.Count)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Block '{block.Name}' does not return the declared number of values.");
        }

        var nestedCallStack = callStack.Append(block.Name).ToArray();
        for (var index = 0; index < returns.Count; index++)
        {
            var output = moduleCall.Outputs[index];
            var expression = ResolveExpression(returns[index], bindings);
            if (expression is CompilerReferenceExpression reference)
            {
                graph.AddNode(
                    name: $"{moduleCall.Name}__identity{index}",
                    opType: "Identity",
                    domain: string.Empty,
                    docString: string.Empty,
                    inputs: [(IOnnxGraphEdge)new OnnxEdge(reference.Name)],
                    outputs: [(IOnnxGraphEdge)new OnnxEdge(output.Name)],
                    attributes: []);
                continue;
            }

            if (expression is not CompilerInvocationExpression invocation)
            {
                throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Unsupported,
                    $"Block '{block.Name}' returns an expression that cannot be lowered to ONNX.");
            }

            if (CompilerOperatorMappingRegistry.TryGetTorchSharpCall(invocation.Target, out var mapping, out var receiver)
                && mapping is not null)
            {
                EmitMappedInvocation(
                    graph: graph,
                    name: $"{moduleCall.Name}__{mapping.OnnxName.ToLowerInvariant()}{index}",
                    output: output,
                    invocation: invocation,
                    mapping: mapping,
                    receiver: receiver,
                    bindings: bindings,
                    diagnostics: diagnostics,
                    caller: caller);
                continue;
            }

            if (TryGetInvocationName(invocation.Target, out var nestedBlockName)
                && blocks.ContainsKey(nestedBlockName))
            {
                var nestedArguments = invocation.Arguments
                    .Select(argument => ResolveExpression(argument, bindings))
                    .ToArray();
                EmitModuleCall(
                    graph: graph,
                    moduleCall: CompilerModuleCall.CreateWithArguments(
                        name: $"{moduleCall.Name}__{nestedBlockName}{index}",
                        targetBlock: nestedBlockName,
                        arguments: nestedArguments,
                        outputs: [output],
                        span: invocation.Span),
                    blocks: blocks,
                    diagnostics: diagnostics,
                    caller: nestedBlockName,
                    callStack: nestedCallStack);
                continue;
            }

            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Block '{block.Name}' calls an operation without a registered ONNX mapping.");
        }
    }

    private static void EmitMappedInvocation(
        OnnxGraph graph,
        string name,
        CompilerValueReference output,
        CompilerInvocationExpression invocation,
        CompilerOperatorMapping mapping,
        CompilerExpression? receiver,
        IReadOnlyDictionary<string, CompilerExpression> bindings,
        List<CompilerDiagnostic> diagnostics,
        string? caller
    )
    {
        var expressions = new List<CompilerExpression>();
        if (receiver is not null)
        {
            expressions.Add(receiver);
        }

        var requiredInputs = mapping.InputCount - expressions.Count;
        if (requiredInputs < 0
            || invocation.Arguments.Count < requiredInputs + mapping.FixedTorchSharpArguments.Count
            || invocation.Arguments.Count > requiredInputs + mapping.AttributeNames.Count + mapping.FixedTorchSharpArguments.Count)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Call to '{mapping.TorchSharpNames[0]}' has an unsupported input or attribute configuration.");
        }

        expressions.AddRange(invocation.Arguments.Take(requiredInputs));
        var inputs = expressions.Select(expression => ResolveReference(expression, bindings)).ToArray();
        var attributes = new List<CompilerAttribute>();
        foreach (var (argument, index) in invocation.Arguments
            .Skip(requiredInputs)
            .Take(mapping.AttributeNames.Count)
            .Select((argument, index) => (argument, index)))
        {
            var resolvedArgument = ResolveExpression(argument, bindings);
            if (resolvedArgument is not CompilerLiteralExpression literal)
            {
                throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Unsupported,
                    $"Call to '{mapping.TorchSharpNames[0]}' requires literal arguments for mapped attributes.");
            }

            attributes.Add(new CompilerAttribute(mapping.AttributeNames[index], literal.Literal));
        }

        foreach (var (argument, index) in invocation.Arguments
            .Skip(invocation.Arguments.Count - mapping.FixedTorchSharpArguments.Count)
            .Select((argument, index) => (argument, index)))
        {
            if (ResolveExpression(argument, bindings) is not CompilerLiteralExpression
                {
                    Literal: CompilerFloatingPointLiteral fixedValue,
                }
                || (float)fixedValue.Value != mapping.FixedTorchSharpArguments[index])
            {
                throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Unsupported,
                    $"Call to '{mapping.TorchSharpNames[0]}' has unsupported fixed trailing arguments.");
            }
        }

        var operation = new CompilerOperation(
            name: name,
            descriptor: mapping.Descriptor,
            inputs: inputs,
            outputs: [output],
            attributes: attributes,
            span: invocation.Span);
        if (!mapping.Accepts(operation))
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Call to '{mapping.TorchSharpNames[0]}' has an unsupported ONNX signature.");
        }

        EmitOperation(graph, operation, diagnostics, caller);
    }

    private static IReadOnlyList<CompilerExpression> FindReturns(
        CompilerStatement statement,
        IReadOnlyDictionary<string, CompilerExpression> bindings
    )
    {
        var results = new List<CompilerExpression>();
        CollectReturns(statement, results, bindings);
        return results;
    }

    private static void CollectReturns(
        CompilerStatement statement,
        List<CompilerExpression> results,
        IReadOnlyDictionary<string, CompilerExpression> bindings
    )
    {
        switch (statement)
        {
            case CompilerReturnStatement { Expression: not null } returnStatement:
                if (returnStatement.Expression is CompilerTupleExpression tuple)
                {
                    results.AddRange(tuple.Items);
                }
                else
                {
                    results.Add(returnStatement.Expression);
                }
                break;
            case CompilerBlockStatement block:
                foreach (var child in block.Statements)
                {
                    CollectReturns(child, results, bindings);
                }

                break;
            case CompilerStaticIfStatement conditional:
                if (TryEvaluateStaticBoolean(ResolveExpression(conditional.Condition, bindings), out var condition))
                {
                    CollectReturns(
                        condition ? conditional.WhenTrue : conditional.WhenFalse ?? new CompilerBlockStatement([]),
                        results,
                        bindings);
                }

                break;
            default:
                throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Unsupported,
                    $"Statement '{statement.GetType().Name}' in an inlined block cannot be lowered to ONNX.");
        }
    }

    private static bool TryEvaluateStaticBoolean(CompilerExpression expression, out bool value)
    {
        if (expression is CompilerLiteralExpression { Literal: CompilerBooleanLiteral literal })
        {
            value = literal.Value;
            return true;
        }

        if (expression is CompilerUnaryExpression { Operator: "!", Expression: var operand }
            && TryEvaluateStaticBoolean(operand, out var operandValue))
        {
            value = !operandValue;
            return true;
        }

        value = false;
        return false;
    }

    private static CompilerValueReference ResolveReference(
        CompilerExpression expression,
        IReadOnlyDictionary<string, CompilerExpression> bindings
    )
    {
        if (ResolveExpression(expression, bindings) is not CompilerReferenceExpression reference)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                "Only helper arguments that reference input values can be lowered to ONNX.");
        }

        return new CompilerValueReference(reference.Name);
    }

    private static CompilerExpression ResolveExpression(
        CompilerExpression expression,
        IReadOnlyDictionary<string, CompilerExpression> bindings
    )
    {
        if (expression is CompilerReferenceExpression reference
            && bindings.TryGetValue(reference.Name, out var boundExpression))
        {
            if (boundExpression is CompilerReferenceExpression boundReference
                && string.Equals(boundReference.Name, reference.Name, StringComparison.Ordinal))
            {
                return expression;
            }

            return ResolveExpression(boundExpression, bindings);
        }

        return expression;
    }

    private static bool TryGetInvocationName(CompilerExpression expression, out string name)
    {
        if (expression is CompilerReferenceExpression reference)
        {
            name = reference.Name;
            return true;
        }

        name = string.Empty;
        return false;
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
        var instance = Activator.CreateInstance(valueType, name, type);
        if (instance is not OnnxValue onnxValue)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"An ONNX value of runtime type '{valueType.FullName}' could not be created.");
        }

        return onnxValue;
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
                code: CompilerDiagnosticCodes.Unsupported,
                message: $"Operation '{operation.Name}' is emitted through generic ONNX passthrough because no semantic mapping is registered.",
                stage: CompilerDiagnosticStage.Emit,
                severity: CompilerDiagnosticSeverity.Warning,
                span: operation.Span,
                context: new CompilerDiagnosticContext(
                    caller: caller,
                    callee: operation.Descriptor.Name)));
        }
        else if (!CompilerOperatorMappingRegistry.TryGetOnnx(
            operation.Descriptor.Domain,
            operation.Descriptor.Name,
            out _))
        {
            diagnostics.Add(new CompilerDiagnostic(
                code: CompilerDiagnosticCodes.Unsupported,
                message: $"Operation '{operation.Name}' declares a semantic capability without a registered compiler mapping.",
                stage: CompilerDiagnosticStage.Emit,
                severity: CompilerDiagnosticSeverity.Error,
                span: operation.Span,
                context: new CompilerDiagnosticContext(caller, operation.Descriptor.Name)));
            return;
        }

        if (string.Equals(operation.Descriptor.Name, "Swish", StringComparison.Ordinal))
        {
            EmitSwish(graph, operation);
            return;
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

    private static void EmitSwish(OnnxGraph graph, CompilerOperation operation)
    {
        if (operation.Inputs.Count != 1 || operation.Outputs.Count != 1)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                "Swish requires exactly one input and one output.");
        }

        var input = operation.Inputs[0].Name;
        var sigmoidInput = input;
        var alpha = CompilerOperatorMapping.GetFloatAttribute(operation, "alpha", 1f);
        if (alpha != 1f)
        {
            sigmoidInput = $"{operation.Name}__scaled";
            var scale = graph.AddTensor($"{operation.Name}__alpha", [], [alpha]);
            graph.AddNode(
                name: $"{operation.Name}__scale",
                opType: "Mul",
                domain: string.Empty,
                docString: string.Empty,
                inputs: [(IOnnxGraphEdge)new OnnxEdge(input), scale],
                outputs: [(IOnnxGraphEdge)new OnnxEdge(sigmoidInput)],
                attributes: []);
        }

        var sigmoidOutput = $"{operation.Name}__sigmoid";
        graph.AddNode(
            name: $"{operation.Name}__sigmoid_node",
            opType: "Sigmoid",
            domain: string.Empty,
            docString: string.Empty,
            inputs: [(IOnnxGraphEdge)new OnnxEdge(sigmoidInput)],
            outputs: [(IOnnxGraphEdge)new OnnxEdge(sigmoidOutput)],
            attributes: []);
        graph.AddNode(
            name: operation.Name,
            opType: "Mul",
            domain: string.Empty,
            docString: string.Empty,
            inputs: [(IOnnxGraphEdge)new OnnxEdge(input), (IOnnxGraphEdge)new OnnxEdge(sigmoidOutput)],
            outputs: [(IOnnxGraphEdge)new OnnxEdge(operation.Outputs[0].Name)],
            attributes: []);
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
        var instance = Activator.CreateInstance(
            attributeType,
            attribute.Name,
            value);
        if (instance is not OnnxAttribute onnxAttribute)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"ONNX attribute '{attribute.Name}' could not be created.");
        }

        return onnxAttribute;
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
                if (nestedModelResult.Value is not { } nestedModel)
                {
                    throw new CompilerConversionException(
                        CompilerDiagnosticCodes.InvalidSource,
                        $"Nested graph '{graphLiteral.Graph.Name}' emitted no ONNX model.");
                }

                return nestedModel.Graph;
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
        var elementType = array.GetType().GetElementType();
        if (elementType is null)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Tensor initializer '{name}' has no runtime element type.");
        }

        method.MakeGenericMethod(elementType)
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
        var valueElementType = values.GetType().GetElementType();
        var indexElementType = indices.GetType().GetElementType();
        if (valueElementType is null || indexElementType is null)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Sparse initializer '{name}' has no runtime element type.");
        }

        method.MakeGenericMethod(
                valueElementType,
                indexElementType)
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
        var result = model.Graph.SparseInitializers.Single();
        return result;
    }

    private static OnnxTensor CreateTensor(CompilerTensorLiteral tensor)
    {
        var model = OnnxModel.Create();
        AddTensor(model.Graph, "attribute", tensor);
        var result = model.Graph.Initializers.Single();
        return result;
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
                $"ONNX value type '{type.GetType().Name}' is not supported by the compiler intermediate representation."),
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
                $"CLR tensor element type '{type.FullName}' is not supported by the compiler intermediate representation."),
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
            var real = Convert.ToDouble(GetRequiredNumericProperty(value, "Real"));
            var imaginary = Convert.ToDouble(GetRequiredNumericProperty(value, "Imaginary"));
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
            var raw = GetRequiredNumericProperty(value, "Value");
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
            CompilerSignedIntegerLiteral signed => ConvertToRequiredValue(
                signed.Value,
                ToSystemType(signed.ElementType),
                signed.ElementType),
            CompilerUnsignedIntegerLiteral unsigned => ConvertToRequiredValue(
                unsigned.Value,
                ToSystemType(unsigned.ElementType),
                unsigned.ElementType),
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
            CompilerElementType.Float16 => ConvertToRequiredValue(
                literal.Value,
                ToSystemType(literal.ElementType),
                literal.ElementType),
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
        var complex = Activator.CreateInstance(
            type,
            literal.ElementType == CompilerElementType.Complex64
                ? (object)(float)literal.Real
                : literal.Real,
            literal.ElementType == CompilerElementType.Complex64
                ? (object)(float)literal.Imaginary
                : literal.Imaginary);
        if (complex is null)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Complex scalar type '{literal.ElementType}' could not be constructed.");
        }

        return complex;
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
            var packed = encodedFactory.Invoke(null, [encoded]);
            if (packed is null)
            {
                throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Unsupported,
                    $"Packed scalar type '{literal.ElementType}' returned no value from FromEncoded.");
            }

            return packed;
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

    private static object GetRequiredNumericProperty(object value, string propertyName)
    {
        var property = value.GetType().GetProperty(propertyName);
        if (property?.GetValue(value) is not { } propertyValue)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.InvalidSource,
                $"Numeric value '{value.GetType().FullName}' does not expose a usable '{propertyName}' property.");
        }

        return propertyValue;
    }

    private static object ConvertToRequiredValue(
        object value,
        Type targetType,
        CompilerElementType elementType
    )
    {
        var converted = Convert.ChangeType(value, targetType);
        if (converted is null)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Scalar value of element type '{elementType}' could not be converted to '{targetType.FullName}'.");
        }

        return converted;
    }

    private static Type FindType(string name)
    {
        return Type.GetType(name)
            ?? throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"Runtime type '{name}' is unavailable on this target framework.");
    }
}
