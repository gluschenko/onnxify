using Onnxify;

namespace Onnxify.Compiler;

/// <summary>
/// Identifies the kind of source entering the compiler.
/// </summary>
public enum CompilerSourceKind
{
    /// <summary>
    /// An ONNX model supplied by the core Onnxify library.
    /// </summary>
    Onnx = 1,

    /// <summary>
    /// C# source representing TorchSharp computation.
    /// </summary>
    CSharpTorchSharp = 2,
}

/// <summary>
/// Identifies the kind of output produced by a compiler sink.
/// </summary>
public enum CompilerTargetKind
{
    /// <summary>
    /// An ONNX graph supplied by or emitted through the core Onnxify library.
    /// </summary>
    OnnxGraph = 1,

    /// <summary>
    /// A complete ONNX model including its model-level envelope.
    /// </summary>
    OnnxModel = 2,

    /// <summary>
    /// Generated C# source.
    /// </summary>
    CSharp = 3,
}

/// <summary>
/// Describes a compiler input without coupling the compiler to a consumer package.
/// </summary>
public interface ICompilerSource
{
    /// <summary>
    /// Gets the source representation kind.
    /// </summary>
    CompilerSourceKind Kind { get; }
}

/// <summary>
/// Provides an ONNX model as a compiler source.
/// </summary>
public sealed class OnnxCompilerSource : ICompilerSource
{
    /// <summary>
    /// Initializes a source from an existing core Onnxify model.
    /// </summary>
    /// <param name="model">The model to expose to a compiler frontend.</param>
    public OnnxCompilerSource(OnnxModel model, string? document = null)
    {
        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        Model = model;
        Document = document;
    }

    /// <summary>
    /// Gets the supplied core model.
    /// </summary>
    public OnnxModel Model { get; }

    /// <summary>
    /// Gets the logical source document used when creating diagnostics.
    /// </summary>
    public string? Document { get; }

    /// <inheritdoc />
    public CompilerSourceKind Kind => CompilerSourceKind.Onnx;
}

/// <summary>
/// Provides C# TorchSharp computation source without requiring a TorchSharp runtime reference.
/// </summary>
public sealed class CSharpTorchSharpSource : ICompilerSource
{
    /// <summary>
    /// Initializes a source from C# text.
    /// </summary>
    /// <param name="sourceText">The C# source text to scan in a later compiler phase.</param>
    public CSharpTorchSharpSource(string sourceText)
    {
        if (sourceText is null)
        {
            throw new ArgumentNullException(nameof(sourceText));
        }

        if (sourceText.Length == 0)
        {
            throw new ArgumentException("Source text cannot be empty.", nameof(sourceText));
        }

        SourceText = sourceText;
    }

    /// <summary>
    /// Initializes a source that points at a method in a compiled assembly. The compiler owns
    /// the decompilation and never exposes the underlying decompiler syntax tree.
    /// </summary>
    /// <param name="module">The compiler-neutral module descriptor.</param>
    public CSharpTorchSharpSource(CompilerTorchSharpModuleDescriptor module)
    {
        CompilerStructural.RequireNotNull(module, nameof(module));
        Module = module;
        SourceText = string.Empty;
    }

    /// <summary>
    /// Gets the supplied C# source text.
    /// </summary>
    public string SourceText { get; }

    /// <summary>Gets the optional compiled-module descriptor used for compiler decompilation.</summary>
    public CompilerTorchSharpModuleDescriptor? Module { get; }

    /// <summary>Gets the optional compiled-module descriptor used for compiler decompilation.</summary>
    public CompilerTorchSharpModuleDescriptor? ModuleDescriptor => Module;

    /// <inheritdoc />
    public CompilerSourceKind Kind => CompilerSourceKind.CSharpTorchSharp;
}

/// <summary>
/// Represents the opaque compiler tree that later phases will replace with the shared IR.
/// </summary>
public interface ICompilerTree
{
}

/// <summary>
/// Describes a compiler output target and its result type.
/// </summary>
/// <typeparam name="TOutput">The type emitted by the target.</typeparam>
public interface ICompilerSink<out TOutput>
{
    /// <summary>
    /// Gets the target representation kind.
    /// </summary>
    CompilerTargetKind Kind { get; }
}

/// <summary>
/// Describes an ONNX graph output target.
/// </summary>
public sealed class OnnxCompilerSink : ICompilerSink<OnnxGraph>
{
    /// <inheritdoc />
    public CompilerTargetKind Kind => CompilerTargetKind.OnnxGraph;
}

/// <summary>
/// Describes a complete ONNX model output target.
/// </summary>
public sealed class OnnxModelCompilerSink : ICompilerSink<OnnxModel>
{
    /// <inheritdoc />
    public CompilerTargetKind Kind => CompilerTargetKind.OnnxModel;
}

/// <summary>
/// Describes a generated C# output target.
/// </summary>
public sealed class CSharpCompilerSink : ICompilerSink<string>
{
    /// <inheritdoc />
    public CompilerTargetKind Kind => CompilerTargetKind.CSharp;
}

/// <summary>
/// Defines the initial compiler orchestration boundary.
/// </summary>
public interface ICompilerSession
{
    /// <summary>
    /// Creates an opaque compiler tree from a source representation.
    /// </summary>
    /// <param name="source">The source to pass to a future frontend.</param>
    /// <returns>An opaque compiler tree.</returns>
    CompilerResult<ICompilerTree> CreateTree(ICompilerSource source);

    /// <summary>
    /// Generates a typed output through a target sink.
    /// </summary>
    /// <typeparam name="TOutput">The type emitted by <paramref name="sink" />.</typeparam>
    /// <param name="tree">The compiler tree to pass to a future backend.</param>
    /// <param name="sink">The target sink that defines the output type.</param>
    /// <returns>The generated output.</returns>
    CompilerResult<TOutput> Generate<TOutput>(ICompilerTree tree, ICompilerSink<TOutput> sink);
}
