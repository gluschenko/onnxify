using Onnxify;
using Onnxify.Compiler;
using Onnxify.ModelGenerator;
using Onnxify.TorchSharp;

namespace Onnxify.Compiler.Tests;

public sealed class CompilerBoundaryTests
{
    [Fact]
    public void Onnx_source_preserves_core_model_and_source_kind()
    {
        var model = OnnxModel.Create();
        var source = new OnnxCompilerSource(model);

        Assert.Same(model, source.Model);
        Assert.Equal(CompilerSourceKind.Onnx, source.Kind);
    }

    [Fact]
    public void CSharp_source_preserves_source_text_and_source_kind()
    {
        const string sourceText = "return input;";
        var source = new CSharpTorchSharpSource(sourceText);

        Assert.Equal(sourceText, source.SourceText);
        Assert.Equal(CompilerSourceKind.CSharpTorchSharp, source.Kind);
    }

    [Fact]
    public void Compiler_sinks_report_typed_target_kinds()
    {
        ICompilerSink<OnnxGraph> onnxSink = new OnnxCompilerSink();
        ICompilerSink<string> csharpSink = new CSharpCompilerSink();

        Assert.Equal(CompilerTargetKind.OnnxGraph, onnxSink.Kind);
        Assert.Equal(CompilerTargetKind.CSharp, csharpSink.Kind);
    }

    [Fact]
    public void Compiler_session_contract_supports_opaque_tree_and_typed_output()
    {
        ICompilerSession session = new TestCompilerSession();

        var tree = session.CreateTree(new CSharpTorchSharpSource("return input;"));
        var output = session.Generate(tree, new CSharpCompilerSink());

        Assert.Equal("generated", output);
    }

    [Fact]
    public void Compiler_session_contract_supports_onnx_output()
    {
        ICompilerSession session = new TestCompilerSession();

        var tree = session.CreateTree(new OnnxCompilerSource(OnnxModel.Create()));
        var output = session.Generate(tree, new OnnxCompilerSink());

        Assert.NotNull(output);
    }

    [Fact]
    public void Consumer_projects_can_use_the_compiler_boundary()
    {
        ICompilerSource onnxSource = new OnnxCompilerSource(OnnxModel.Create());
        ICompilerSource torchSource = new CSharpTorchSharpSource("return input;");

        Assert.Equal(CompilerSourceKind.Onnx, onnxSource.Kind);
        Assert.Equal(CompilerSourceKind.CSharpTorchSharp, torchSource.Kind);
    }

    [Fact]
    public void Onnx_source_rejects_null_model()
    {
        Assert.Throws<ArgumentNullException>(() => new OnnxCompilerSource(null!));
    }

    [Fact]
    public void CSharp_source_rejects_empty_source_text()
    {
        Assert.Throws<ArgumentException>(() => new CSharpTorchSharpSource(string.Empty));
    }

    private sealed class TestCompilerSession : ICompilerSession
    {
        public ICompilerTree CreateTree(ICompilerSource source)
        {
            return new TestCompilerTree(source);
        }

        public TOutput Generate<TOutput>(ICompilerTree tree, ICompilerSink<TOutput> sink)
        {
            if (sink.Kind == CompilerTargetKind.CSharp && typeof(TOutput) == typeof(string))
            {
                return (TOutput)(object)"generated";
            }

            if (sink.Kind == CompilerTargetKind.OnnxGraph && typeof(TOutput) == typeof(OnnxGraph))
            {
                return (TOutput)(object)OnnxModel.Create().Graph;
            }

            throw new InvalidOperationException("The test session does not support this compiler sink.");
        }
    }

    private sealed class TestCompilerTree(ICompilerSource source) : ICompilerTree
    {
        public ICompilerSource Source { get; } = source;
    }

    // These methods are compile-only consumer smoke checks. Calling the ModelGenerator method
    // would load its Roslyn analyzer dependencies into this runtime test process.
    private static ICompilerSource CreateModelGeneratorSource(OnnxModelGenerator consumer)
    {
        _ = consumer;
        return new OnnxCompilerSource(OnnxModel.Create());
    }

    private static ICompilerSource CreateTorchSharpSource(TorchTensorDataType consumer)
    {
        _ = consumer;
        return new CSharpTorchSharpSource("return input;");
    }
}
