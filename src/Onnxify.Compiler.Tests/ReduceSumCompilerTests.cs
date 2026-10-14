using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class ReduceSumCompilerTests
{
    [Fact]
    public void ReduceSumRoundTripsTorchSharpOnnxAndGeneratedTorchSharp()
    {
        // Source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_sum_dim_IntList).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/reduction/reduction_ops_test.cc.
        var source = new CSharpTorchSharpSource("""
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                return torch.sum(input, 1);
            }
            """);
        var imported = Compiler.CreateTreeFromTorchSharp(source);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("ReduceSum", operation.Node.OpType);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal("ReduceSum", Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        var onnxImported = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(onnxImported.IsSuccess, FormatDiagnostics(onnxImported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(onnxImported.Value!, new CompilerCSharpGenerationOptions
        {
            Namespace = "Onnxify.GeneratedReduceSum",
            ClassName = "GeneratedReduceSum",
            ModuleName = "generated-reduce-sum",
        });
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        using var module = CompileGeneratedModule(generated.SourceCode());

        float[] values = [1f, -2f, 3f, 4f, 5f, -6f];
        using var eagerInput = global::TorchSharp.torch.tensor(values, [2L, 3L]);
        using var eagerOutput = global::TorchSharp.torch.sum(eagerInput, 1);
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-reducesum-{Guid.NewGuid():N}.onnx");
        try
        {
            emitted.Value!.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run([NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(values, [2, 3]))]);
            var onnxOutput = results.Single().AsTensor<float>().ToArray();
            var torchOutput = ExecuteGeneratedModule(module, values);
            Assert.Equal(eagerOutput.data<float>().ToArray(), onnxOutput);
            Assert.Equal(onnxOutput, torchOutput);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    private static IDisposable CompileGeneratedModule(string source)
    {
        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))!
            .Split(Path.PathSeparator)
            .Select(static path => MetadataReference.CreateFromFile(path))
            .Append(MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location));
        var compilation = CSharpCompilation.Create(
            $"Onnxify.GeneratedReduceSum{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var type = assembly.GetTypes().Single(candidate => candidate.Name == "GeneratedReduceSum");
        return (IDisposable)Activator.CreateInstance(type, "generated-reduce-sum")!;
    }

    private static float[] ExecuteGeneratedModule(IDisposable module, float[] values)
    {
        using var input = global::TorchSharp.torch.tensor(values, [2L, 3L]);
        using var output = (global::TorchSharp.torch.Tensor)module.GetType().GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<float>().ToArray();
    }

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics) => string.Join(
        Environment.NewLine,
        diagnostics.Select(static diagnostic => $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));
}
