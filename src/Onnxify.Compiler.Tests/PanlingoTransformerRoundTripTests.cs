using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class PanlingoTransformerRoundTripTests
{
    [Fact]
    public void PanlingoTransformerRunsThroughGeneratedTorchSharpAndOnnxRuntime()
    {
        // Operator semantics: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py
        // (Gather, LayerNormalization, Softmax, Dropout, Shape, Identity) and
        // third_party/onnxscript/onnxscript/function_libs/torch_lib/ops_test_data.py
        // (gather, layer_norm, softmax, dropout cases).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/gather_op_test.cc,
        // math/softmax_test.cc, nn/dropout_op_test.cc, tensor/shape_op_test.cc, tensor/identity_op_test.cc.
        // LayerNormalization has no dedicated CPU provider test in this checkout; behavior follows its
        // generated ONNX schema docs in src/Onnxify/Assets/onnx_operators.json.
        var modelPath = Path.Combine(AppContext.BaseDirectory, "Assets", "panlingo-transformer.onnx");
        Assert.True(File.Exists(modelPath), $"Missing Panlingo Transformer fixture '{modelPath}'.");
        var imported = Compiler.CreateTreeFromOnnx(modelPath);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var tree = Assert.IsType<CompilerComputationTree>(imported.Value);
        var nodes = tree.Operations.OfType<CompilerOnnxStep>().ToArray();
        Assert.All(nodes, node => Assert.NotEqual(CompilerOperationCapability.Unsupported, node.Descriptor.Capability));
        Assert.Equal(27, nodes.Count(node => node.Node is global::Onnxify.Gather));
        Assert.Equal(17, nodes.Count(node => node.Node is global::Onnxify.LayerNormalization));
        Assert.Equal(8, nodes.Count(node => node.Node is global::Onnxify.Softmax));
        Assert.Equal(24, nodes.Count(node => node.Node is global::Onnxify.Dropout));
        Assert.Single(nodes, node => node.Node is global::Onnxify.Shape);
        Assert.Single(nodes, node => node.Node is global::Onnxify.Identity);

        var generated = Compiler.GenerateCSharp(tree, new CompilerCSharpGenerationOptions
        {
            Namespace = "Onnxify.GeneratedPanlingoTransformer",
            ClassName = "GeneratedPanlingoTransformer",
            ModuleName = "panlingo-transformer",
        });
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        using var module = CompileGeneratedModule(generated.Value!);

        var roundTripPath = Path.Combine(Path.GetTempPath(), $"oxy036-{Guid.NewGuid():N}.onnx");
        try
        {
            var emitted = Compiler.GenerateOnnx(tree);
            Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
            emitted.Value!.Save(roundTripPath, overwrite: true);
            using var sourceSession = new InferenceSession(modelPath);
            using var emittedSession = new InferenceSession(roundTripPath);
            var inputName = sourceSession.InputMetadata.Keys.Single();
            var outputName = sourceSession.OutputMetadata.Keys.Single();
            var ids = Enumerable.Range(0, 256).Select(index => (long)(index % 128)).ToArray();
            using var sourceResults = sourceSession.Run([NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<long>(ids, [1, 256]))]);
            using var emittedResults = emittedSession.Run([NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<long>(ids, [1, 256]))]);
            var expected = sourceResults.Single().AsTensor<float>().ToArray();
            var onnxRoundTrip = emittedResults.Single().AsTensor<float>().ToArray();
            Assert.Equal([1, 256, 13594], sourceResults.Single().AsTensor<float>().Dimensions.ToArray());
            var generatedOutput = ExecuteGeneratedModule(module, ids);
            Assert.Equal(expected.Length, generatedOutput.Length);
            AssertOutputsClose(expected, generatedOutput);
            AssertOutputsClose(expected, onnxRoundTrip);
        }
        finally
        {
            if (File.Exists(roundTripPath)) File.Delete(roundTripPath);
        }
    }

    private static IDisposable CompileGeneratedModule(string source)
    {
        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))!
            .Split(Path.PathSeparator)
            .Select(static path => MetadataReference.CreateFromFile(path))
            .Append(MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location));
        var compilation = CSharpCompilation.Create(
            $"Onnxify.GeneratedPanlingo{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var type = assembly.GetTypes().Single(candidate => candidate.Name == "GeneratedPanlingoTransformer");
        return (IDisposable)Activator.CreateInstance(type, "panlingo-transformer")!;
    }

    private static float[] ExecuteGeneratedModule(IDisposable module, long[] ids)
    {
        using var input = global::TorchSharp.torch.tensor(ids, [1L, 256L], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var output = (global::TorchSharp.torch.Tensor)module.GetType().GetMethod("forward")!.Invoke(module, [input])!;
        Assert.Equal([1L, 256L, 13594L], output.shape);
        return output.data<float>().ToArray();
    }

    private static void AssertOutputsClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var tolerance = 0.002f + Math.Abs(expected[i]) * 0.0002f;
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"Element {i}: {expected[i]} vs {actual[i]} (tol {tolerance}).");
        }
    }

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics) => string.Join(
        Environment.NewLine,
        diagnostics.Select(static diagnostic => $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));
}
