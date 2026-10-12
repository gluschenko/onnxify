using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class PanlingoLstmRoundTripTests
{
    [Fact]
    public void PanlingoLstmRunsThroughGeneratedTorchSharpAndOnnxRuntime()
    {
        // Source: third_party/onnxscript/tests/function_libs/torch_lib/e2e_ops_tests.py
        // (test_lstm_bidirectional). Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/rnn/deep_cpu_lstm_op_test.cc
        // (BidirectionalSimpleWeightsNoBiasTwoRows, ONNXRuntime_TestLSTMBidirectionalBasic).
        var modelPath = Path.Combine(AppContext.BaseDirectory, "Assets", "panlingo-lstm.onnx");
        Assert.True(File.Exists(modelPath), $"Missing Panlingo LSTM fixture '{modelPath}'.");

        var imported = Compiler.CreateTreeFromOnnx(modelPath);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var tree = Assert.IsType<CompilerComputationTree>(imported.Value);
        var operations = tree.Operations.OfType<CompilerOnnxStep>().ToArray();
        Assert.Equal(18, operations.Length);
        Assert.All(operations, operation => Assert.NotEqual(CompilerOperationCapability.Unsupported, operation.Descriptor.Capability));
        var lstm = Assert.Single(operations, operation => operation.Node.OpType == "LSTM");
        Assert.Equal(CompilerOperationCapability.Bidirectional, lstm.Descriptor.Capability);
        Assert.Equal("bidirectional", lstm.Node.Attributes.Single(attribute => attribute.Name == "direction").GetValue());
        Assert.Equal(256L, lstm.Node.Attributes.Single(attribute => attribute.Name == "hidden_size").GetValue());
        Assert.Equal(4, lstm.Inputs.Count);
        Assert.Equal(3, lstm.Outputs.Count);
        Assert.Equal(2, operations.Count(operation => operation.Node.OpType == "ReduceSum"));
        Assert.Single(operations, operation => operation.Node.OpType == "Gather");
        Assert.Single(operations, operation => operation.Node.OpType == "Identity");

        var generated = Compiler.GenerateCSharp(tree, new CompilerCSharpGenerationOptions
        {
            Namespace = "Onnxify.GeneratedPanlingoLstm",
            ClassName = "GeneratedPanlingoLstm",
            ModuleName = "panlingo-lstm",
        });
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        using var module = CompileGeneratedModule(generated.Value!);

        var roundTripPath = Path.Combine(Path.GetTempPath(), $"oxy039-{Guid.NewGuid():N}.onnx");
        try
        {
            var emitted = Compiler.GenerateOnnx(tree);
            Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
            emitted.Value!.Save(roundTripPath, overwrite: true);

            AssertOutputsMatchForShape(modelPath, roundTripPath, module, batchSize: 1, sequenceLength: 32);
            AssertOutputsMatchForShape(modelPath, roundTripPath, module, batchSize: 2, sequenceLength: 16);
        }
        finally
        {
            if (File.Exists(roundTripPath)) File.Delete(roundTripPath);
        }
    }

    private static void AssertOutputsMatchForShape(
        string sourcePath,
        string roundTripPath,
        IDisposable module,
        int batchSize,
        int sequenceLength)
    {
        var ids = Enumerable.Range(0, batchSize * sequenceLength)
            .Select(static index => (long)(index % 32))
            .ToArray();
        var shape = new[] { batchSize, sequenceLength };
        var torchShape = new[] { (long)batchSize, sequenceLength };
        var sourceOutput = RunOnnx(sourcePath, ids, shape);
        var emittedOutput = RunOnnx(roundTripPath, ids, shape);
        var torchOutput = ExecuteGeneratedModule(module, ids, torchShape);

        Assert.Equal(batchSize * 12, sourceOutput.Length);
        AssertOutputsClose(sourceOutput, emittedOutput);
        AssertOutputsClose(sourceOutput, torchOutput);
    }

    private static IDisposable CompileGeneratedModule(string source)
    {
        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))!
            .Split(Path.PathSeparator)
            .Select(static path => MetadataReference.CreateFromFile(path))
            .Append(MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location));
        var compilation = CSharpCompilation.Create(
            $"Onnxify.GeneratedPanlingoLstm{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var type = assembly.GetTypes().Single(candidate => candidate.Name == "GeneratedPanlingoLstm");
        return (IDisposable)Activator.CreateInstance(type, "panlingo-lstm")!;
    }

    private static float[] ExecuteGeneratedModule(IDisposable module, long[] ids, long[] shape)
    {
        using var input = global::TorchSharp.torch.tensor(ids, shape, dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var output = (global::TorchSharp.torch.Tensor)module.GetType().GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<float>().ToArray();
    }

    private static float[] RunOnnx(string modelPath, long[] ids, int[] shape)
    {
        using var session = new InferenceSession(modelPath);
        var inputName = session.InputMetadata.Keys.Single();
        using var results = session.Run([NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<long>(ids, shape))]);
        return results.Single().AsTensor<float>().ToArray();
    }

    private static void AssertOutputsClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var index = 0; index < expected.Count; index++)
        {
            var tolerance = 0.002f + Math.Abs(expected[index]) * 0.0002f;
            Assert.True(Math.Abs(expected[index] - actual[index]) <= tolerance,
                $"Output element {index} differed: expected {expected[index]}, got {actual[index]}, tolerance {tolerance}.");
        }
    }

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics) => string.Join(
        Environment.NewLine,
        diagnostics.Select(static diagnostic => $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));

}
