using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class MobileNetV2DynamicBatchRoundTripTests
{
    [Fact]
    public void MobileNetV2TwelveRunsThroughGeneratedTorchSharpAndOnnxRuntime()
    {
        // Fixture: src/Onnxify.Tests/Assets/mobilenetv2-12.onnx (opset-12 dynamic-batch model).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/clip_test.cc (Clip),
        // third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/shape_op_test.cc (Shape),
        // and third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/unsqueeze_op_test.cc (legacy axes attribute).
        var path = Path.Combine(AppContext.BaseDirectory, "Assets", "mobilenetv2-12.onnx");
        using var sourceSession = new InferenceSession(path);
        var inputName = sourceSession.InputMetadata.Keys.Single();
        var inputDimensions = sourceSession.InputMetadata[inputName].Dimensions;
        Assert.True(inputDimensions[0] < 0, "Fixture input is expected to have a dynamic batch dimension.");
        var shape = inputDimensions.Select(x => x < 0 ? 1 : (int)x).ToArray();
        var imported = Compiler.CreateTreeFromOnnx(path);
        if (!imported.IsSuccess)
        {
            Assert.Fail(string.Join(Environment.NewLine, imported.Diagnostics.Select(x => x.Message)));
        }
        var tree = (CompilerComputationTree)imported.Value!;
        var operations = tree.Operations.OfType<CompilerOnnxStep>().ToArray();
        Assert.All(operations, operation => Assert.NotEqual(CompilerOperationCapability.Unsupported, operation.Descriptor.Capability));
        Assert.Equal(35, operations.Count(operation => operation.Node.OpType == "Clip"));
        Assert.Single(operations, operation => operation.Node.OpType == "Constant");
        Assert.Single(operations, operation => operation.Node.OpType == "Unsqueeze");
        Assert.Single(operations, operation => operation.Node.OpType == "Shape");
        var emitted = Compiler.GenerateOnnx(tree);
        Assert.True(emitted.IsSuccess, string.Join(Environment.NewLine, emitted.Diagnostics.Select(x => x.Message)));
        var roundTripPath = Path.Combine(Path.GetTempPath(), $"oxy041-{Guid.NewGuid():N}.onnx");
        try
        {
            emitted.Value!.Save(roundTripPath, overwrite: true);
            var generated = Compiler.GenerateSourceCode(tree, new CompilerCSharpGenerationOptions
            {
                Namespace = "Onnxify.GeneratedDynamicMobileNet",
                ClassName = "GeneratedDynamicMobileNet",
                ModuleName = "generated-dynamic-mobilenet",
            });
            Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(x => x.Message)));
            var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))!.Split(Path.PathSeparator)
                .Select(static p => MetadataReference.CreateFromFile(p)).Append(MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location));
            var compilation = CSharpCompilation.Create($"Probe{Guid.NewGuid():N}", [CSharpSyntaxTree.ParseText(generated.SourceCode())], references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
            using var bytes = new MemoryStream();
            var result = compilation.Emit(bytes);
            Assert.True(result.Success, string.Join(Environment.NewLine, result.Diagnostics.Where(x => x.Severity == DiagnosticSeverity.Error)));
            bytes.Position = 0;
            var assembly = AssemblyLoadContext.Default.LoadFromStream(bytes);
            var type = assembly.GetTypes().Single(x => x.Name == "GeneratedDynamicMobileNet");
            using var module = (IDisposable)Activator.CreateInstance(type, path, "generated-dynamic-mobilenet")!;
            using var roundTripSession = new InferenceSession(roundTripPath);
            Assert.NotEmpty(roundTripSession.InputMetadata);
            Assert.NotEmpty(roundTripSession.OutputMetadata);

            var values = Enumerable.Range(0, shape.Aggregate(1, static (a, b) => a * b))
                .Select(static i => (i % 251) / 250f)
                .ToArray();
            using var torchInput = global::TorchSharp.torch.tensor(
                values,
                shape.Select(static x => (long)x).ToArray(),
                dtype: global::TorchSharp.torch.ScalarType.Float32);
            using var torchResult = (global::TorchSharp.torch.Tensor)type.GetMethod("forward")!.Invoke(module, [torchInput])!;
            var actual = torchResult.data<float>().ToArray();
            using var sourceResult = sourceSession.Run(
            [
                NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(values, shape)),
            ]);
            var expectedTensor = sourceResult.Single().AsTensor<float>();
            var expected = expectedTensor.ToArray();
            using var exportedResult = roundTripSession.Run(
            [
                NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(values, shape)),
            ]);
            var exported = exportedResult.Single().AsTensor<float>().ToArray();
            Assert.Equal([1, 1000], expectedTensor.Dimensions.ToArray());
            Assert.Equal([1L, 1000L], torchResult.shape);
            Assert.Equal([1, 1000], exportedResult.Single().AsTensor<float>().Dimensions.ToArray());
            AssertOutputsClose(expected, actual, 0.001f);
            AssertOutputsClose(expected, exported, 0.00001f);
        }
        finally { if (File.Exists(roundTripPath)) File.Delete(roundTripPath); }
    }

    private static void AssertOutputsClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float absoluteTolerance
    )
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var tolerance = absoluteTolerance + Math.Abs(expected[i]) * 0.0001f;
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"output[{i}]: expected {expected[i]}, got {actual[i]}, tolerance {tolerance}");
        }
    }
}
