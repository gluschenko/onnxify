using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Compiler;

namespace Onnxify.Tests;

public sealed class CompilerMobileNetRoundTripTests
{
    [Fact]
    public void MobileNetAsset_RoundTripsThroughCompilerTorchSharpAndOnnxRuntime()
    {
        // Runtime semantics are covered by ONNX Runtime Clip, Conv, and GlobalAveragePool provider tests.
        // The source model is the repository fixture src/Onnxify.Tests/Assets/mobilenet_v2_1.4_224.onnx.
        var sourcePath = Path.Combine(
            AppContext.BaseDirectory,
            "Assets",
            "mobilenet_v2_1.4_224.onnx");
        Assert.True(File.Exists(sourcePath), $"Missing MobileNet fixture '{sourcePath}'.");

        var imported = global::Onnxify.Compiler.Compiler.CreateTreeFromOnnx(sourcePath);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var tree = Assert.IsType<CompilerComputationTree>(imported.Value);
        var nodes = tree.Operations.OfType<CompilerOnnxStep>().ToArray();
        Assert.Equal(100, nodes.Length);
        Assert.Equal(52, nodes.Count(static step => step.Node is global::Onnxify.Conv));
        Assert.Equal(35, nodes.Count(static step => step.Node is global::Onnxify.Clip));
        Assert.Single(nodes, static step => step.Node is global::Onnxify.GlobalAveragePool);
        Assert.All(
            nodes,
            static step => Assert.NotEqual(CompilerOperationCapability.Unsupported, step.Descriptor.Capability));

        var exported = global::Onnxify.Compiler.Compiler.GenerateOnnx(tree);
        Assert.True(exported.IsSuccess, FormatDiagnostics(exported.Diagnostics));
        var roundTrippedPath = Path.Combine(
            Path.GetTempPath(),
            $"onnxify-tests-mobilenet-{Guid.NewGuid():N}.onnx");

        try
        {
            exported.Value!.Save(roundTrippedPath, overwrite: true);
            var generated = global::Onnxify.Compiler.Compiler.GenerateCSharp(tree, new CompilerCSharpGenerationOptions
            {
                Namespace = "Onnxify.Tests.GeneratedMobileNet",
                ClassName = "GeneratedMobileNetModule",
                ModuleName = "generated-mobilenet",
            });
            Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
            Assert.Contains("torch.nn.functional.conv2d", generated.Value, StringComparison.Ordinal);
            Assert.Contains(".clamp(", generated.Value, StringComparison.Ordinal);
            Assert.Contains("torch.nn.functional.adaptive_avg_pool2d", generated.Value, StringComparison.Ordinal);

            using var module = CompileGeneratedModule(generated.Value!);
            var values = Enumerable.Range(0, 3 * 224 * 224)
                .Select(static index => (float)(Math.Sin(index * 0.017) * 0.75))
                .ToArray();
            var torchOutput = ExecuteGeneratedModule(module, values);
            var originalOutput = ExecuteOnnx(sourcePath, values);
            var exportedOutput = ExecuteOnnx(roundTrippedPath, values);

            Assert.Equal(1001, torchOutput.Length);
            AssertOutputsClose(originalOutput, torchOutput);
            AssertOutputsClose(originalOutput, exportedOutput);
        }
        finally
        {
            if (File.Exists(roundTrippedPath))
            {
                File.Delete(roundTrippedPath);
            }
        }
    }

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics)
    {
        return string.Join(
            Environment.NewLine,
            diagnostics.Select(static diagnostic =>
                $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));
    }

    private static IDisposable CompileGeneratedModule(string source)
    {
        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))!
            .Split(Path.PathSeparator)
            .Select(static path => MetadataReference.CreateFromFile(path))
            .Append(MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location));
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Tests.GeneratedMobileNet{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(
            emit.Success,
            string.Join(
                Environment.NewLine,
                emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));

        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name == "GeneratedMobileNetModule");
        return (IDisposable)Activator.CreateInstance(moduleType, "generated-mobilenet")!;
    }

    private static float[] ExecuteGeneratedModule(IDisposable module, float[] inputValues)
    {
        using var input = global::TorchSharp.torch.tensor(
            inputValues,
            [1L, 3L, 224L, 224L],
            dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)module.GetType()
            .GetMethod("forward")!
            .Invoke(module, [input])!;
        return output.data<float>().ToArray();
    }

    private static float[] ExecuteOnnx(string modelPath, float[] inputValues)
    {
        using var session = new InferenceSession(modelPath);
        using var results = session.Run(
        [
            NamedOnnxValue.CreateFromTensor(
                "pixel_values",
                new DenseTensor<float>(inputValues, [1, 3, 224, 224])),
        ]);
        return results.Single().AsTensor<float>().ToArray();
    }

    private static void AssertOutputsClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var index = 0; index < expected.Count; index++)
        {
            var tolerance = 0.001f + Math.Abs(expected[index]) * 0.0001f;
            Assert.True(
                Math.Abs(expected[index] - actual[index]) <= tolerance,
                $"Output element {index} differed: expected {expected[index]}, got {actual[index]}, tolerance {tolerance}.");
        }
    }
}
