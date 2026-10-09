using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class RepositoryAssetRoundTripTests
{
    public static IEnumerable<object[]> RepositoryModels =>
    [
        ["mobilenet_v2_1.4_224.onnx"],
        ["yolo26s.onnx"],
    ];

    [Theory]
    [MemberData(nameof(RepositoryModels))]
    public void RepositoryAssetRoundTripsThroughCompiler(string fileName)
    {
        var sourcePath = Path.Combine(AppContext.BaseDirectory, "Assets", fileName);
        Assert.True(File.Exists(sourcePath), $"Missing compiler round-trip fixture '{sourcePath}'.");

        // Source: src/Onnxify.Tests/OnnxModelTests.cs
        // (FromFile_WithMobileNetDuplicatedLogitsValueInfo_LoadsGraph and
        // FromFile_WithYolo26sNoneDimension_LoadsExportsAndPrintsGraph).
        // No direct ONNXScript converter test exists for these repository model fixtures;
        // the compiler boundary is validated by core model reload and ONNX Runtime.
        var original = OnnxModel.FromFile(
            sourcePath,
            new OnnxModelBaseOptions
            {
                NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.PreserveUntyped,
            });

        var imported = Compiler.CreateTreeFromOnnx(sourcePath);

        Assert.True(
            imported.IsSuccess,
            FormatDiagnostics(imported.Diagnostics));
        Assert.NotNull(imported.Value);

        var emitted = Compiler.GenerateOnnx(imported.Value!);

        Assert.True(
            emitted.IsSuccess,
            FormatDiagnostics(emitted.Diagnostics));
        Assert.NotNull(emitted.Value);

        var roundTrippedPath = Path.Combine(
            Path.GetTempPath(),
            $"onnxify-compiler-asset-{Guid.NewGuid():N}.onnx");

        try
        {
            emitted.Value!.Save(roundTrippedPath, overwrite: true);
            var roundTripped = OnnxModel.FromFile(
                roundTrippedPath,
                new OnnxModelBaseOptions
                {
                    NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.PreserveUntyped,
                });

            Assert.Equal(original.Graph.Name, roundTripped.Graph.Name);
            Assert.Equal(
                original.Graph.Inputs.Select(static value => value.Name),
                roundTripped.Graph.Inputs.Select(static value => value.Name));
            Assert.Equal(
                original.Graph.Outputs.Select(static value => value.Name),
                roundTripped.Graph.Outputs.Select(static value => value.Name));
            Assert.Equal(original.Graph.Nodes.Count, roundTripped.Graph.Nodes.Count);
            Assert.Equal(
                original.Graph.Nodes.Select(NodeSignature),
                roundTripped.Graph.Nodes.Select(NodeSignature));
            Assert.Equal(
                original.Graph.Initializers.Select(static initializer => initializer.Name),
                roundTripped.Graph.Initializers.Select(static initializer => initializer.Name));
            Assert.Equal(
                original.Graph.SparseInitializers.Select(static initializer => initializer.Name),
                roundTripped.Graph.SparseInitializers.Select(static initializer => initializer.Name));

            using var session = new global::Microsoft.ML.OnnxRuntime.InferenceSession(roundTrippedPath);
            Assert.NotEmpty(session.InputMetadata);
            Assert.NotEmpty(session.OutputMetadata);
        }
        finally
        {
            if (File.Exists(roundTrippedPath))
            {
                File.Delete(roundTrippedPath);
            }
        }
    }

    [Fact]
    public void MobileNetRoundTripsThroughGeneratedTorchSharpAndOnnxRuntime()
    {
        // Source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py
        // (clamp, nn.functional.conv2d, and nn.functional.avg_pool2d). GlobalAveragePool has no direct
        // ONNXScript case in this table; its semantics follow the ONNX operator definition.
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/clip_test.cc (Clip),
        // nn/conv_op_test.cc (Conv2D_1), and nn/pool_op_test.cc (GlobalAveragePool).
        var sourcePath = Path.Combine(AppContext.BaseDirectory, "Assets", "mobilenet_v2_1.4_224.onnx");
        var original = OnnxModel.FromFile(
            sourcePath,
            new OnnxModelBaseOptions
            {
                NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.PreserveUntyped,
            });
        var imported = Compiler.CreateTreeFromOnnx(sourcePath);

        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var tree = Assert.IsType<CompilerComputationTree>(imported.Value);
        var operations = tree.Operations.OfType<CompilerOnnxStep>().ToArray();
        Assert.Equal(100, operations.Length);
        Assert.All(
            operations,
            operation => Assert.NotEqual(CompilerOperationCapability.Unsupported, operation.Descriptor.Capability));
        Assert.Equal(52, operations.Count(static operation => operation.Node is global::Onnxify.Conv));
        Assert.Equal(35, operations.Count(static operation => operation.Node is global::Onnxify.Clip));
        Assert.Single(operations, static operation => operation.Node is global::Onnxify.GlobalAveragePool);

        var emitted = Compiler.GenerateOnnx(tree);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        var roundTrippedPath = Path.Combine(
            Path.GetTempPath(),
            $"onnxify-compiler-mobilenet-{Guid.NewGuid():N}.onnx");

        try
        {
            emitted.Value!.Save(roundTrippedPath, overwrite: true);
            var roundTripped = OnnxModel.FromFile(
                roundTrippedPath,
                new OnnxModelBaseOptions
                {
                    NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.PreserveUntyped,
                });
            Assert.Equal(
                original.Graph.Inputs.Select(static value => value.Name),
                roundTripped.Graph.Inputs.Select(static value => value.Name));
            Assert.Equal(
                original.Graph.Outputs.Select(static value => value.Name),
                roundTripped.Graph.Outputs.Select(static value => value.Name));
            Assert.Equal(original.Graph.Nodes.Select(NodeSignature), roundTripped.Graph.Nodes.Select(NodeSignature));
            Assert.Equal(
                original.Graph.Initializers.Select(static initializer => initializer.Name),
                roundTripped.Graph.Initializers.Select(static initializer => initializer.Name));

            var generated = Compiler.GenerateCSharp(tree, new CompilerCSharpGenerationOptions
            {
                Namespace = "Onnxify.GeneratedMobileNetRoundTrip",
                ClassName = "GeneratedMobileNetTorchModule",
                ModuleName = "generated-mobilenet",
            });
            Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
            Assert.Contains("torch.nn.functional.conv2d", generated.Value, StringComparison.Ordinal);
            Assert.Contains(".clamp(", generated.Value, StringComparison.Ordinal);
            Assert.Contains("torch.nn.functional.adaptive_avg_pool2d", generated.Value, StringComparison.Ordinal);

            using var generatedModule = CompileGeneratedModule(generated.Value!);
            var inputValues = Enumerable.Range(0, 3 * 224 * 224)
                .Select(static index => (float)(Math.Sin(index * 0.017) * 0.75))
                .ToArray();
            var torchSharpOutput = ExecuteGeneratedModule(generatedModule, inputValues);
            var originalOutput = ExecuteOnnx(sourcePath, inputValues);
            var roundTrippedOutput = ExecuteOnnx(roundTrippedPath, inputValues);

            Assert.Equal(1001, torchSharpOutput.Length);
            AssertOutputsClose(originalOutput, torchSharpOutput);
            AssertOutputsClose(originalOutput, roundTrippedOutput);
        }
        finally
        {
            if (File.Exists(roundTrippedPath))
            {
                File.Delete(roundTrippedPath);
            }
        }
    }

    private static string NodeSignature(OnnxNode node)
    {
        var attributes = node.Attributes
            .OrderBy(static attribute => attribute.Name, StringComparer.Ordinal)
            .Select(static attribute => attribute.Name);

        return string.Join(
            "|",
            node.Name,
            node.Domain,
            node.OpType,
            string.Join(",", node.Inputs.Select(static input => input.Name)),
            string.Join(",", node.Outputs.Select(static output => output.Name)),
            string.Join(",", attributes));
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
            $"Onnxify.GeneratedMobileNet{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(
            emit.Success,
            string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));

        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name == "GeneratedMobileNetTorchModule");
        return (IDisposable)Activator.CreateInstance(moduleType, "generated-mobilenet")!;
    }

    private static float[] ExecuteGeneratedModule(IDisposable module, float[] inputValues)
    {
        var moduleType = module.GetType();
        using var input = global::TorchSharp.torch.tensor(
            inputValues,
            [1L, 3L, 224L, 224L],
            dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType
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
