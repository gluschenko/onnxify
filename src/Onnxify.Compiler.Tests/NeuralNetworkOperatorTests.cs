using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class NeuralNetworkOperatorTests
{
    [Fact]
    public void LstmScansCanonicalCompilerTorchSharpFormAndMatchesOnnxRuntime()
    {
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor weights, global::TorchSharp.torch.Tensor recurrent, global::TorchSharp.torch.Tensor bias) { return torch.nn.functional.lstm(input, weights, recurrent, bias, 5L, \"bidirectional\", 0L); }";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("LSTM", operation.Node.OpType);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);
        Assert.Equal("bidirectional", operation.Node.Attributes.Single(attribute => attribute.Name == "direction").GetValue());
        Assert.Equal(5L, operation.Node.Attributes.Single(attribute => attribute.Name == "hidden_size").GetValue());
        Assert.Equal(4, operation.Inputs.Count);

        var exported = Compiler.GenerateOnnx(imported.Value);

        Assert.True(exported.IsSuccess, FormatDiagnostics(exported.Diagnostics));
        Assert.Single(exported.Value!.Graph.Nodes, node => node.OpType == "LSTM");

        var path = Path.Combine(Path.GetTempPath(), $"oxy039-lstm-operator-{Guid.NewGuid():N}.onnx");
        try
        {
            exported.Value.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var inputValues = Enumerable.Range(0, 3 * 2 * 4).Select(static index => (index - 8) * 0.03f).ToArray();
            var weightValues = Enumerable.Range(0, 2 * 4 * 5 * 4).Select(static index => (index % 13 - 6) * 0.02f).ToArray();
            var recurrentValues = Enumerable.Range(0, 2 * 4 * 5 * 5).Select(static index => (index % 11 - 5) * 0.015f).ToArray();
            var biasValues = Enumerable.Range(0, 2 * 8 * 5).Select(static index => (index % 7 - 3) * 0.01f).ToArray();
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(inputValues, [3, 2, 4])),
                NamedOnnxValue.CreateFromTensor("weights", new DenseTensor<float>(weightValues, [2, 20, 4])),
                NamedOnnxValue.CreateFromTensor("recurrent", new DenseTensor<float>(recurrentValues, [2, 20, 5])),
                NamedOnnxValue.CreateFromTensor("bias", new DenseTensor<float>(biasValues, [2, 40])),
            ]);
            var expected = results.Single().AsTensor<float>().ToArray();

            var generated = Compiler.GenerateCSharp(imported.Value, new CompilerCSharpGenerationOptions
            {
                Namespace = "Onnxify.GeneratedLstmOperator",
                ClassName = "GeneratedLstmOperator",
                ModuleName = "generated-lstm-operator",
            });
            Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
            using var module = CompileGeneratedModule(generated.Value!);
            using var inputTensor = global::TorchSharp.torch.tensor(inputValues, [3L, 2L, 4L]);
            using var weightsTensor = global::TorchSharp.torch.tensor(weightValues, [2L, 20L, 4L]);
            using var recurrentTensor = global::TorchSharp.torch.tensor(recurrentValues, [2L, 20L, 5L]);
            using var biasTensor = global::TorchSharp.torch.tensor(biasValues, [2L, 40L]);
            using var actualTensor = (global::TorchSharp.torch.Tensor)module.GetType().GetMethod("forward")!
                .Invoke(module, [inputTensor, weightsTensor, recurrentTensor, biasTensor])!;
            var actual = actualTensor.data<float>().ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < expected.Length; index++)
            {
                Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0f, 0.002f);
            }
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
            $"GeneratedLstmOperator{Guid.NewGuid():N}", [CSharpSyntaxTree.ParseText(source)], references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var result = compilation.Emit(assemblyBytes);
        Assert.True(result.Success, string.Join(Environment.NewLine,
            result.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)) + Environment.NewLine + source);
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var type = assembly.GetTypes().Single(candidate => candidate.Name == "GeneratedLstmOperator");
        return (IDisposable)Activator.CreateInstance(type, "generated-lstm-operator")!;
    }

    [Theory]
    [InlineData("torch.clamp(input, -1f, 2f)")]
    [InlineData("input.clamp(-1f, 2f)")]
    public void ClipScansTorchSharpAndExportsTypedNode(string expression)
    {
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/clip_test.cc (Clip).
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) {{ return {expression}; }}";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Clip>(operation.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var exported = Compiler.GenerateOnnx(imported.Value);

        Assert.True(exported.IsSuccess, FormatDiagnostics(exported.Diagnostics));
        Assert.IsType<Clip>(Assert.Single(exported.Value!.Graph.Nodes));
        var values = new[] { -3f, 0f, 4f };
        using var inputTensor = global::TorchSharp.torch.tensor(values, [values.Length], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var torchOutput = inputTensor.clamp(-1f, 2f);
        Assert.Equal(torchOutput.data<float>().ToArray(), ExecuteClip(exported.Value!, values));
    }

    [Fact]
    public void ConvScansTorchSharpAndExportsTypedNode()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_conv2d).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/nn/conv_op_test.cc (Conv2D_1).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor weight, global::TorchSharp.torch.Tensor bias) { return torch.nn.functional.conv2d(input, weight, bias, new long[] { 1L, 1L }, new long[] { 0L, 0L }, new long[] { 1L, 1L }, 1L); }";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Conv>(operation.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var exported = Compiler.GenerateOnnx(imported.Value);

        Assert.True(exported.IsSuccess, FormatDiagnostics(exported.Diagnostics));
        Assert.IsType<Conv>(Assert.Single(exported.Value!.Graph.Nodes));
        var inputValues = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f };
        var weightValues = new[] { 1f, 0f, 0f, 1f };
        var biasValues = new[] { 0f };
        using var inputTensor = global::TorchSharp.torch.tensor(inputValues, [1L, 1L, 3L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var weightTensor = global::TorchSharp.torch.tensor(weightValues, [1L, 1L, 2L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var biasTensor = global::TorchSharp.torch.tensor(biasValues, [1L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var torchOutput = global::TorchSharp.torch.nn.functional.conv2d(
            inputTensor,
            weightTensor,
            biasTensor,
            [1L, 1L],
            [0L, 0L],
            [1L, 1L],
            1L);
        Assert.Equal(torchOutput.data<float>().ToArray(), ExecuteConv(exported.Value!, inputValues, weightValues, biasValues));
    }

    [Fact]
    public void ConvScansGeneratedAsymmetricPaddingDecomposition()
    {
        // TorchSharp generated form from ONNX Conv: third_party/onnxruntime/onnxruntime/test/providers/cpu/nn/conv_op_test.cc (Conv2D_1).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor weight) { return torch.nn.functional.conv2d(torch.nn.functional.pad(input, new long[] { 1L, 0L, 0L, 0L }), weight, null, new long[] { 1L, 1L }, new long[] { 0L, 0L }, new long[] { 1L, 1L }, 1L); }";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        var conv = Assert.IsType<Conv>(operation.Node);
        Assert.Equal([0L, 1L, 0L, 0L], conv.Pads!);

        var exported = Compiler.GenerateOnnx(imported.Value);

        Assert.True(exported.IsSuccess, FormatDiagnostics(exported.Diagnostics));
        Assert.IsType<Conv>(Assert.Single(exported.Value!.Graph.Nodes));
    }

    [Fact]
    public void ClipScansGeneratedScalarTensorBounds()
    {
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor minimum, global::TorchSharp.torch.Tensor maximum) { return input.clamp(minimum.item<float>(), maximum.item<float>()); }";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        var clip = Assert.IsType<Clip>(operation.Node);
        Assert.Equal("minimum", clip.Min!.Name);
        Assert.Equal("maximum", clip.Max!.Name);
    }

    [Fact]
    public void GlobalAveragePoolScansOnlyExactTorchSharpOutputSize()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_adaptive_avg_pool2d).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/nn/pool_op_test.cc (GlobalAveragePool).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return torch.nn.functional.adaptive_avg_pool2d(input, new long[] { 1L, 1L }); }";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<GlobalAveragePool>(operation.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var exported = Compiler.GenerateOnnx(imported.Value);

        Assert.True(exported.IsSuccess, FormatDiagnostics(exported.Diagnostics));
        Assert.IsType<GlobalAveragePool>(Assert.Single(exported.Value!.Graph.Nodes));
        var inputValues = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f };
        using var inputTensor = global::TorchSharp.torch.tensor(inputValues, [1L, 1L, 3L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var torchOutput = global::TorchSharp.torch.nn.functional.adaptive_avg_pool2d(inputTensor, [1L, 1L]);
        Assert.Equal(torchOutput.data<float>().ToArray(), ExecuteGlobalAveragePool(exported.Value!, inputValues));

        var unsupported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(
            "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return torch.nn.functional.adaptive_avg_pool2d(input, new long[] { 2L, 2L }); }"));
        AssertUnsupportedDiagnostic(unsupported, "[1, 1]");
        Assert.NotNull(Assert.Single(unsupported.Diagnostics).Span);
    }

    [Theory]
    [InlineData("public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, float minimum) { return torch.clamp(input, minimum, 2f); }")]
    [InlineData("public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.clamp(max: 2f, min: -1f); }")]
    [InlineData("public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor weight, long padding) { return torch.nn.functional.conv2d(input, weight, null, 1L, padding, 1L, 1L); }")]
    [InlineData("public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor weight) { return torch.nn.functional.conv2d(input: input, weight: weight); }")]
    public void TorchSharpNeuralNetworkOperatorsRejectDynamicParameters(string source)
    {
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.NotNull(diagnostic.Span);
    }

    [Fact]
    public void ConvRejectsUnsupportedAutoPadding()
    {
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/nn/conv_op_test.cc (Conv2D_1).
        var model = OnnxModel.Create();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 1, 5, 5]));
        var weight = model.Graph.AddInput("weight", OnnxTensorType.Create<float>([1, 1, 3, 3]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 1, 3, 3]));
        model.Graph.AddNode(new Conv(
            "conv",
            new ConvInputOutputOptions
            {
                X = input,
                W = weight,
                AutoPad = "SAME_UPPER",
                Y = output,
            }));

        var result = Compiler.CreateTreeFromOnnx(model);

        AssertUnsupportedDiagnostic(result, "auto_pad");
    }

    [Fact]
    public void ClipAcceptsSingleElementBounds()
    {
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/clip_test.cc (Clip).
        var model = OnnxModel.Create();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([4]));
        var min = model.Graph.AddInput("min", OnnxTensorType.Create<float>([1]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([4]));
        model.Graph.AddNode(new Clip(
            "clip",
            new ClipInputOutputOptions
            {
                Input = input,
                Min = min,
                Output = output,
            }));

        var result = Compiler.CreateTreeFromOnnx(model);
        Assert.True(result.IsSuccess, FormatDiagnostics(result.Diagnostics));

        var generated = Compiler.GenerateCSharp(result.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("min.item<float>()", generated.Value, StringComparison.Ordinal);
    }

    [Fact]
    public void GlobalAveragePoolRejectsNonImageRank()
    {
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/nn/pool_op_test.cc (GlobalAveragePool).
        var model = OnnxModel.Create();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 3, 5]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 3, 1]));
        model.Graph.AddNode(new GlobalAveragePool(
            "global-average-pool",
            new GlobalAveragePoolInputOutputOptions
            {
                X = input,
                Y = output,
            }));

        var result = Compiler.CreateTreeFromOnnx(model);

        AssertUnsupportedDiagnostic(result, "rank-4 NCHW");
    }

    private static void AssertUnsupportedDiagnostic(
        CompilerResult<CompilerComputationTree> result,
        string messageFragment)
    {
        Assert.False(result.IsSuccess);
        Assert.Contains(
            result.Diagnostics,
            diagnostic => diagnostic.Code == CompilerDiagnosticCodes.Unsupported
                && diagnostic.Message.Contains(messageFragment, StringComparison.Ordinal));
    }

    private static float[] ExecuteClip(OnnxModel model, float[] input)
    {
        return ExecuteOnnx(model, [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(input, [input.Length]))]);
    }

    private static float[] ExecuteConv(OnnxModel model, float[] input, float[] weight, float[] bias)
    {
        return ExecuteOnnx(model,
        [
            NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(input, [1, 1, 3, 3])),
            NamedOnnxValue.CreateFromTensor("weight", new DenseTensor<float>(weight, [1, 1, 2, 2])),
            NamedOnnxValue.CreateFromTensor("bias", new DenseTensor<float>(bias, [1])),
        ]);
    }

    private static float[] ExecuteGlobalAveragePool(OnnxModel model, float[] input)
    {
        return ExecuteOnnx(model,
        [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(input, [1, 1, 3, 3]))]);
    }

    private static float[] ExecuteOnnx(OnnxModel model, IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-network-operator-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(inputs);
            return results.Single().AsTensor<float>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics)
    {
        return string.Join(Environment.NewLine, diagnostics.Select(static diagnostic =>
            $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));
    }
}
