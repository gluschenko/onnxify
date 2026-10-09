using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class NeuralNetworkOperatorTests
{
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
    public void ClipRejectsNonScalarBounds()
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

        AssertUnsupportedDiagnostic(result, "scalar min and max");
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
}
