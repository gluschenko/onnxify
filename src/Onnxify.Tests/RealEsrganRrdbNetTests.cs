using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Examples.Models;
using TorchSharp;
using static TorchSharp.torch;
using TorchTensor = TorchSharp.torch.Tensor;

namespace Onnxify.Tests;

public sealed class RealEsrganRrdbNetTests
{
    [Theory]
    [InlineData(1, 8, 8, 8, 8)]
    [InlineData(2, 8, 8, 16, 16)]
    [InlineData(4, 8, 8, 32, 32)]
    public void Forward_ForSmallModel_UpscalesByConfiguredScale(
        int scale,
        long inputHeight,
        long inputWidth,
        long outputHeight,
        long outputWidth
    )
    {
        ResetTorchSeed();

        using var module = CreateSmallModel(scale);
        module.eval();
        using var input = CreateFloat32Tensor([1, 3, inputHeight, inputWidth]);
        using var output = module.forward(input);

        Assert.Equal([1, 3, outputHeight, outputWidth], output.shape.ToArray());
    }

    [Fact]
    public void Export_ForSmallModel_MatchesTorchSharpOutput()
    {
        ResetTorchSeed();

        using var module = CreateSmallModel(scale: 4);
        module.eval();
        using var input = CreateFloat32Tensor([1, 3, 8, 8]);
        using var torchOutput = module.forward(input);
        using var detachedOutput = torchOutput.detach();
        using var cpuOutput = detachedOutput.cpu();
        var expected = cpuOutput.data<float>().ToArray();
        var expectedShape = ToIntShape(cpuOutput.shape);

        var model = module.Export(inputHeight: 8, inputWidth: 8);
        var actual = RunOnnxModel(model, input);

        Assert.Equal(expectedShape, actual.Dimensions.ToArray());
        Assert.Equal(expected.Length, actual.Length);
        AssertClose(expected, actual.Buffer.ToArray(), absoluteTolerance: 2e-4f);
    }

    [Fact]
    public void FactoryMethods_CreateKnownRealEsrganVariants()
    {
        using var x4Plus = RealEsrganRrdbNet.CreateX4Plus();
        using var x4PlusAnime6B = RealEsrganRrdbNet.CreateX4PlusAnime6B();
        using var x2Plus = RealEsrganRrdbNet.CreateX2Plus();

        Assert.Equal(4, x4Plus.Scale);
        Assert.Equal(23, x4Plus.BlockCount);
        Assert.Equal(4, x4PlusAnime6B.Scale);
        Assert.Equal(6, x4PlusAnime6B.BlockCount);
        Assert.Equal(2, x2Plus.Scale);
        Assert.Equal(23, x2Plus.BlockCount);
    }

    [Fact]
    public void Constructor_ForUnsupportedScale_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new RealEsrganRrdbNet(scale: 3).Dispose()
        );
    }

    private static RealEsrganRrdbNet CreateSmallModel(int scale)
    {
        return new RealEsrganRrdbNet(
            featureChannels: 4,
            blockCount: 1,
            growthChannels: 2,
            scale: scale
        );
    }

    private static void ResetTorchSeed()
    {
        torch.random.manual_seed(12_345);
    }

    private static DenseTensor<float> RunOnnxModel(OnnxModel model, TorchTensor input)
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            model.Save(path);

            using var session = new InferenceSession(path);
            using var results = session.Run([CreateNamedInput(input)]);
            var output = Assert.Single(results);
            var tensor = output.AsTensor<float>();

            return new DenseTensor<float>(tensor.ToArray(), tensor.Dimensions.ToArray());
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static NamedOnnxValue CreateNamedInput(TorchTensor input)
    {
        return NamedOnnxValue.CreateFromTensor(
            "input",
            new DenseTensor<float>(input.data<float>().ToArray(), ToIntShape(input.shape))
        );
    }

    private static TorchTensor CreateFloat32Tensor(long[] shape)
    {
        var elementCount = checked((int)shape.Aggregate(1L, static (total, dimension) => total * dimension));
        var values = Enumerable.Range(0, elementCount)
            .Select(static index => ((index % 251) - 125) / 125f)
            .ToArray();

        return torch.tensor(values, shape, dtype: ScalarType.Float32);
    }

    private static int[] ToIntShape(IReadOnlyList<long> shape)
    {
        return shape.Select(static dimension => checked((int)dimension)).ToArray();
    }

    private static void AssertClose(float[] expected, float[] actual, float absoluteTolerance)
    {
        var maxDifference = 0f;
        var maxIndex = -1;

        for (var index = 0; index < expected.Length; index++)
        {
            var difference = MathF.Abs(expected[index] - actual[index]);
            if (difference > maxDifference)
            {
                maxDifference = difference;
                maxIndex = index;
            }
        }

        Assert.True(
            maxDifference <= absoluteTolerance,
            $"Maximum absolute difference {maxDifference} at flat index {maxIndex} exceeded tolerance {absoluteTolerance}."
        );
    }
}
