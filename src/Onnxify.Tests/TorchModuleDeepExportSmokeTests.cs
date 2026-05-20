using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Examples.Models;
using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using TorchModule = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>;
using TorchTensor = TorchSharp.torch.Tensor;

namespace Onnxify.Tests;

public sealed class TorchModuleDeepExportSmokeTests
{
    [Fact]
    public void DeepExport_ForLstmLidExampleModel_MatchesTorchSharpOutput()
    {
        ResetTorchSeed();

        var charToIdx = new Dictionary<string, int>
        {
            ["PAD"] = 0,
            ["a"] = 1,
            ["b"] = 2,
            ["c"] = 3,
            ["d"] = 4,
            ["e"] = 5,
        };
        var langToIdx = new Dictionary<string, int>
        {
            ["en"] = 0,
            ["de"] = 1,
            ["fr"] = 2,
        };

        using var module = new LSTMLIDModel(
            charToIdx,
            langToIdx,
            numClasses: langToIdx.Count,
            embeddingDim: 4,
            hiddenDim: 5,
            layers: 1
        );
        using var input = CreateInt64Tensor(
            [2, 5],
            [1, 2, 3, 0, 4, 2, 1, 0, 3, 5]
        );

        AssertDeepExportMatchesTorchSharp(
            module,
            input,
            inputType: OnnxTensorType.Create<long>([2, 5]),
            outputType: OnnxTensorType.Create<float>([2, langToIdx.Count]),
            absoluteTolerance: 1e-4f
        );
    }

    [Fact]
    public void DeepExport_ForTinyYoloLikeExampleModel_MatchesTorchSharpOutput()
    {
        ResetTorchSeed();

        using var module = new TinyYoloLikeDetector(classCount: 3);
        using var input = CreateFloat32Tensor([1, 3, 64, 64]);

        AssertDeepExportMatchesTorchSharp(
            module,
            input,
            inputType: OnnxTensorType.Create<float>([1, 3, 64, 64]),
            outputType: OnnxTensorType.Create<float>([1, module.PredictionCount, module.AttributeCount]),
            absoluteTolerance: 2e-4f
        );
    }

    [Fact]
    public void DeepExport_ForTorchSharpShowcaseExampleModel_MatchesTorchSharpOutput()
    {
        ResetTorchSeed();

        using var module = new TorchSharpExportShowcase(numClasses: 4);
        using var input = CreateFloat32Tensor([1, 3, 16, 16]);

        AssertDeepExportMatchesTorchSharp(
            module,
            input,
            inputType: OnnxTensorType.Create<float>([1, 3, 16, 16]),
            outputType: OnnxTensorType.Create<float>([1, 4]),
            absoluteTolerance: 5e-2f
        );
    }

    [Fact]
    public void DeepExport_ForMobileNetV1LikeExampleModel_MatchesTorchSharpOutput()
    {
        ResetTorchSeed();

        using var module = new MobileNetV1LikeClassifier(numClasses: 5);
        using var input = CreateFloat32Tensor([1, 3, 96, 96]);

        AssertDeepExportMatchesTorchSharp(
            module,
            input,
            inputType: OnnxTensorType.Create<float>([1, 3, 96, 96]),
            outputType: OnnxTensorType.Create<float>([1, module.NumClasses]),
            absoluteTolerance: 5e-4f
        );
    }

    private static void AssertDeepExportMatchesTorchSharp(
        TorchModule module,
        TorchTensor input,
        OnnxTensorType inputType,
        OnnxTensorType outputType,
        float absoluteTolerance
    )
    {
        module.eval();

        using var torchOutput = module.forward(input);
        using var detachedOutput = torchOutput.detach();
        using var cpuOutput = detachedOutput.cpu();
        var expected = cpuOutput.data<float>().ToArray();
        var expectedShape = ToIntShape(cpuOutput.shape);

        var model = module.Export(
            input: inputType,
            output: outputType,
            options: new OnnxModelCreationOptions
            {
                Opset = 22,
                ProducerName = "onnxify-tests",
            }
        );

        var actual = RunOnnxModel(model, input);

        Assert.Equal(expectedShape, actual.Dimensions.ToArray());
        Assert.Equal(expected.Length, actual.Length);

        var actualValues = actual.Buffer.ToArray();
        var maxDifference = 0f;
        var maxIndex = -1;

        for (var index = 0; index < expected.Length; index++)
        {
            var difference = MathF.Abs(expected[index] - actualValues[index]);
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
        var shape = ToIntShape(input.shape);
        return input.dtype switch
        {
            ScalarType.Float32 => NamedOnnxValue.CreateFromTensor(
                "input",
                new DenseTensor<float>(input.data<float>().ToArray(), shape)
            ),
            ScalarType.Int64 => NamedOnnxValue.CreateFromTensor(
                "input",
                new DenseTensor<long>(input.data<long>().ToArray(), shape)
            ),
            _ => throw new NotSupportedException($"Unsupported smoke-test input dtype '{input.dtype}'."),
        };
    }

    private static TorchTensor CreateFloat32Tensor(long[] shape)
    {
        var elementCount = checked((int)shape.Aggregate(1L, static (total, dimension) => total * dimension));
        var values = Enumerable.Range(0, elementCount)
            .Select(static index => ((index % 251) - 125) / 125f)
            .ToArray();

        return torch.tensor(values, shape, dtype: ScalarType.Float32);
    }

    private static TorchTensor CreateInt64Tensor(long[] shape, long[] values)
    {
        return torch.tensor(values, shape, dtype: ScalarType.Int64);
    }

    private static int[] ToIntShape(IReadOnlyList<long> shape)
    {
        return shape.Select(static dimension => checked((int)dimension)).ToArray();
    }
}
