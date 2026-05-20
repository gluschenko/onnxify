using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Examples.Models;
using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
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
    public void DeepExport_ForDocumentedTransformerSyntaxExample_MatchesTorchSharpOutput()
    {
        ResetTorchSeed();

        using var module = new DocumentedTransformerSyntaxModule();
        using var input = CreateInt64Tensor(
            [2, module.MaxSequenceLength],
            [0, 1, 2, 3, 4, 5, 6, 7]
        );

        AssertDeepExportMatchesTorchSharp(
            module,
            input,
            inputType: OnnxTensorType.Create<long>([2, module.MaxSequenceLength]),
            outputType: OnnxTensorType.Create<float>([2, module.MaxSequenceLength, module.VocabularySize]),
            absoluteTolerance: 5e-4f
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

        var model = module.ExportOnnxModel(
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

    private sealed class DocumentedTransformerSyntaxModule : TorchModule
    {
        private const int VOCABULARY_SIZE = 17;
        private const int MAX_SEQUENCE_LENGTH = 4;
        private const int ATTENTION_HEADS = 2;
        private const int ATTENTION_DIMENSIONS = 8;

        private readonly global::TorchSharp.Modules.Embedding _tokenEmbedding;
        private readonly global::TorchSharp.Modules.Embedding _positionEmbedding;
        private readonly DocumentedAttentionBlock _block;
        private readonly global::TorchSharp.Modules.LayerNorm _outputNorm;

        public DocumentedTransformerSyntaxModule()
            : base(nameof(DocumentedTransformerSyntaxModule))
        {
            _tokenEmbedding = Embedding(VOCABULARY_SIZE, ATTENTION_DIMENSIONS);
            _positionEmbedding = Embedding(MAX_SEQUENCE_LENGTH, ATTENTION_DIMENSIONS);
            _block = new DocumentedAttentionBlock(
                ATTENTION_HEADS,
                ATTENTION_DIMENSIONS,
                MAX_SEQUENCE_LENGTH
            );
            _outputNorm = LayerNorm([ATTENTION_DIMENSIONS]);

            RegisterComponents();
        }

        public int VocabularySize => VOCABULARY_SIZE;

        public int MaxSequenceLength => MAX_SEQUENCE_LENGTH;

        public override TorchTensor forward(TorchTensor tokens)
        {
            ValidateInputShape(tokens);

            using var positions = CreatePositionIds(tokens.shape[0], tokens.device);

            var x = _tokenEmbedding.forward(tokens) + _positionEmbedding.forward(positions);
            x = _block.forward(x);
            x = _outputNorm.forward(x);

            return ComputeLogits(x);
        }

        private TorchTensor ComputeLogits(TorchTensor hiddenStates)
        {
            using var tiedWeight = _tokenEmbedding.weight!.transpose(0, 1);
            return matmul(hiddenStates, tiedWeight);
        }

        private TorchTensor CreatePositionIds(long batchSize, Device device)
        {
            return arange(MAX_SEQUENCE_LENGTH, dtype: ScalarType.Int64, device: device)
                .unsqueeze(0)
                .expand(batchSize, MAX_SEQUENCE_LENGTH);
        }

        private static void ValidateInputShape(TorchTensor tokens)
        {
            if (tokens.shape.Length != 2)
            {
                throw new ArgumentException("Expected token ids with rank 2.", nameof(tokens));
            }
        }
    }

    private sealed class DocumentedAttentionBlock : TorchModule
    {
        private readonly int _headCount;
        private readonly int _headDimension;
        private readonly int _attentionDimensions;
        private readonly int _maxSequenceLength;
        private readonly float _scale;
        private readonly global::TorchSharp.Modules.Linear _attention;
        private readonly global::TorchSharp.Modules.Linear _projection;

        public DocumentedAttentionBlock(
            int headCount,
            int attentionDimensions,
            int maxSequenceLength
        ) : base(nameof(DocumentedAttentionBlock))
        {
            _headCount = headCount;
            _headDimension = attentionDimensions / headCount;
            _attentionDimensions = attentionDimensions;
            _maxSequenceLength = maxSequenceLength;
            _scale = 1.0f / MathF.Sqrt(_headDimension);
            _attention = Linear(attentionDimensions, attentionDimensions * 3);
            _projection = Linear(attentionDimensions, attentionDimensions);

            RegisterComponents();
        }

        public override TorchTensor forward(TorchTensor x)
        {
            var batchSize = x.shape[0];
            var sequenceLength = x.shape[1];

            var qkv = _attention.forward(x)
                .view([batchSize, sequenceLength, 3, _headCount, _headDimension])
                .permute(2, 0, 3, 1, 4);

            var query = qkv[0];
            var key = qkv[1];
            var value = qkv[2];

            using var causalMask =
                triu(
                    full(
                        [_maxSequenceLength, _maxSequenceLength],
                        -10_000f,
                        dtype: x.dtype,
                        device: x.device
                    ),
                    diagonal: 1
                )
                .slice(0, 0, sequenceLength, 1)
                .slice(1, 0, sequenceLength, 1)
                .unsqueeze(0)
                .unsqueeze(0);

            var scores = matmul(query, key.transpose(2, 3)) * _scale;
            var weights = softmax(scores + causalMask, dim: -1);
            var context = matmul(weights, value)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view([batchSize, sequenceLength, _attentionDimensions]);

            return _projection.forward(context);
        }
    }
}
