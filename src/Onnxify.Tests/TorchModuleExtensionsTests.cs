using Onnxify.TorchSharp;
using System.Reflection;
using System.Runtime.CompilerServices;
using static TorchSharp.torch;
using TorchModule = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>;

namespace Onnxify.Tests;

public sealed class TorchModuleExtensionsTests
{
    [Fact]
    public void DeepExport_ForLstmStyleForward_EmitsGraphFromForwardAst()
    {
        using var module = new DeepExportTestModule();
        module.eval();

        var model = module.Export(
            input: OnnxTensorType.Create<long>(["batch_size", "seq_len"]),
            output: OnnxTensorType.Create<float>(["batch_size", 3]),
            options: new OnnxModelCreationOptions
            {
                Opset = 22,
            }
        );

        Assert.Collection(
            model.Graph.Inputs,
            input => Assert.Equal("input", input.Name)
        );

        Assert.Collection(
            model.Graph.Outputs,
            output => Assert.Equal("output", output.Name)
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Gather");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "LSTM");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Gemm");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "ReduceSum");
        Assert.Equal("Identity", model.Graph.Nodes.Last().OpType);
    }

    [Fact]
    public void DeepExport_ForMiniGptStyleForward_EmitsRecursiveTransformerGraph()
    {
        using var module = new DeepExportMiniGptModule();
        module.eval();

        var model = module.Export(
            input: OnnxTensorType.Create<long>(["batch", module.MaxSequenceLength]),
            output: OnnxTensorType.Create<float>(["batch", module.MaxSequenceLength, module.VocabularySize]),
            options: new OnnxModelCreationOptions
            {
                Opset = 22,
            }
        );

        Assert.Contains(model.Graph.Initializers, initializer => initializer.Name.StartsWith("position_ids", StringComparison.Ordinal));
        Assert.Contains(model.Graph.Initializers, initializer => initializer.Name.StartsWith("causal_mask", StringComparison.Ordinal));
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Gather");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Reshape");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Transpose");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "MatMul");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Softmax");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Gemm");
        Assert.Equal("Identity", model.Graph.Nodes.Last().OpType);
    }

    [Fact]
    public void Export_ForGlu_EmitsSplitSigmoidAndMul()
    {
        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2, 4, 6]));
        var module = CreateModule<global::TorchSharp.Modules.GLU>(("dim", "_dim", 1L));

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        Assert.Collection(
            graph.Nodes,
            split =>
            {
                Assert.Equal("Split", split.OpType);
                Assert.Equal(1L, Convert.ToInt64(split.Attributes.Single(x => x.Name == "axis").GetValue()));
                Assert.Equal(2L, Convert.ToInt64(split.Attributes.Single(x => x.Name == "num_outputs").GetValue()));
            },
            sigmoidNode => Assert.Equal("Sigmoid", sigmoidNode.OpType),
            mulNode => Assert.Equal("Mul", mulNode.OpType)
        );
    }

    [Fact]
    public void Export_ForGroupNorm_EmitsGroupNormalizationNode()
    {
        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2, 4, 8, 8]));
        var module = CreateModule<global::TorchSharp.Modules.GroupNorm>(
            ("num_groups", "_num_groups", 2L),
            ("eps", "_eps", 1e-5)
        );

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("GroupNormalization", node.OpType);
        Assert.Equal(2L, Convert.ToInt64(node.Attributes.Single(x => x.Name == "num_groups").GetValue()));
        Assert.Equal(1e-05f, Convert.ToSingle(node.Attributes.Single(x => x.Name == "epsilon").GetValue()));

        Assert.Collection(
            graph.Initializers,
            scale =>
            {
                var tensor = Assert.IsType<OnnxTensor<float>>(scale);
                Assert.Equal([4L], tensor.Shape);
            },
            bias =>
            {
                var tensor = Assert.IsType<OnnxTensor<float>>(bias);
                Assert.Equal([4L], tensor.Shape);
            }
        );
    }

    [Fact]
    public void Export_ForAdditionalPadModules_EmitsExpectedPadModesAndVectors()
    {
        AssertPadExport(
            module: CreateModule<global::TorchSharp.Modules.ReflectionPad3d>(("padding", "_padding", new long[] { 1L, 2L, 3L, 4L, 5L, 6L })),
            inputShape: [1L, 2L, 7L, 8L, 9L],
            expectedMode: "reflect",
            expectedPads: [0L, 0L, 5L, 3L, 1L, 0L, 0L, 6L, 4L, 2L]
        );

        AssertPadExport(
            module: CreateModule<global::TorchSharp.Modules.ReplicationPad1d>(("padding", "_padding", new long[] { 2L })),
            inputShape: [1L, 3L, 9L],
            expectedMode: "edge",
            expectedPads: [0L, 0L, 2L, 0L, 0L, 2L]
        );

        AssertPadExport(
            module: CreateModule<global::TorchSharp.Modules.ReplicationPad2d>(("padding", "_padding", new long[] { 1L, 2L, 3L, 4L })),
            inputShape: [1L, 3L, 7L, 8L],
            expectedMode: "edge",
            expectedPads: [0L, 0L, 3L, 1L, 0L, 0L, 4L, 2L]
        );

        AssertPadExport(
            module: CreateModule<global::TorchSharp.Modules.ReplicationPad3d>(("padding", "_padding", new long[] { 1L, 2L, 3L, 4L, 5L, 6L })),
            inputShape: [1L, 2L, 7L, 8L, 9L],
            expectedMode: "edge",
            expectedPads: [0L, 0L, 5L, 3L, 1L, 0L, 0L, 6L, 4L, 2L]
        );
    }

    [Fact]
    public void Export_ForBilinearUpsample_EmitsLinearResize()
    {
        using var module = nn.Upsample(
            new long[] { 10L, 12L },
            null!,
            global::TorchSharp.torch.UpsampleMode.Bilinear,
            false,
            false
        );
        module.eval();

        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([1L, 3L, 4L, 5L]));

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("Resize", node.OpType);
        Assert.Equal("linear", Assert.IsType<string>(node.Attributes.Single(x => x.Name == "mode").GetValue()));
        Assert.Equal(
            "pytorch_half_pixel",
            Assert.IsType<string>(node.Attributes.Single(x => x.Name == "coordinate_transformation_mode").GetValue())
        );
        Assert.Equal(0L, Convert.ToInt64(node.Attributes.Single(x => x.Name == "antialias").GetValue()));
    }

    [Fact]
    public void Export_ForBicubicUpsample_EmitsCubicResize()
    {
        using var module = nn.Upsample(
            new long[] { 9L, 11L },
            null!,
            global::TorchSharp.torch.UpsampleMode.Bicubic,
            false,
            false
        );
        module.eval();

        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([1L, 2L, 3L, 4L]));

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("Resize", node.OpType);
        Assert.Equal("cubic", Assert.IsType<string>(node.Attributes.Single(x => x.Name == "mode").GetValue()));
        Assert.Equal(-0.75f, Convert.ToSingle(node.Attributes.Single(x => x.Name == "cubic_coeff_a").GetValue()));
    }

    [Fact]
    public void Export_ForGru_EmitsGruNode()
    {
        using var module = nn.GRU(3, 5, 1, true, true, 0.0, false);
        module.eval();

        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 4L, 3L]));

        var output = module.Export(graph, input);

        Assert.NotNull(output.Y);
        Assert.NotNull(output.YH);

        var node = Assert.Single(graph.Nodes, x => x.OpType == "GRU");
        Assert.Equal("forward", Assert.IsType<string>(node.Attributes.Single(x => x.Name == "direction").GetValue()));
        Assert.Equal(5L, Convert.ToInt64(node.Attributes.Single(x => x.Name == "hidden_size").GetValue()));

        Assert.Contains(
            graph.Initializers.OfType<OnnxTensor<float>>(),
            tensor => tensor.Shape.SequenceEqual([1L, 15L, 3L])
        );
        Assert.Contains(
            graph.Initializers.OfType<OnnxTensor<float>>(),
            tensor => tensor.Shape.SequenceEqual([1L, 15L, 5L])
        );
    }

    [Fact]
    public void Export_ForInstanceNorm2d_EmitsInstanceNormalizationNode()
    {
        using var module = nn.InstanceNorm2d(4, 1e-4, 0.1, false, false);
        module.eval();

        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 4L, 6L, 6L]));

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("InstanceNormalization", node.OpType);
        Assert.Equal(1e-04f, Convert.ToSingle(node.Attributes.Single(x => x.Name == "epsilon").GetValue()));

        Assert.Collection(
            graph.Initializers,
            scale =>
            {
                var tensor = Assert.IsType<OnnxTensor<float>>(scale);
                Assert.Equal([4L], tensor.Shape);
                Assert.All(tensor.Value, value => Assert.Equal(1f, value));
            },
            bias =>
            {
                var tensor = Assert.IsType<OnnxTensor<float>>(bias);
                Assert.Equal([4L], tensor.Shape);
                Assert.All(tensor.Value, value => Assert.Equal(0f, value));
            }
        );
    }

    [Fact]
    public void Export_ForUnflatten_EmitsReshapeWithInsertedDimensions()
    {
        using var module = nn.Unflatten(1, new long[] { 2L, 3L });

        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([5L, 6L, 7L]));

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("Reshape", node.OpType);

        var shapeTensor = Assert.Single(graph.Initializers);
        var typedTensor = Assert.IsType<OnnxTensor<long>>(shapeTensor);
        Assert.Equal([0L, 2L, 3L, 0L], typedTensor.Value.ToArray());
    }

    private static void AssertPadExport(
        TorchModule module,
        long[] inputShape,
        string expectedMode,
        long[] expectedPads
    )
    {
        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>(inputShape.Select(static x => (OnnxDimension)x)));

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("Pad", node.OpType);
        Assert.Equal(expectedMode, Assert.IsType<string>(node.Attributes.Single(x => x.Name == "mode").GetValue()));

        var padsTensor = Assert.Single(graph.Initializers);
        var typedTensor = Assert.IsType<OnnxTensor<long>>(padsTensor);
        Assert.Equal(expectedPads, typedTensor.Value.ToArray());
    }

    private static OnnxGraph CreateGraph(int opset)
    {
        return OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = opset,
            ProducerName = "torch-module-tests",
        }).Graph;
    }

    private static TModule CreateModule<TModule>(params (string Name, string BackingName, object Value)[] assignments)
        where TModule : TorchModule
    {
        var module = (TModule)RuntimeHelpers.GetUninitializedObject(typeof(TModule));

        foreach (var (name, backingName, value) in assignments)
        {
            SetMember(module, value, name, backingName);
        }

        return module;
    }

    private static void SetMember(object instance, object value, params string[] candidateNames)
    {
        const BindingFlags FLAGS = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        foreach (var name in candidateNames)
        {
            var property = instance.GetType().GetProperty(name, FLAGS);
            if (property is not null && property.SetMethod is not null)
            {
                property.SetValue(instance, value);
                return;
            }

            var field = instance.GetType().GetField(name, FLAGS);
            if (field is not null)
            {
                field.SetValue(instance, value);
                return;
            }
        }

        throw new InvalidOperationException(
            $"Could not find any writable member named '{string.Join("' or '", candidateNames)}' on '{instance.GetType().FullName}'."
        );
    }

    private sealed class DeepExportTestModule : TorchModule
    {
        private readonly global::TorchSharp.Modules.Embedding _embedding;
        private readonly global::TorchSharp.Modules.LSTM _lstm;
        private readonly global::TorchSharp.Modules.Linear _linear;

        public DeepExportTestModule()
            : base(nameof(DeepExportTestModule))
        {
            _embedding = nn.Embedding(8, 4);
            _lstm = nn.LSTM(4, 5, numLayers: 1, bidirectional: true, batchFirst: true);
            _linear = nn.Linear(10, 3);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = _embedding.forward(input);
            var (lstm, _, _) = _lstm.forward(x);
            var linear = _linear.forward(lstm);
            return sum(linear, 1);
        }
    }

    private sealed class DeepExportMiniGptModule : TorchModule
    {
        private const int VOCABULARY_SIZE = 16;
        private const int MAX_SEQUENCE_LENGTH = 4;
        private const int ATTENTION_HEADS = 2;
        private const int ATTENTION_DIMENSIONS = 8;

        private readonly global::TorchSharp.Modules.Embedding _tokenEmbedding;
        private readonly global::TorchSharp.Modules.Embedding _positionEmbedding;
        private readonly DeepExportTransformerBlock _block;
        private readonly global::TorchSharp.Modules.LayerNorm _outputNorm;

        public DeepExportMiniGptModule()
            : base(nameof(DeepExportMiniGptModule))
        {
            _tokenEmbedding = nn.Embedding(VOCABULARY_SIZE, ATTENTION_DIMENSIONS);
            _positionEmbedding = nn.Embedding(MAX_SEQUENCE_LENGTH, ATTENTION_DIMENSIONS);
            _block = new DeepExportTransformerBlock(ATTENTION_HEADS, ATTENTION_DIMENSIONS, MAX_SEQUENCE_LENGTH);
            _outputNorm = nn.LayerNorm([ATTENTION_DIMENSIONS]);

            RegisterComponents();
        }

        public int VocabularySize => VOCABULARY_SIZE;

        public int MaxSequenceLength => MAX_SEQUENCE_LENGTH;

        public override Tensor forward(Tensor tokens)
        {
            ValidateInputShape(tokens);

            using var positions = CreatePositionIds(tokens.shape[0], tokens.device);

            var x = _tokenEmbedding.forward(tokens) + _positionEmbedding.forward(positions);
            x = _block.forward(x);
            x = _outputNorm.forward(x);
            return ComputeLogits(x);
        }

        private Tensor ComputeLogits(Tensor hiddenStates)
        {
            using var tiedWeight = _tokenEmbedding.weight!.transpose(0, 1);
            return matmul(hiddenStates, tiedWeight);
        }

        private Tensor CreatePositionIds(long batchSize, Device device)
        {
            return arange(MAX_SEQUENCE_LENGTH, dtype: ScalarType.Int64, device: device)
                .unsqueeze(0)
                .expand(batchSize, MAX_SEQUENCE_LENGTH);
        }

        private static void ValidateInputShape(Tensor tokens)
        {
            if (tokens.shape.Length != 2)
            {
                throw new ArgumentException("Expected token ids with rank 2.", nameof(tokens));
            }
        }
    }

    private sealed class DeepExportTransformerBlock : TorchModule
    {
        private readonly global::TorchSharp.Modules.LayerNorm _attentionNorm;
        private readonly DeepExportCausalSelfAttention _attention;
        private readonly global::TorchSharp.Modules.LayerNorm _feedForwardNorm;
        private readonly global::TorchSharp.Modules.Linear _feedForward;

        public DeepExportTransformerBlock(
            int headCount,
            int attentionDimensions,
            int maxContextLength
        ) : base(nameof(DeepExportTransformerBlock))
        {
            _attentionNorm = nn.LayerNorm([attentionDimensions]);
            _attention = new DeepExportCausalSelfAttention(headCount, attentionDimensions, maxContextLength);
            _feedForwardNorm = nn.LayerNorm([attentionDimensions]);
            _feedForward = nn.Linear(attentionDimensions, attentionDimensions);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = x + _attention.forward(_attentionNorm.forward(x));
            x = x + _feedForward.forward(_feedForwardNorm.forward(x));
            return x;
        }
    }

    private sealed class DeepExportCausalSelfAttention : TorchModule
    {
        private readonly int _headCount;
        private readonly int _headDimension;
        private readonly int _attentionDimensions;
        private readonly int _maxContextLength;
        private readonly float _scale;
        private readonly global::TorchSharp.Modules.Linear _attention;
        private readonly global::TorchSharp.Modules.Linear _projection;

        public DeepExportCausalSelfAttention(
            int headCount,
            int attentionDimensions,
            int maxContextLength
        ) : base(nameof(DeepExportCausalSelfAttention))
        {
            _headCount = headCount;
            _headDimension = attentionDimensions / headCount;
            _attentionDimensions = attentionDimensions;
            _maxContextLength = maxContextLength;
            _scale = 1.0f / MathF.Sqrt(_headDimension);
            _attention = nn.Linear(attentionDimensions, attentionDimensions * 3);
            _projection = nn.Linear(attentionDimensions, attentionDimensions);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var batchSize = x.shape[0];
            var sequenceLength = x.shape[1];

            var qkv = _attention.forward(x);
            qkv = qkv
                .view([batchSize, sequenceLength, 3, _headCount, _headDimension])
                .permute(2, 0, 3, 1, 4);

            var query = qkv[0];
            var key = qkv[1];
            var value = qkv[2];

            using var causalMask =
                triu(
                    full([_maxContextLength, _maxContextLength], -10_000f, dtype: x.dtype, device: x.device),
                    diagonal: 1
                )
                .slice(0, 0, sequenceLength, 1)
                .slice(1, 0, sequenceLength, 1)
                .unsqueeze(0)
                .unsqueeze(0);

            var attentionScores = matmul(query, key.transpose(2, 3)) * _scale;
            attentionScores = attentionScores + causalMask;
            var attentionWeights = softmax(attentionScores, dim: -1);
            var attentionOutput = matmul(attentionWeights, value);

            attentionOutput = attentionOutput
                .permute(0, 2, 1, 3)
                .contiguous()
                .view([batchSize, sequenceLength, _attentionDimensions]);

            return _projection.forward(attentionOutput);
        }
    }
}
