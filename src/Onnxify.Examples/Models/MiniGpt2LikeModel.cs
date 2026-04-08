using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Onnxify.Examples.Models;

public sealed class MiniGpt2LikeModel : Module<Tensor, Tensor>
{
    private readonly MiniGpt2Settings _settings;

    private readonly global::TorchSharp.Modules.Embedding _tokenEmbedding;
    private readonly global::TorchSharp.Modules.Embedding _positionEmbedding;
    private readonly MiniGpt2TransformerBlock _block1;
    private readonly MiniGpt2TransformerBlock _block2;
    private readonly global::TorchSharp.Modules.LayerNorm _outputNorm;

    public MiniGpt2LikeModel(
        string name = "mini_gpt2_like",
        MiniGpt2Settings? settings = null
    ) : base(name)
    {
        _settings = settings ?? MiniGpt2Settings.CreateDefault();

        _tokenEmbedding = Embedding(_settings.VocabSize, _settings.AttentionDimensions);
        _positionEmbedding = Embedding(_settings.MaxContextLength, _settings.AttentionDimensions);

        _block1 = new MiniGpt2TransformerBlock("transformer_block_1", _settings);
        _block2 = new MiniGpt2TransformerBlock("transformer_block_2", _settings);
        _outputNorm = LayerNorm([_settings.AttentionDimensions], eps: _settings.LayerNormEps);

        RegisterComponents();
    }

    public int VocabularySize => _settings.VocabSize;

    public int MaxSequenceLength => _settings.MaxContextLength;

    public override Tensor forward(Tensor tokens)
    {
        ValidateInputShape(tokens);

        using var positions = CreatePositionIds(tokens.shape[0], tokens.device);

        var x = _tokenEmbedding.forward(tokens) + _positionEmbedding.forward(positions);
        x = _block1.forward(x);
        x = _block2.forward(x);
        x = _outputNorm.forward(x);
        return ComputeLogits(x);
    }

    public OnnxModel Export()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 22,
        });

        var graph = model.Graph;
        var input = graph.AddInput(
            name: "input",
            type: OnnxTensorType.Create<long>(["batch", _settings.MaxContextLength])
        );

        var positionIds = graph.AddTensor(
            name: "position_ids",
            shape: [1, _settings.MaxContextLength],
            value: Enumerable.Range(0, _settings.MaxContextLength).Select(static x => (long)x).ToArray()
        );

        var causalMask = graph.AddTensor(
            name: "causal_mask",
            shape: [1, 1, _settings.MaxContextLength, _settings.MaxContextLength],
            value: CreateAdditiveCausalMask(_settings.MaxContextLength)
        );

        var x = _tokenEmbedding.Export(graph, input);
        var positions = _positionEmbedding.Export(graph, positionIds);

        x = graph.Add(
            name: "token_plus_position",
            options: new AddInputOptions
            {
                A = x,
                B = positions,
            }
        );

        x = _block1.Export(graph, x, causalMask);
        x = _block2.Export(graph, x, causalMask);
        x = _outputNorm.Export(graph, x);
        x = MiniGpt2OnnxExport.AddTiedOutputProjection(graph, "lm_head", x, _tokenEmbedding);

        var outputEdge = graph.AddEdge("output");
        graph.Identity(
            name: "output_identity",
            options: new IdentityInputOutputOptions
            {
                Input = x,
                Output = outputEdge,
            }
        );

        graph.AddOutput(
            name: "output",
            type: OnnxTensorType.Create<float>(["batch", _settings.MaxContextLength, _settings.VocabSize])
        );

        model.AddMetadataProps("architecture", "mini-gpt2-like");
        model.AddMetadataProps("attention", "decoder_only_causal_self_attention");
        model.AddMetadataProps("attention_projection", "fused_qkv");
        model.AddMetadataProps("sequence_length", _settings.MaxContextLength.ToString());
        model.AddMetadataProps("vocabulary_size", _settings.VocabSize.ToString());

        return model;
    }

    private Tensor ComputeLogits(Tensor hiddenStates)
    {
        var weight = _tokenEmbedding.weight
            ?? throw new InvalidOperationException("Token embedding weights are not initialized.");

        using var tiedWeight = weight.transpose(0, 1);
        return torch.matmul(hiddenStates, tiedWeight);
    }

    private Tensor CreatePositionIds(long batchSize, Device device)
    {
        return torch.arange(_settings.MaxContextLength, dtype: ScalarType.Int64, device: device)
            .unsqueeze(0)
            .expand(batchSize, _settings.MaxContextLength);
    }

    private void ValidateInputShape(Tensor tokens)
    {
        if (tokens.shape.Length != 2)
        {
            throw new ArgumentException(
                $"Expected token ids with rank 2 [batch, sequence]. Got rank {tokens.shape.Length}.",
                nameof(tokens)
            );
        }

        if (tokens.shape[1] != _settings.MaxContextLength)
        {
            throw new ArgumentException(
                $"Expected sequence length {_settings.MaxContextLength}, but received {tokens.shape[1]}.",
                nameof(tokens)
            );
        }
    }

    private static float[] CreateAdditiveCausalMask(int sequenceLength)
    {
        var mask = new float[sequenceLength * sequenceLength];
        for (var row = 0; row < sequenceLength; row++)
        {
            for (var col = 0; col < sequenceLength; col++)
            {
                mask[(row * sequenceLength) + col] = col <= row ? 0f : -10_000f;
            }
        }

        return mask;
    }
}

public sealed class MiniGpt2Settings
{
    public required int VocabSize { get; init; }
    public required int MaxContextLength { get; init; }
    public required int AttentionHeads { get; init; }
    public required int AttentionDimensions { get; init; }
    public required int FeedForwardDimensions { get; init; }
    public required float LayerNormEps { get; init; }

    public static MiniGpt2Settings CreateDefault()
    {
        return new MiniGpt2Settings
        {
            VocabSize = 64,
            MaxContextLength = 8,
            AttentionHeads = 4,
            AttentionDimensions = 32,
            FeedForwardDimensions = 128,
            LayerNormEps = 1e-5f,
        };
    }
}

internal sealed class MiniGpt2TransformerBlock : Module<Tensor, Tensor>
{
    private readonly global::TorchSharp.Modules.LayerNorm _attentionNorm;
    private readonly MiniGpt2CausalSelfAttention _attention;
    private readonly global::TorchSharp.Modules.LayerNorm _feedForwardNorm;
    private readonly MiniGpt2FeedForward _feedForward;

    public MiniGpt2TransformerBlock(
        string name,
        MiniGpt2Settings settings
    ) : base(name)
    {
        _attentionNorm = LayerNorm([settings.AttentionDimensions], eps: settings.LayerNormEps);
        _attention = new MiniGpt2CausalSelfAttention($"{name}_attention", settings);
        _feedForwardNorm = LayerNorm([settings.AttentionDimensions], eps: settings.LayerNormEps);
        _feedForward = new MiniGpt2FeedForward($"{name}_mlp", settings);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = x + _attention.forward(_attentionNorm.forward(x));
        x = x + _feedForward.forward(_feedForwardNorm.forward(x));
        return x;
    }

    public IOnnxGraphEdge Export(
        OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge causalMask
    )
    {
        var attentionNorm = _attentionNorm.Export(graph, input);
        var attention = _attention.Export(graph, attentionNorm, causalMask);

        var residual = graph.Add(
            name: graph.NextName("attn_residual"),
            options: new AddInputOptions
            {
                A = input,
                B = attention,
            }
        );

        var feedForwardNorm = _feedForwardNorm.Export(graph, residual);
        var mlp = _feedForward.Export(graph, feedForwardNorm);

        return graph.Add(
            name: graph.NextName("mlp_residual"),
            options: new AddInputOptions
            {
                A = residual,
                B = mlp,
            }
        );
    }
}

internal sealed class MiniGpt2CausalSelfAttention : Module<Tensor, Tensor>
{
    private readonly int _headCount;
    private readonly int _headDimension;
    private readonly int _attentionDimensions;
    private readonly int _maxContextLength;
    private readonly float _scale;

    private readonly global::TorchSharp.Modules.Linear _attention;
    private readonly global::TorchSharp.Modules.Linear _projection;

    public MiniGpt2CausalSelfAttention(
        string name,
        MiniGpt2Settings settings
    ) : base(name)
    {
        if (settings.AttentionDimensions % settings.AttentionHeads != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(settings),
                $"Attention dimensions {settings.AttentionDimensions} must be divisible by head count {settings.AttentionHeads}."
            );
        }

        _headCount = settings.AttentionHeads;
        _headDimension = settings.AttentionDimensions / settings.AttentionHeads;
        _attentionDimensions = settings.AttentionDimensions;
        _maxContextLength = settings.MaxContextLength;
        _scale = 1.0f / MathF.Sqrt(_headDimension);

        _attention = Linear(settings.AttentionDimensions, settings.AttentionDimensions * 3);
        _projection = Linear(settings.AttentionDimensions, settings.AttentionDimensions);

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

        using var causalMask = torch.triu(
                torch.full(
                    [_maxContextLength, _maxContextLength],
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

        var attentionScores = torch.matmul(query, key.transpose(2, 3)) * _scale;
        attentionScores = attentionScores + causalMask;

        var attentionWeights = torch.softmax(attentionScores, dim: -1);
        var attentionOutput = torch.matmul(attentionWeights, value);

        attentionOutput = attentionOutput
            .permute(0, 2, 1, 3)
            .contiguous()
            .view([batchSize, sequenceLength, _attentionDimensions]);

        return _projection.forward(attentionOutput);
    }

    public IOnnxGraphEdge Export(
        OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge causalMask
    )
    {
        var query = MiniGpt2OnnxExport.AddLinearRows(
            graph,
            "attn_query",
            input,
            _attention,
            rowStart: 0,
            rowCount: _attentionDimensions
        );
        query = MiniGpt2OnnxExport.ReshapeToHeads(graph, "attn_query_heads", query, _headCount, _headDimension);

        var key = MiniGpt2OnnxExport.AddLinearRows(
            graph,
            "attn_key",
            input,
            _attention,
            rowStart: _attentionDimensions,
            rowCount: _attentionDimensions
        );
        key = MiniGpt2OnnxExport.ReshapeToHeads(graph, "attn_key_heads", key, _headCount, _headDimension);

        var value = MiniGpt2OnnxExport.AddLinearRows(
            graph,
            "attn_value",
            input,
            _attention,
            rowStart: _attentionDimensions * 2,
            rowCount: _attentionDimensions
        );
        value = MiniGpt2OnnxExport.ReshapeToHeads(graph, "attn_value_heads", value, _headCount, _headDimension);

        var keyTransposed = graph.Transpose(
            name: graph.NextName("attn_key_transpose"),
            options: new TransposeInputOptions
            {
                Data = key,
                Perm = [0, 1, 3, 2],
            }
        );

        var attentionScores = graph.MatMul(
            name: graph.NextName("attn_scores"),
            options: new MatMulInputOptions
            {
                A = query,
                B = keyTransposed,
            }
        );

        var scale = graph.AddTensor(
            name: graph.NextName("attn_scale"),
            shape: [1],
            value: [_scale]
        );

        attentionScores = graph.Mul(
            name: graph.NextName("attn_scores_scaled"),
            options: new MulInputOptions
            {
                A = attentionScores,
                B = scale,
            }
        );

        attentionScores = graph.Add(
            name: graph.NextName("attn_scores_masked"),
            options: new AddInputOptions
            {
                A = attentionScores,
                B = causalMask,
            }
        );

        var attentionWeights = graph.Softmax(
            name: graph.NextName("attn_weights"),
            options: new SoftmaxInputOptions
            {
                Input = attentionScores,
                Axis = -1,
            }
        );

        var attentionOutput = graph.MatMul(
            name: graph.NextName("attn_output"),
            options: new MatMulInputOptions
            {
                A = attentionWeights,
                B = value,
            }
        );

        attentionOutput = MiniGpt2OnnxExport.MergeHeads(
            graph,
            "attn_merge_heads",
            attentionOutput,
            _attentionDimensions
        );

        return MiniGpt2OnnxExport.AddLinear(graph, "attn_projection", attentionOutput, _projection);
    }
}

internal sealed class MiniGpt2FeedForward : Module<Tensor, Tensor>
{
    private readonly global::TorchSharp.Modules.Linear _input;
    private readonly global::TorchSharp.Modules.GELU _gelu;
    private readonly global::TorchSharp.Modules.Linear _output;

    public MiniGpt2FeedForward(
        string name,
        MiniGpt2Settings settings
    ) : base(name)
    {
        _input = Linear(settings.AttentionDimensions, settings.FeedForwardDimensions);
        _gelu = GELU();
        _output = Linear(settings.FeedForwardDimensions, settings.AttentionDimensions);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = _input.forward(x);
        x = _gelu.forward(x);
        x = _output.forward(x);
        return x;
    }

    public IOnnxGraphEdge Export(OnnxGraph graph, IOnnxGraphEdge input)
    {
        var x = MiniGpt2OnnxExport.AddLinear(graph, "mlp_input", input, _input);
        x = _gelu.Export(graph, x);
        return MiniGpt2OnnxExport.AddLinear(graph, "mlp_output", x, _output);
    }
}

internal static class MiniGpt2OnnxExport
{
    public static IOnnxGraphEdge AddLinear(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        global::TorchSharp.Modules.Linear linear
    )
    {
        return AddLinearCore(
            graph,
            prefix,
            input,
            GetFloatData(linear.weight),
            rows: checked((int)linear.weight.shape[0]),
            columns: checked((int)linear.weight.shape[1]),
            linear.bias is null ? null : GetFloatData(linear.bias)
        );
    }

    public static IOnnxGraphEdge AddLinearRows(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        global::TorchSharp.Modules.Linear linear,
        int rowStart,
        int rowCount
    )
    {
        var totalRows = checked((int)linear.weight.shape[0]);
        var totalColumns = checked((int)linear.weight.shape[1]);

        var weight = SliceRows(
            GetFloatData(linear.weight),
            totalRows,
            totalColumns,
            rowStart,
            rowCount
        );

        float[]? bias = null;
        if (linear.bias is not null)
        {
            var sourceBias = GetFloatData(linear.bias);
            bias = new float[rowCount];
            Array.Copy(sourceBias, rowStart, bias, 0, rowCount);
        }

        return AddLinearCore(graph, prefix, input, weight, rowCount, totalColumns, bias);
    }

    public static IOnnxGraphEdge AddTiedOutputProjection(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        global::TorchSharp.Modules.Embedding embedding
    )
    {
        var weight = embedding.weight
            ?? throw new InvalidOperationException("Token embedding weights are not initialized.");

        var weightShape = weight.shape.ToArray();
        var transposedWeight = Transpose2D(
            GetFloatData(weight),
            rows: checked((int)weightShape[0]),
            columns: checked((int)weightShape[1])
        );

        var initializer = graph.AddTensor(
            name: $"{graph.NextName(prefix)}_w",
            shape: [weightShape[1], weightShape[0]],
            value: transposedWeight
        );

        return graph.MatMul(
            name: graph.NextName(prefix),
            options: new MatMulInputOptions
            {
                A = input,
                B = initializer,
            }
        );
    }

    public static IOnnxGraphEdge ReshapeToHeads(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        int headCount,
        int headDimension
    )
    {
        var name = graph.NextName(prefix);
        var shape = graph.AddTensor(
            name: $"{name}_shape",
            shape: [4],
            value: [0L, 0L, headCount, headDimension]
        );

        var reshaped = graph.Reshape(
            name: name,
            options: new ReshapeInputOptions
            {
                Data = input,
                Shape = shape,
            }
        );

        return graph.Transpose(
            name: $"{name}_transpose",
            options: new TransposeInputOptions
            {
                Data = reshaped,
                Perm = [0, 2, 1, 3],
            }
        );
    }

    public static IOnnxGraphEdge MergeHeads(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        int hiddenSize
    )
    {
        var name = graph.NextName(prefix);
        var transposed = graph.Transpose(
            name: $"{name}_transpose",
            options: new TransposeInputOptions
            {
                Data = input,
                Perm = [0, 2, 1, 3],
            }
        );

        var shape = graph.AddTensor(
            name: $"{name}_shape",
            shape: [3],
            value: [0L, 0L, hiddenSize]
        );

        return graph.Reshape(
            name: name,
            options: new ReshapeInputOptions
            {
                Data = transposed,
                Shape = shape,
            }
        );
    }

    private static IOnnxGraphEdge AddLinearCore(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        float[] weight,
        int rows,
        int columns,
        float[]? bias
    )
    {
        var name = graph.NextName(prefix);
        var transposedWeight = Transpose2D(weight, rows, columns);

        var weightTensor = graph.AddTensor(
            name: $"{name}_w",
            shape: [columns, rows],
            value: transposedWeight
        );

        var output = graph.MatMul(
            name: name,
            options: new MatMulInputOptions
            {
                A = input,
                B = weightTensor,
            }
        );

        if (bias is null)
        {
            return output;
        }

        var biasTensor = graph.AddTensor(
            name: $"{name}_b",
            shape: [rows],
            value: bias
        );

        return graph.Add(
            name: $"{name}_bias",
            options: new AddInputOptions
            {
                A = output,
                B = biasTensor,
            }
        );
    }

    private static float[] GetFloatData(Tensor tensor)
    {
        return tensor.detach().cpu().data<float>().ToArray();
    }

    private static float[] SliceRows(
        float[] input,
        int totalRows,
        int totalColumns,
        int rowStart,
        int rowCount
    )
    {
        if (rowStart < 0 || rowCount < 0 || rowStart + rowCount > totalRows)
        {
            throw new ArgumentOutOfRangeException(nameof(rowStart));
        }

        var output = new float[rowCount * totalColumns];
        for (var row = 0; row < rowCount; row++)
        {
            Array.Copy(
                input,
                (rowStart + row) * totalColumns,
                output,
                row * totalColumns,
                totalColumns
            );
        }

        return output;
    }

    private static float[] Transpose2D(float[] input, int rows, int columns)
    {
        var output = new float[input.Length];
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                output[(column * rows) + row] = input[(row * columns) + column];
            }
        }

        return output;
    }
}
