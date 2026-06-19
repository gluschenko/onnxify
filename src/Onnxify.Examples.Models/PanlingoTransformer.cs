using Onnxify.TorchSharp;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Onnxify.Examples.Models;

public sealed class PanlingoTransformer : Module<Tensor, Tensor>
{
    private readonly PanlingoTransformerSettings _settings;
    private readonly Device _device;
    private readonly global::TorchSharp.Modules.Embedding _tokenEmbedding;
    private readonly global::TorchSharp.Modules.Embedding _positionEmbedding;
    private readonly ModuleList<PanlingoTransformerBlock> _blocks;
    private readonly global::TorchSharp.Modules.LayerNorm _outputNorm;
    private readonly global::TorchSharp.Modules.Linear _output;
    private readonly bool _tieWeights;
    private Tensor? _positionCache;

    public PanlingoTransformer(
        PanlingoTransformerSettings? settings = null,
        Device? device = null,
        bool tieWeights = true
    ) : base(nameof(PanlingoTransformer))
    {
        _settings = settings ?? PanlingoTransformerSettings.CreateDefault();
        _device = device ?? CPU;
        _tieWeights = tieWeights;

        _tokenEmbedding = Embedding(
            _settings.VocabSize,
            _settings.AttentionDimentions,
            dtype: _settings.ScalarTypeFloat
        );
        _positionEmbedding = Embedding(
            _settings.MaxContextLength,
            _settings.AttentionDimentions,
            dtype: _settings.ScalarTypeFloat
        );

        _blocks = [];
        for (var index = 0; index < _settings.NumLayers; index++)
        {
            _blocks.append(new PanlingoTransformerBlock(_settings, _device, index));
        }

        _outputNorm = LayerNorm(
            _settings.AttentionDimentions,
            eps: _settings.LayerNormEps,
            dtype: _settings.ScalarTypeFloat
        );
        _output = Linear(
            _settings.AttentionDimentions,
            _settings.VocabSize,
            hasBias: false,
            dtype: _settings.ScalarTypeFloat
        );

        RegisterComponents();
        this.to(_device);

        if (_tieWeights && _tokenEmbedding.weight is not null)
        {
            _output.weight = _tokenEmbedding.weight;
        }
    }

    public int VocabularySize => _settings.VocabSize;

    public int MaxContextLength => _settings.MaxContextLength;

    public override Tensor forward(Tensor tokens)
    {
        var batch = tokens.shape[0];
        var seqLen = tokens.shape[1];

        var positions = GetPositions(seqLen, batch);
        var tokenEmbedding = _tokenEmbedding.forward(tokens);
        var positionEmbedding = _positionEmbedding.forward(positions);

        var x = tokenEmbedding + positionEmbedding;
        foreach (var block in _blocks)
        {
            x = block.forward(x);
        }

        x = _outputNorm.forward(x);
        return _output.forward(x);
    }

    private Tensor GetPositions(long seqLen, long batch)
    {
        if (
            _positionCache is not null
            && !_positionCache.IsInvalid
            && _positionCache.shape[1] == seqLen
            && _positionCache.shape[0] == batch
        )
        {
            return _positionCache;
        }

        using var noGrad = torch.no_grad();
        _positionCache = arange(0, _settings.MaxContextLength, device: _device, dtype: ScalarType.Int32)
            .unsqueeze(0)
            .expand(batch, _settings.MaxContextLength)
            .slice(1, 0, seqLen, 1);

        return _positionCache;
    }
}

public sealed class PanlingoTransformerSettings
{
    public required int VocabSize { get; init; }
    public required int MaxContextLength { get; init; }
    public required int NumLayers { get; init; }
    public required int AttentionHeads { get; init; }
    public required int AttentionDimentions { get; init; }
    public required int FeedForwardDimentions { get; init; }
    public required float Dropout { get; init; }
    public required float AttentionDropout { get; init; }
    public required float LayerNormEps { get; init; }
    public required float Temperature { get; init; }
    public required float RepeatPenalty { get; init; }
    public required ScalarType ScalarTypeInt { get; init; }
    public required ScalarType ScalarTypeFloat { get; init; }

    public static PanlingoTransformerSettings CreateDefault()
    {
        return new PanlingoTransformerSettings
        {
            VocabSize = 17,
            MaxContextLength = 4,
            NumLayers = 1,
            AttentionHeads = 2,
            AttentionDimentions = 8,
            FeedForwardDimentions = 16,
            Dropout = 0f,
            AttentionDropout = 0f,
            LayerNormEps = 1e-5f,
            Temperature = 1f,
            RepeatPenalty = 1f,
            ScalarTypeInt = ScalarType.Int64,
            ScalarTypeFloat = ScalarType.Float32,
        };
    }
}

internal sealed class PanlingoTransformerBlock : Module<Tensor, Tensor>
{
    private readonly global::TorchSharp.Modules.LayerNorm _attentionNorm;
    private readonly PanlingoCausalSelfAttention _attention;
    private readonly global::TorchSharp.Modules.LayerNorm _feedForwardNorm;
    private readonly PanlingoFeedForwardLayer _feedForward;

    public PanlingoTransformerBlock(
        PanlingoTransformerSettings settings,
        Device device,
        int index
    ) : base($"{nameof(PanlingoTransformerBlock)}_{index}")
    {
        _attentionNorm = LayerNorm(
            settings.AttentionDimentions,
            eps: settings.LayerNormEps,
            dtype: settings.ScalarTypeFloat
        );
        _attention = new PanlingoCausalSelfAttention(settings, device);
        _feedForwardNorm = LayerNorm(
            settings.AttentionDimentions,
            eps: settings.LayerNormEps,
            dtype: settings.ScalarTypeFloat
        );
        _feedForward = new PanlingoFeedForwardLayer(settings, device);

        RegisterComponents();
        this.to(device);
    }

    public override Tensor forward(Tensor x)
    {
        x = x + _attention.forward(_attentionNorm.forward(x));
        x = x + _feedForward.forward(_feedForwardNorm.forward(x));
        return x;
    }
}

internal sealed class PanlingoCausalSelfAttention : Module<Tensor, Tensor>
{
    private readonly int _nHead;
    private readonly int _headDim;
    private readonly float _scale;
    private readonly global::TorchSharp.Modules.Linear _attention;
    private readonly global::TorchSharp.Modules.Dropout _attentionDropout;
    private readonly global::TorchSharp.Modules.Linear _projection;
    private readonly global::TorchSharp.Modules.Dropout _residualDrop;
    private readonly Device _device;
    private readonly PanlingoTransformerSettings _settings;
    private Tensor? _casualMask;

    public PanlingoCausalSelfAttention(
        PanlingoTransformerSettings settings,
        Device device
    ) : base(nameof(PanlingoCausalSelfAttention))
    {
        _nHead = settings.AttentionHeads;
        _headDim = settings.AttentionDimentions / settings.AttentionHeads;
        _scale = 1.0f / MathF.Sqrt(_headDim);
        _device = device;
        _settings = settings;

        _attention = Linear(
            inputSize: settings.AttentionDimentions,
            outputSize: settings.AttentionDimentions * 3,
            hasBias: true,
            dtype: settings.ScalarTypeFloat
        );
        _attentionDropout = Dropout(settings.AttentionDropout);
        _projection = Linear(
            inputSize: settings.AttentionDimentions,
            outputSize: settings.AttentionDimentions,
            hasBias: true,
            dtype: settings.ScalarTypeFloat
        );
        _residualDrop = Dropout(settings.Dropout);

        RegisterComponents();
        this.to(device);
    }

    public override Tensor forward(Tensor x)
    {
        var (batchSize, seqLen, dModel) = (x.shape[0], x.shape[1], x.shape[2]);
        var qkv = _attention.forward(x);
        var newShape = new long[] { batchSize, seqLen, 3, _nHead, _headDim };
        qkv = qkv.view(newShape).permute(2, 0, 3, 1, 4);

        var q = qkv[0];
        var k = qkv[1];
        var v = qkv[2];

        var kT = k.transpose(-2, -1);
        var attn = q.matmul(kT) * _scale;
        var mask = GetCausalMask((short)seqLen);
        attn = attn + mask;

        attn = functional.softmax(attn, dim: -1);
        attn = _attentionDropout.forward(attn);

        var y = attn.matmul(v);
        y = y.permute(0, 2, 1, 3).contiguous();
        y = y.view(batchSize, seqLen, dModel);

        y = _projection.forward(y);
        return _residualDrop.forward(y);
    }

    private Tensor GetCausalMask(short maxSeqLen)
    {
        if (
            _casualMask is not null
            && !_casualMask.IsInvalid
            && _casualMask.shape[2] == maxSeqLen
            && _casualMask.shape[3] == maxSeqLen
        )
        {
            return _casualMask;
        }

        using var noGrad = torch.no_grad();

        var ones = torch.ones([maxSeqLen, maxSeqLen], device: _device, dtype: _settings.ScalarTypeInt);
        var mask = ones.tril();
        var zero = torch.zeros_like(mask, dtype: _settings.ScalarTypeInt);
        var negInf = torch.full_like(mask, float.NegativeInfinity, dtype: _settings.ScalarTypeFloat);

        mask = torch.where(mask.eq(0), negInf, zero);
        mask = mask.unsqueeze(0).unsqueeze(0);

        _casualMask = mask;
        return _casualMask;
    }
}

internal sealed class PanlingoFeedForwardLayer : Module<Tensor, Tensor>
{
    private readonly global::TorchSharp.Modules.Linear _input;
    private readonly global::TorchSharp.Modules.Linear _hidden;
    private readonly global::TorchSharp.Modules.Dropout _dropout;

    public PanlingoFeedForwardLayer(
        PanlingoTransformerSettings settings,
        Device device
    ) : base(nameof(PanlingoFeedForwardLayer))
    {
        _input = Linear(
            inputSize: settings.AttentionDimentions,
            outputSize: settings.FeedForwardDimentions,
            hasBias: true,
            dtype: settings.ScalarTypeFloat
        );
        _hidden = Linear(
            inputSize: settings.FeedForwardDimentions,
            outputSize: settings.AttentionDimentions,
            hasBias: true,
            dtype: settings.ScalarTypeFloat
        );
        _dropout = Dropout(settings.Dropout);

        RegisterComponents();
        this.to(device);
    }

    public override Tensor forward(Tensor x)
    {
        var h = _input.forward(x);
        h = functional.gelu(h);
        h = _hidden.forward(h);
        return _dropout.forward(h);
    }
}
