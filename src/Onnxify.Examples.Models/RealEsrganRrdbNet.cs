using Onnxify.TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchModules = TorchSharp.Modules;

namespace Onnxify.Examples.Models;

/// <summary>
/// TorchSharp port of the RRDB generator used by Real-ESRGAN models such as RealESRGAN_x4plus.
/// </summary>
public sealed class RealEsrganRrdbNet : Module<Tensor, Tensor>
{
    private const float ResidualScale = 0.2f;

    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _featureChannels;
    private readonly int _blockCount;
    private readonly int _growthChannels;
    private readonly int _scale;

    private readonly TorchModules.PixelUnshuffle? _pixelUnshuffle;
    private readonly TorchModules.Conv2d _convFirst;
    private readonly RealEsrganRrdb[] _rrdbBlocks;
    private readonly Module<Tensor, Tensor> _body;
    private readonly TorchModules.Conv2d _trunkConv;
    private readonly TorchModules.Conv2d _upConv1;
    private readonly TorchModules.Conv2d _upConv2;
    private readonly TorchModules.Conv2d _hrConv;
    private readonly TorchModules.Conv2d _convLast;
    private readonly TorchModules.LeakyReLU _activation;

    public RealEsrganRrdbNet(
        string name = "realesrgan_rrdbnet",
        int inputChannels = 3,
        int outputChannels = 3,
        int featureChannels = 64,
        int blockCount = 23,
        int growthChannels = 32,
        int scale = 4
    ) : base(name)
    {
        ValidateConfiguration(inputChannels, outputChannels, featureChannels, blockCount, growthChannels, scale);

        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _featureChannels = featureChannels;
        _blockCount = blockCount;
        _growthChannels = growthChannels;
        _scale = scale;

        var firstConvInputChannels = scale switch
        {
            1 => inputChannels * 16,
            2 => inputChannels * 4,
            _ => inputChannels,
        };

        _pixelUnshuffle = scale switch
        {
            1 => PixelUnshuffle(4),
            2 => PixelUnshuffle(2),
            _ => null,
        };

        _convFirst = Conv2d(firstConvInputChannels, featureChannels, kernel_size: 3, padding: 1);
        _rrdbBlocks = Enumerable.Range(0, blockCount)
            .Select(index => new RealEsrganRrdb($"rrdb_{index}", featureChannels, growthChannels))
            .ToArray();
        _body = Sequential(
            _rrdbBlocks
                .Select((block, index) => ($"rrdb_{index}", (Module<Tensor, Tensor>)block))
                .ToArray()
        );
        _trunkConv = Conv2d(featureChannels, featureChannels, kernel_size: 3, padding: 1);
        _upConv1 = Conv2d(featureChannels, featureChannels, kernel_size: 3, padding: 1);
        _upConv2 = Conv2d(featureChannels, featureChannels, kernel_size: 3, padding: 1);
        _hrConv = Conv2d(featureChannels, featureChannels, kernel_size: 3, padding: 1);
        _convLast = Conv2d(featureChannels, outputChannels, kernel_size: 3, padding: 1);
        _activation = LeakyReLU(0.2, inplace: true);

        RegisterComponents();
    }

    public int InputChannels => _inputChannels;

    public int OutputChannels => _outputChannels;

    public int FeatureChannels => _featureChannels;

    public int BlockCount => _blockCount;

    public int GrowthChannels => _growthChannels;

    public int Scale => _scale;

    public static RealEsrganRrdbNet CreateX4Plus(string name = "realesrgan_x4plus")
    {
        return new RealEsrganRrdbNet(name, scale: 4, blockCount: 23);
    }

    public static RealEsrganRrdbNet CreateX4PlusAnime6B(string name = "realesrgan_x4plus_anime_6b")
    {
        return new RealEsrganRrdbNet(name, scale: 4, blockCount: 6);
    }

    public static RealEsrganRrdbNet CreateX2Plus(string name = "realesrgan_x2plus")
    {
        return new RealEsrganRrdbNet(name, scale: 2, blockCount: 23);
    }

    public override Tensor forward(Tensor input)
    {
        var x = _pixelUnshuffle is null
            ? input
            : _pixelUnshuffle.forward(input);

        var features = _convFirst.forward(x);
        var body = _body.forward(features);
        body = _trunkConv.forward(body);
        features = features + body;

        features = UpsampleNearest2D(features);
        features = _activation.forward(_upConv1.forward(features));
        features = UpsampleNearest2D(features);
        features = _activation.forward(_upConv2.forward(features));
        features = _activation.forward(_hrConv.forward(features));

        return _convLast.forward(features);
    }

    public OnnxModel Export(long inputHeight = 64, long inputWidth = 64)
    {
        if (inputHeight <= 0 || inputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(inputHeight),
                "Real-ESRGAN export requires positive static input height and width."
            );
        }

        var pixelUnshuffleFactor = _scale switch
        {
            1 => 4,
            2 => 2,
            _ => 1,
        };
        if (inputHeight % pixelUnshuffleFactor != 0 || inputWidth % pixelUnshuffleFactor != 0)
        {
            throw new ArgumentException(
                $"Real-ESRGAN scale {_scale} export requires input height and width divisible by {pixelUnshuffleFactor}."
            );
        }

        eval();

        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 22,
        });
        var graph = model.Graph;

        var input = graph.AddInput(
            name: "input",
            type: OnnxTensorType.Create<float>(["batch", _inputChannels, inputHeight, inputWidth])
        );

        var x = _pixelUnshuffle is null
            ? input
            : _pixelUnshuffle.Export(graph, input);

        var features = _convFirst.Export(graph, x);
        var body = features;
        foreach (var block in _rrdbBlocks)
        {
            body = block.Export(graph, body);
        }

        body = _trunkConv.Export(graph, body);
        features = graph.ExportAdd(features, body);

        features = ExportNearestUpsample2D(graph, features);
        features = _activation.Export(graph, _upConv1.Export(graph, features));
        features = ExportNearestUpsample2D(graph, features);
        features = _activation.Export(graph, _upConv2.Export(graph, features));
        features = _activation.Export(graph, _hrConv.Export(graph, features));
        features = _convLast.Export(graph, features);

        var outputEdge = graph.AddEdge("output");
        graph.Identity(
            name: "output_identity",
            options: new IdentityInputOutputOptions
            {
                Input = features,
                Output = outputEdge,
            }
        );

        graph.AddOutput(
            name: "output",
            type: OnnxTensorType.Create<float>(["batch", _outputChannels, inputHeight * _scale, inputWidth * _scale])
        );

        model.AddMetadataProps("architecture", "real-esrgan-rrdbnet");
        model.AddMetadataProps("scale", _scale.ToString());
        model.AddMetadataProps("features", _featureChannels.ToString());
        model.AddMetadataProps("rrdb_blocks", _blockCount.ToString());
        model.AddMetadataProps("growth_channels", _growthChannels.ToString());

        return model;
    }

    private static void ValidateConfiguration(
        int inputChannels,
        int outputChannels,
        int featureChannels,
        int blockCount,
        int growthChannels,
        int scale
    )
    {
        if (inputChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputChannels));
        }

        if (outputChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputChannels));
        }

        if (featureChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(featureChannels));
        }

        if (blockCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blockCount));
        }

        if (growthChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(growthChannels));
        }

        if (scale is not 1 and not 2 and not 4)
        {
            throw new ArgumentOutOfRangeException(
                nameof(scale),
                scale,
                "Real-ESRGAN RRDBNet supports scale 1, 2, or 4."
            );
        }
    }

    private sealed class RealEsrganRrdb : Module<Tensor, Tensor>
    {
        private readonly RealEsrganResidualDenseBlock _block1;
        private readonly RealEsrganResidualDenseBlock _block2;
        private readonly RealEsrganResidualDenseBlock _block3;

        public RealEsrganRrdb(string name, int featureChannels, int growthChannels)
            : base(name)
        {
            _block1 = new RealEsrganResidualDenseBlock("rdb1", featureChannels, growthChannels);
            _block2 = new RealEsrganResidualDenseBlock("rdb2", featureChannels, growthChannels);
            _block3 = new RealEsrganResidualDenseBlock("rdb3", featureChannels, growthChannels);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = _block1.forward(input);
            x = _block2.forward(x);
            x = _block3.forward(x);

            return (x * ResidualScale) + input;
        }

        public IOnnxGraphEdge Export(OnnxGraph graph, IOnnxGraphEdge input)
        {
            var x = _block1.Export(graph, input);
            x = _block2.Export(graph, x);
            x = _block3.Export(graph, x);

            return graph.ExportAdd(ScaleEdge(graph, x, ResidualScale), input);
        }
    }

    private sealed class RealEsrganResidualDenseBlock : Module<Tensor, Tensor>
    {
        private readonly TorchModules.Conv2d _conv1;
        private readonly TorchModules.Conv2d _conv2;
        private readonly TorchModules.Conv2d _conv3;
        private readonly TorchModules.Conv2d _conv4;
        private readonly TorchModules.Conv2d _conv5;
        private readonly TorchModules.LeakyReLU _activation;

        public RealEsrganResidualDenseBlock(string name, int featureChannels, int growthChannels)
            : base(name)
        {
            _conv1 = Conv2d(featureChannels, growthChannels, kernel_size: 3, padding: 1);
            _conv2 = Conv2d(featureChannels + growthChannels, growthChannels, kernel_size: 3, padding: 1);
            _conv3 = Conv2d(featureChannels + (growthChannels * 2), growthChannels, kernel_size: 3, padding: 1);
            _conv4 = Conv2d(featureChannels + (growthChannels * 3), growthChannels, kernel_size: 3, padding: 1);
            _conv5 = Conv2d(featureChannels + (growthChannels * 4), featureChannels, kernel_size: 3, padding: 1);
            _activation = LeakyReLU(0.2, inplace: true);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x1 = _activation.forward(_conv1.forward(input));
            var x2 = _activation.forward(_conv2.forward(cat(new[] { input, x1 }, 1)));
            var x3 = _activation.forward(_conv3.forward(cat(new[] { input, x1, x2 }, 1)));
            var x4 = _activation.forward(_conv4.forward(cat(new[] { input, x1, x2, x3 }, 1)));
            var x5 = _conv5.forward(cat(new[] { input, x1, x2, x3, x4 }, 1));

            return (x5 * ResidualScale) + input;
        }

        public IOnnxGraphEdge Export(OnnxGraph graph, IOnnxGraphEdge input)
        {
            var x1 = _activation.Export(graph, _conv1.Export(graph, input));
            var x2 = _activation.Export(graph, _conv2.Export(graph, graph.ExportConcat(new[] { input, x1 }, dim: 1)));
            var x3 = _activation.Export(graph, _conv3.Export(graph, graph.ExportConcat(new[] { input, x1, x2 }, dim: 1)));
            var x4 = _activation.Export(graph, _conv4.Export(graph, graph.ExportConcat(new[] { input, x1, x2, x3 }, dim: 1)));
            var x5 = _conv5.Export(graph, graph.ExportConcat(new[] { input, x1, x2, x3, x4 }, dim: 1));

            return graph.ExportAdd(ScaleEdge(graph, x5, ResidualScale), input);
        }
    }

    private static IOnnxGraphEdge ScaleEdge(OnnxGraph graph, IOnnxGraphEdge input, float scale)
    {
        var scalar = graph.AddTensor<float>(
            name: graph.NextName("realesrgan_residual_scale"),
            shape: [],
            value: [scale]
        );

        return graph.ExportMul(input, scalar);
    }

    private static Tensor UpsampleNearest2D(Tensor input)
    {
        return global::TorchSharp.torch.nn.functional.interpolate(
            input,
            Array.Empty<long>(),
            [2.0, 2.0],
            global::TorchSharp.torch.InterpolationMode.Nearest,
            align_corners: null,
            recompute_scale_factor: false,
            antialias: false
        );
    }

    private static IOnnxGraphEdge ExportNearestUpsample2D(OnnxGraph graph, IOnnxGraphEdge input)
    {
        var name = graph.NextName("realesrgan_nearest_upsample");
        var scales = graph.AddTensor<float>(
            name: $"{name}_scales",
            shape: [2],
            value: [2f, 2f]
        );

        return graph.Resize(
            name: name,
            options: new ResizeInputOptions
            {
                X = input,
                Scales = scales,
                Axes = [2, 3],
                Antialias = 0,
                CoordinateTransformationMode = "asymmetric",
                CubicCoeffA = -0.75f,
                Mode = "nearest",
                NearestMode = "floor",
            }
        );
    }
}
