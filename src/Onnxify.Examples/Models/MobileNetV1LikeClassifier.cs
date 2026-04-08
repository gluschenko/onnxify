using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Onnxify.Examples.Models;

public sealed class MobileNetV1LikeClassifier : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _features;
    private readonly Module<Tensor, Tensor> _classifier;
    private readonly int _numClasses;

    public MobileNetV1LikeClassifier(
        string name = "mobilenet_v1_like",
        int numClasses = 10
    ) : base(name)
    {
        _numClasses = numClasses;

        _features = Sequential(
            ("stem", ConvBnReLU6("stem", 3, 32, stride: 2)),
            ("block1", DepthwiseSeparableBlock("block1", 32, 64, stride: 1)),
            ("block2", DepthwiseSeparableBlock("block2", 64, 128, stride: 2)),
            ("block3", DepthwiseSeparableBlock("block3", 128, 128, stride: 1)),
            ("block4", DepthwiseSeparableBlock("block4", 128, 256, stride: 2)),
            ("block5", DepthwiseSeparableBlock("block5", 256, 256, stride: 1)),
            ("block6", DepthwiseSeparableBlock("block6", 256, 512, stride: 2)),
            ("block7", DepthwiseSeparableBlock("block7", 512, 512, stride: 1)),
            ("block8", DepthwiseSeparableBlock("block8", 512, 512, stride: 1)),
            ("block9", DepthwiseSeparableBlock("block9", 512, 512, stride: 1)),
            ("block10", DepthwiseSeparableBlock("block10", 512, 512, stride: 1)),
            ("block11", DepthwiseSeparableBlock("block11", 512, 512, stride: 1)),
            ("block12", DepthwiseSeparableBlock("block12", 512, 1024, stride: 2)),
            ("block13", DepthwiseSeparableBlock("block13", 1024, 1024, stride: 1)),
            ("pool", AdaptiveAvgPool2d(1)),
            ("flatten", Flatten())
        );

        _classifier = Sequential(
            ("dropout", Dropout(0.2)),
            ("fc", Linear(1024, _numClasses))
        );

        RegisterComponents();
    }

    public int NumClasses => _numClasses;

    public override Tensor forward(Tensor input)
    {
        var x = _features.forward(input);
        x = _classifier.forward(x);
        return x;
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
            type: OnnxTensorType.Create<float>(["batch", 3, 96, 96])
        );

        var x = _features.Export(graph, input);
        x = _classifier.Export(graph, x);

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
            type: OnnxTensorType.Create<float>(["batch", _numClasses])
        );

        model.AddMetadataProps("architecture", "mobilenet-v1-like");
        model.AddMetadataProps("input_size", "3x96x96");
        model.AddMetadataProps("classifier_head", "adaptive_avg_pool -> flatten -> dropout -> linear");
        model.AddMetadataProps("backbone", "depthwise-separable-conv");

        return model;
    }

    private static Module<Tensor, Tensor> DepthwiseSeparableBlock(
        string prefix,
        long inputChannels,
        long outputChannels,
        long stride
    )
    {
        return Sequential(
            ($"{prefix}_depthwise", ConvBnReLU6(
                prefix: $"{prefix}_depthwise",
                inputChannels: inputChannels,
                outputChannels: inputChannels,
                stride: stride,
                groups: inputChannels
            )),
            ($"{prefix}_pointwise", ConvBnReLU6(
                prefix: $"{prefix}_pointwise",
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                stride: 1,
                kernelSize: 1,
                padding: 0
            ))
        );
    }

    private static Module<Tensor, Tensor> ConvBnReLU6(
        string prefix,
        long inputChannels,
        long outputChannels,
        long stride,
        long kernelSize = 3,
        long padding = 1,
        long groups = 1
    )
    {
        return Sequential(
            ($"{prefix}_conv", Conv2d(
                inputChannels,
                outputChannels,
                kernel_size: kernelSize,
                stride: stride,
                padding: padding,
                groups: groups
            )),
            ($"{prefix}_bn", BatchNorm2d(outputChannels)),
            ($"{prefix}_relu6", ReLU6())
        );
    }
}
