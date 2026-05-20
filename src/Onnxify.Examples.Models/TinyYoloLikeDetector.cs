using Onnxify.TorchSharp;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Onnxify.Examples.Models;

public sealed class TinyYoloLikeDetector : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _backbone;
    private readonly Module<Tensor, Tensor> _head;

    private readonly int _anchorCount;
    private readonly int _classCount;
    private readonly int _gridHeight;
    private readonly int _gridWidth;
    private readonly int _attributeCount;

    public TinyYoloLikeDetector(
        string name = "tiny_yolo_like",
        int anchorCount = 3,
        int classCount = 4,
        int gridHeight = 8,
        int gridWidth = 8
    ) : base(name)
    {
        _anchorCount = anchorCount;
        _classCount = classCount;
        _gridHeight = gridHeight;
        _gridWidth = gridWidth;
        _attributeCount = classCount + 5;

        _backbone = Sequential(
            ("conv1", Conv2d(3, 16, kernel_size: 3, padding: 1)),
            ("act1", SiLU()),
            ("pool1", MaxPool2d(kernel_size: 2, stride: 2)),

            ("conv2", Conv2d(16, 32, kernel_size: 3, padding: 1)),
            ("act2", SiLU()),
            ("pool2", MaxPool2d(kernel_size: 2, stride: 2)),

            ("conv3", Conv2d(32, 64, kernel_size: 3, padding: 1)),
            ("act3", SiLU()),
            ("pool3", MaxPool2d(kernel_size: 2, stride: 2)),

            ("conv4", Conv2d(64, 128, kernel_size: 3, padding: 1)),
            ("act4", SiLU())
        );

        _head = Sequential(
            ("head_conv", Conv2d(128, 128, kernel_size: 3, padding: 1)),
            ("head_act", SiLU()),
            ("pred", Conv2d(128, _anchorCount * _attributeCount, kernel_size: 1))
        );

        RegisterComponents();
    }

    public int ClassCount => _classCount;

    public int AnchorCount => _anchorCount;

    public int PredictionCount => _anchorCount * _gridHeight * _gridWidth;

    public int AttributeCount => _attributeCount;

    public override Tensor forward(Tensor input)
    {
        var x = _backbone.forward(input);
        x = _head.forward(x);

        var batch = x.shape[0];
        x = x
            .view([batch, _anchorCount, _attributeCount, _gridHeight, _gridWidth])
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view([batch, PredictionCount, _attributeCount]);

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
            type: OnnxTensorType.Create<float>(["batch", 3, 64, 64])
        );

        var x = _backbone.Export(graph, input);
        x = _head.Export(graph, x);

        var reshapeToGridShape = graph.AddTensor(
            name: "detector_grid_shape",
            shape: [5],
            value: [0L, _anchorCount, _attributeCount, _gridHeight, _gridWidth]
        );

        x = graph.Reshape(
            name: "detector_grid_reshape",
            options: new ReshapeInputOptions
            {
                Data = x,
                Shape = reshapeToGridShape,
            }
        );

        x = graph.Transpose(
            name: "detector_grid_transpose",
            options: new TransposeInputOptions
            {
                Data = x,
                Perm = [0, 1, 3, 4, 2],
            }
        );

        var reshapeToPredictionsShape = graph.AddTensor(
            name: "detector_predictions_shape",
            shape: [3],
            value: [0L, PredictionCount, _attributeCount]
        );

        x = graph.Reshape(
            name: "detector_predictions_reshape",
            options: new ReshapeInputOptions
            {
                Data = x,
                Shape = reshapeToPredictionsShape,
            }
        );

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
            type: OnnxTensorType.Create<float>(["batch", PredictionCount, _attributeCount])
        );

        model.AddMetadataProps("architecture", "tiny-yolo-like");
        model.AddMetadataProps("grid_size", $"{_gridHeight}x{_gridWidth}");
        model.AddMetadataProps("anchors", _anchorCount.ToString());
        model.AddMetadataProps("classes", _classCount.ToString());
        model.AddMetadataProps("output_layout", "[batch, boxes, (x,y,w,h,obj,classes...)]");

        return model;
    }
}
