using Onnxify.TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Onnxify.Examples.Models
{
    public sealed class TorchSharpExportShowcase : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _features;
        private readonly Module<Tensor, Tensor> _classifier;
        private readonly int _numClasses;

        public TorchSharpExportShowcase(string name = "torchsharp_export_showcase", int numClasses = 4)
            : base(name)
        {
            _numClasses = numClasses;

            _features = Sequential(
                ("pad", ReflectionPad2d((1L, 1L, 1L, 1L))),
                ("conv1", Conv2d(3, 8, kernel_size: 3)),
                ("gelu", GELU()),
                ("avgpool", AvgPool2d(kernel_size: 2, stride: 2)),
                ("conv2", Conv2d(8, 16, kernel_size: 3, padding: 1)),
                ("mish", Mish()),
                ("maxpool", MaxPool2d(kernel_size: 2, stride: 2)),
                ("pixel_unshuffle", PixelUnshuffle(2)),
                ("prelu", PReLU(1)),
                ("conv3", Conv2d(64, 16, kernel_size: 1)),
                ("silu", SiLU()),
                ("pixel_shuffle", PixelShuffle(2)),
                ("layer_norm", LayerNorm(new long[] { 4, 4, 4 })),
                ("adaptive_avg_pool", AdaptiveAvgPool2d(1)),
                ("flatten", Flatten())
            );

            _classifier = Sequential(
                ("fc1", Linear(4, 8)),
                ("selu", SELU()),
                ("softplus", Softplus(1.0, 20.0)),
                ("fc2", Linear(8, _numClasses)),
                ("log_softmax", LogSoftmax(1))
            );

            RegisterComponents();
        }

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
                type: OnnxTensorType.Create<float>(["batch", 3, 16, 16])
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

            return model;
        }
    }
}
