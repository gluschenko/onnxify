using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Onnxify.Examples.Models
{
    public class AlexNet : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _features;
        private readonly Module<Tensor, Tensor> _classifier;
        private readonly int _numClasses;

        public AlexNet(string name, int numClasses, Device? device = null) : base(name)
        {
            _numClasses = numClasses;

            _features = Sequential(
                // Conv1
                ("conv1", Conv2d(3, 96, kernel_size: 11, stride: 4)),
                ("relu1", ReLU(inplace: true)),
                ("pool1", MaxPool2d(kernel_size: 3, stride: 2)),

                // Conv2 (groups=2)
                ("conv2", Conv2d(96, 256, kernel_size: 5, padding: 2, groups: 2)),
                ("relu2", ReLU(inplace: true)),
                ("pool2", MaxPool2d(kernel_size: 3, stride: 2)),

                // Conv3
                ("conv3", Conv2d(256, 384, kernel_size: 3, padding: 1)),
                ("relu3", ReLU(inplace: true)),

                // Conv4 (groups=2)
                ("conv4", Conv2d(384, 384, kernel_size: 3, padding: 1, groups: 2)),
                ("relu4", ReLU(inplace: true)),

                // Conv5 (groups=2)
                ("conv5", Conv2d(384, 256, kernel_size: 3, padding: 1, groups: 2)),
                ("relu5", ReLU(inplace: true)),
                ("pool5", MaxPool2d(kernel_size: 3, stride: 2))
            );

            _classifier = Sequential(
                ("dropout1", Dropout()),
                ("fc1", Linear(256 * 6 * 6, 4096)),
                ("relu6", ReLU(inplace: true)),

                ("dropout2", Dropout()),
                ("fc2", Linear(4096, 4096)),
                ("relu7", ReLU(inplace: true)),

                ("fc3", Linear(4096, numClasses))
            );

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
            {
                this.to(device);
            }
        }

        public override Tensor forward(Tensor input)
        {
            var x = _features.forward(input);
            x = x.view([x.shape[0], 256 * 6 * 6]);
            x = _classifier.forward(x);
            return x;
        }

        public OnnxModel Export()
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions());
            var graph = model.Graph;

            var input = graph.AddInput(
                name: "input",
                type: OnnxTensorType.Create<float>(["batch", 3, 227, 227])
            );

            var x = _features.Export(graph, input);

            x = graph.Flatten(
                name: "flatten",
                options: new FlattenInputOptions
                {
                    Input = x,
                    Axis = 1,
                }
            );

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
