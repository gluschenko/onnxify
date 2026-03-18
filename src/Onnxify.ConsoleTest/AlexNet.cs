using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Onnxify.ConsoleTest
{
    public class AlexNet : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _features;
        private readonly Module<Tensor, Tensor> _avgPool;
        private readonly Module<Tensor, Tensor> _classifier;

        public AlexNet(string name, int numClasses, Device? device = null) : base(name)
        {
            _features = Sequential(
                ("c1", Conv2d(3, 64, kernel_size: 3, stride: 2, padding: 1)),
                ("r1", ReLU(inplace: true)),
                ("mp1", MaxPool2d(kernel_size: [2, 2])),
                ("c2", Conv2d(64, 192, kernel_size: 3, padding: 1)),
                ("r2", ReLU(inplace: true)),
                ("mp2", MaxPool2d(kernel_size: [2, 2])),
                ("c3", Conv2d(192, 384, kernel_size: 3, padding: 1)),
                ("r3", ReLU(inplace: true)),
                ("c4", Conv2d(384, 256, kernel_size: 3, padding: 1)),
                ("r4", ReLU(inplace: true)),
                ("c5", Conv2d(256, 256, kernel_size: 3, padding: 1)),
                ("r5", ReLU(inplace: true)),
                ("mp3", MaxPool2d(kernel_size: [2, 2]))
            );

            _avgPool = AdaptiveAvgPool2d([2, 2]);

            _classifier = Sequential(
                ("d1", Dropout()),
                ("l1", Linear(256 * 2 * 2, 4096)),
                ("r1", ReLU(inplace: true)),
                ("d2", Dropout()),
                ("l2", Linear(4096, 4096)),
                ("r3", ReLU(inplace: true)),
                ("d3", Dropout()),
                ("l3", Linear(4096, numClasses))
            );

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
            {
                this.to(device);
            }
        }

        public override Tensor forward(Tensor input)
        {
            var f = _features.forward(input);
            var avg = _avgPool.forward(f);

            var x = avg.view([avg.shape[0], 256 * 2 * 2]);

            return _classifier.forward(x);
        }

        public OnnxModel Onnxify()
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions());



            return model;
        }
    }
}
