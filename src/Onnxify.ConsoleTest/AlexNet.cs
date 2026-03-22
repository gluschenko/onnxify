using TorchSharp;
using TorchSharp.Modules;
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
            var graph = model.Graph;

            var input = graph.AddInput(
                name: "input",
                type: OnnxTensorType.Create<float>([1, 3, 224, 224])
            );

            IOnnxGraphEdge x = input;

            var features = (Sequential)_features;
            var classifier = (Sequential)_classifier;

            int convIndex = 0;
            int poolIndex = 0;
            int reluIndex = 0;
            int linearIndex = 0;

            foreach (var module in features.children())
            {
                var typeName = module.GetType().Name;

                if (typeName.Contains("Conv2d", StringComparison.OrdinalIgnoreCase))
                {
                    dynamic conv = module;

                    long[] weightShape = ((IEnumerable<long>)conv.weight.shape).ToArray();
                    float[] weightValue = conv.weight
                        .detach()
                        .cpu()
                        .data<float>()
                        .ToArray();

                    var w = graph.AddTensor<float>(
                        name: $"conv{convIndex}_w",
                        shape: weightShape,
                        value: weightValue
                    );

                    IOnnxGraphEdge? b = null;

                    if (conv.bias is not null)
                    {
                        long[] biasShape = ((IEnumerable<long>)conv.bias.shape).ToArray();
                        float[] biasValue = conv.bias
                            .detach()
                            .cpu()
                            .data<float>()
                            .ToArray();

                        b = graph.AddTensor<float>(
                            name: $"conv{convIndex}_b",
                            shape: biasShape,
                            value: biasValue
                        );
                    }

                    long[] kernelShape = ((IEnumerable<long>)conv.kernel_size).ToArray();
                    long[] strides = ((IEnumerable<long>)conv.stride).ToArray();
                    long[] padding = ((IEnumerable<long>)conv.padding).ToArray();
                    long[] dilations = ((IEnumerable<long>)conv.dilation).ToArray();

                    x = graph.Conv(
                        name: $"conv{convIndex}",
                        options: new ConvInputOptions
                        {
                            X = x,
                            W = w,
                            B = b,
                            KernelShape = kernelShape,
                            Strides = strides,
                            Pads = [padding[0], padding[1], padding[0], padding[1]],
                            Dilations = dilations,
                            Group = (long)conv.groups
                        }
                    );

                    convIndex++;
                    continue;
                }

                if (typeName.Contains("ReLU", StringComparison.OrdinalIgnoreCase) ||
                    typeName.Equals("ReLU", StringComparison.OrdinalIgnoreCase))
                {
                    x = graph.Relu(
                        name: $"relu{reluIndex}",
                        options: new ReluInputOptions
                        {
                            X = x
                        }
                    );

                    reluIndex++;
                    continue;
                }

                if (typeName.Contains("MaxPool2d", StringComparison.OrdinalIgnoreCase))
                {
                    dynamic pool = module;

                    long[] kernelShape = ((IEnumerable<long>)pool.kernel_size).ToArray();

                    long[]? strides = null;
                    if (pool.stride is not null)
                    {
                        strides = ((IEnumerable<long>)pool.stride).ToArray();
                    }

                    long[]? padding = null;
                    if (pool.padding is not null)
                    {
                        var p = ((IEnumerable<long>)pool.padding).ToArray();
                        padding = [p[0], p[1], p[0], p[1]];
                    }

                    var poolOutput = graph.MaxPool(
                        name: $"maxpool{poolIndex}",
                        options: new MaxPoolInputOptions
                        {
                            X = x,
                            KernelShape = kernelShape,
                            Strides = strides,
                            Pads = padding
                        }
                    );

                    x = poolOutput.Y!;

                    poolIndex++;
                    continue;
                }

                throw new NotSupportedException($"Unsupported features module: {module.GetType().FullName}");
            }

            x = graph.GlobalAveragePool(
                name: "gap",
                options: new GlobalAveragePoolInputOptions
                {
                    X = x
                }
            );

            var reshapeShape = graph.AddTensor<long>(
                name: "reshape_shape",
                shape: [2],
                value: [1, 256]
            );

            x = graph.Reshape(
                name: "flatten",
                options: new ReshapeInputOptions
                {
                    Data = x,
                    Shape = reshapeShape
                }
            );

            foreach (var module in classifier.children())
            {
                var typeName = module.GetType().Name;

                if (typeName.Contains("Dropout", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (typeName.Contains("Linear", StringComparison.OrdinalIgnoreCase))
                {
                    dynamic linear = module;

                    long[] weightShape = ((IEnumerable<long>)linear.weight.shape).ToArray();
                    float[] weightValue = linear.weight
                        .detach()
                        .cpu()
                        .data<float>()
                        .ToArray();

                    var w = graph.AddTensor<float>(
                        name: $"fc{linearIndex}_w",
                        shape: weightShape,
                        value: weightValue
                    );

                    long[] biasShape = ((IEnumerable<long>)linear.bias.shape).ToArray();
                    float[] biasValue = linear.bias
                        .detach()
                        .cpu()
                        .data<float>()
                        .ToArray();

                    var b = graph.AddTensor<float>(
                        name: $"fc{linearIndex}_b",
                        shape: biasShape,
                        value: biasValue
                    );

                    x = graph.Gemm(
                        name: $"fc{linearIndex}",
                        options: new GemmInputOptions
                        {
                            A = x,
                            B = w,
                            C = b,
                            TransB = 1
                        }
                    );

                    linearIndex++;
                    continue;
                }

                if (typeName.Contains("ReLU", StringComparison.OrdinalIgnoreCase) ||
                    typeName.Equals("ReLU", StringComparison.OrdinalIgnoreCase))
                {
                    x = graph.Relu(
                        name: $"classifier_relu{reluIndex}",
                        options: new ReluInputOptions
                        {
                            X = x
                        }
                    );

                    reluIndex++;
                    continue;
                }

                throw new NotSupportedException($"Unsupported classifier module: {module.GetType().FullName}");
            }

            var outputEdge = graph.AddEdge("output");
            graph.Identity(
                name: "output_identity",
                options: new IdentityInputOutputOptions
                {
                    Input = x,
                    Output = outputEdge
                }
            );

            graph.AddOutput(
                name: "output",
                type: OnnxTensorType.Create<float>([1, 1000])
            );

            return model;
        }
    }
}
