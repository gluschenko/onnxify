using Google.Protobuf;
using Onnx;
using Onnxify.Abstractions;
using Onnxify.Operators;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Onnxify.ConsoleTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            Console.Title = nameof(Onnxify);
            Console.InputEncoding = Encoding.Unicode;
            Console.OutputEncoding = Encoding.Unicode;

            Test();
            /*AA();
            A();
            B();
            C();*/

            Console.WriteLine("Press any key to pay respect...");
            Console.ReadKey();
        }

        static void Test()
        {
            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq.onnx");
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq__test.onnx");
            var model = OnnxModel.FromFile(inputPath);

            var text = JsonSerializer.Serialize(model, new JsonSerializerOptions
            {
                WriteIndented = true,
            });

            model.Save(outputPath, true);
            Console.WriteLine(text);
        }

        static void AA()
        {
            Console.WriteLine("AA");

            static string Pascal(string s)
            {
                if (string.IsNullOrEmpty(s))
                {
                    return s;
                }

                return char.ToUpperInvariant(s[0]) + s.Substring(1);
            }

            static string InputName(string name)
            {
                var p = Pascal(name);

                if (p.Equals("Input", StringComparison.OrdinalIgnoreCase))
                {
                    return "Input";
                }

                return "Input" + p;
            }

            static string OutputName(string name)
            {
                var p = Pascal(name);

                if (p.Equals("Output", StringComparison.OrdinalIgnoreCase))
                {
                    return "Output";
                }

                return "Output" + p;
            }

            static string AttributeName(string name)
            {
                var p = Pascal(name);

                if (p.Equals("Attribute", StringComparison.OrdinalIgnoreCase))
                {
                    return "Attribute";
                }

                return "Attribute" + p;
            }

            static string MapType(string type)
            {
                return type switch
                {
                    "T" => nameof(TensorProto),
                    "Tind" => nameof(TensorProto),
                    "T1" => nameof(TensorProto),
                    "tensor(int64)" => nameof(TensorProto),
                    "tensor(int32)" => nameof(TensorProto),
                    "tensor(float)" => nameof(TensorProto),
                    "tensor(double)" => nameof(TensorProto),
                    "tensor(bool)" => nameof(TensorProto),

                    _ => throw new NotSupportedException($"Unsupported ONNX type: {type}")
                };
            }

            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "onnx_operators.json");
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "Onnxify", "Operators", "OnnxOperators.cs");

            var json = File.ReadAllText(inputPath);
            var root = JsonSerializer.Deserialize<OperatorSchemaRoot>(json) ?? throw new Exception();

            var sourceBuilder = new StringBuilder();

            sourceBuilder.AppendLine($"""
            using Onnx;
            using Onnxify.Abstractions;

            namespace Onnxify.Operators;

            """);

            foreach (var op in root.Operators)
            {
                var propBuilder = new StringBuilder();

                foreach (var x in op.Inputs)
                {
                    var required = x.Option == FormalParameterOption.Single ? " required " : " ";
                    var nullable = x.Option == FormalParameterOption.Optional ? "?" : "";

                    propBuilder.AppendLine($$"""
                        public{{required}}FormalParameter<{{MapType(x.Type)}}>{{nullable}} {{InputName(x.Name)}} { get; set; }
                    """);
                }

                if (op.Inputs.Count != 0)
                {
                    propBuilder.AppendLine();
                }

                foreach (var x in op.Outputs)
                {
                    var required = x.Option == FormalParameterOption.Single ? " required " : " ";
                    var nullable = x.Option == FormalParameterOption.Optional ? "?" : "";

                    propBuilder.AppendLine($$"""
                        public{{required}}FormalParameter<{{MapType(x.Type)}}>{{nullable}} {{OutputName(x.Name)}} { get; set; }
                    """);
                }

                if (op.Outputs.Count != 0)
                {
                    propBuilder.AppendLine();
                }

                foreach (var x in op.Attributes)
                {
                    var required = x.Required ? " required " : " ";
                    var nullable = x.Required ? "" : "?";

                    var typeEnum = (AttributeProto.Types.AttributeType)x.Type;

                    propBuilder.AppendLine($$"""
                        public{{required}}OperatorAttribute<{{OperatorHelpers.MapAttributeType(typeEnum)}}>{{nullable}} {{AttributeName(x.Name)}} { get; set; }
                    """);
                }

                if (op.Attributes.Count != 0)
                {
                    propBuilder.AppendLine();
                }

                sourceBuilder.AppendLine($$"""
                /// <summary>
                /// {{op.Name}} operator:
                /// <para>
                /// {{(op.Doc ?? "").Trim().Replace("\n", $"{Environment.NewLine}/// ")}}
                /// </para>
                /// </summary>
                public sealed class {{op.Name}} : {{nameof(Operator)}}
                {
                    public override string Name => "{{op.Name}}";
                    public override string Domain => "{{op.Domain}}";
                    public override int SinceVersion => {{op.SinceVersion}};

                    {{propBuilder.ToString().TrimStart()}}
                }

                """);
            }

            var sourceCode = sourceBuilder.ToString();

            File.WriteAllText(outputPath, sourceCode);
        }

        static void A()
        {
            Console.WriteLine("A");

            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "alexnet.onnx");

            var model = new AlexNet("test-model", 10, TorchSharp.torch.CPU);

            var builder = new OnnxGraphBuilder();

            foreach (var (name, tensor) in model.state_dict())
            {
                builder.AddWeight(name, tensor);
            }

            var onnx = AlexNetExporter.Export(model);

            using (var fs = File.Create(outputPath))
            {
                onnx.WriteTo(fs);
            }
        }

        static void B()
        {
            Console.WriteLine("B");

            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq.onnx");
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq__.onnx");
            var outputPath2 = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq.txt");

            var data = File.ReadAllBytes(inputPath);
            var model = ModelProto.Parser.ParseFrom(data);

            var text = model.Graph.ToString();
            File.WriteAllTextAsync(outputPath2, text);

            using (var fs = File.Create(outputPath))
            {
                model.WriteTo(fs);
            }
        }

        static void C()
        {
            Console.WriteLine("C");

            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "test.onnx");
            var outputPath2 = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "test.txt");

            var model = new ModelProto
            {
                IrVersion = 8,
                ProducerName = "alexnet-builder",
                Graph = new GraphProto
                {
                    Name = "AlexNet"
                }
            };

            var graph = model.Graph;

            // INPUT
            graph.Input.Add(new ValueInfoProto
            {
                Name = "input",
                Type = new TypeProto
                {
                    TensorType = new TypeProto.Types.Tensor
                    {
                        ElemType = (int)TensorProto.Types.DataType.Float,
                        Shape = new TensorShapeProto
                        {
                            Dim =
                            {
                                new TensorShapeProto.Types.Dimension { DimValue = 1 },
                                new TensorShapeProto.Types.Dimension { DimValue = 3 },
                                new TensorShapeProto.Types.Dimension { DimValue = 224 },
                                new TensorShapeProto.Types.Dimension { DimValue = 224 }
                            }
                        }
                    }
                }
            });

            // OUTPUT
            graph.Output.Add(new ValueInfoProto
            {
                Name = "output",
                Type = new TypeProto
                {
                    TensorType = new TypeProto.Types.Tensor
                    {
                        ElemType = (int)TensorProto.Types.DataType.Float,
                        Shape = new TensorShapeProto
                        {
                            Dim =
                            {
                                new TensorShapeProto.Types.Dimension { DimValue = 1 },
                                new TensorShapeProto.Types.Dimension { DimValue = 1000 }
                            }
                        }
                    }
                }
            });

            // Conv1
            graph.Node.Add(new NodeProto
            {
                OpType = "Conv",
                Name = "conv1",
                Input = { "input", "conv1_w", "conv1_b" },
                Output = { "conv1_out" }
            });

            graph.Node.Add(new NodeProto
            {
                OpType = "Relu",
                Name = "relu1",
                Input = { "conv1_out" },
                Output = { "relu1_out" }
            });

            graph.Node.Add(new NodeProto
            {
                OpType = "MaxPool",
                Name = "pool1",
                Input = { "relu1_out" },
                Output = { "pool1_out" },
                Attribute =
                {
                    new AttributeProto
                    {
                        Name = "kernel_shape",
                        Type = AttributeProto.Types.AttributeType.Ints,
                        Ints = {2,2}
                    },
                    new AttributeProto
                    {
                        Name = "strides",
                        Type = AttributeProto.Types.AttributeType.Ints,
                        Ints = {2,2}
                    }
                }
            });

            // Conv2
            graph.Node.Add(new NodeProto
            {
                OpType = "Conv",
                Name = "conv2",
                Input = { "pool1_out", "conv2_w", "conv2_b" },
                Output = { "conv2_out" },
            });

            graph.Node.Add(new NodeProto
            {
                OpType = "Relu",
                Name = "relu2",
                Input = { "conv2_out" },
                Output = { "relu2_out" }
            });

            graph.Node.Add(new NodeProto
            {
                OpType = "MaxPool",
                Name = "pool2",
                Input = { "relu2_out" },
                Output = { "pool2_out" }
            });

            // Flatten
            graph.Node.Add(new NodeProto
            {
                OpType = "Flatten",
                Name = "flatten",
                Input = { "pool2_out" },
                Output = { "flat_out" }
            });

            // Fully connected
            graph.Node.Add(new NodeProto
            {
                OpType = "Gemm",
                Name = "fc",
                Input = { "flat_out", "fc_w", "fc_b" },
                Output = { "output" }
            });

            // ---- INITIALIZERS (weights) ----

            graph.Initializer.Add(CreateTensor("conv1_w", [64, 3, 11, 11]));
            graph.Initializer.Add(CreateTensor("conv1_b", [64]));

            graph.Initializer.Add(CreateTensor("conv2_w", [192, 64, 5, 5]));
            graph.Initializer.Add(CreateTensor("conv2_b", [192]));

            graph.Initializer.Add(CreateTensor("fc_w", [1000, 9216]));
            graph.Initializer.Add(CreateTensor("fc_b", [1000]));

            var text = model.Graph.ToString();
            File.WriteAllTextAsync(outputPath2, text);

            using (var fs = File.Create(outputPath))
            {
                model.WriteTo(fs);
            }
        }

        static TensorProto CreateTensor(string name, long[] shape)
        {
            var size = shape.Aggregate(1L, (a, b) => a * b);

            var tensor = new TensorProto
            {
                Name = name,
                DataType = (int)TensorProto.Types.DataType.Float
            };

            tensor.Dims.AddRange(shape);

            for (int i = 0; i < size; i++)
            {
                var value = ((Random.Shared.NextSingle() - 0.5f) * 2f) * 0.01f;
                tensor.FloatData.Add(value);
            }

            return tensor;
        }
    }
}
