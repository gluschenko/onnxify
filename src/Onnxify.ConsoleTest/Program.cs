using System.Globalization;
using System.Text;
using Google.Protobuf;
using Onnx;
using Onnxify.ProjectGenerator;

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

            Test0();
            Test1();
            Test2();
            Test4();
            Test5();
            Test6();

            Console.WriteLine("Press any key to pay respect...");
            Console.ReadKey();
        }

        static void Test0()
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions());

            var conv1_w = model.Graph.AddTensor<float>(
                name: "conv1_w",
                shape: [64, 3, 11, 11],
                value: new float[64 * 3 * 11 * 11]
            );

            var conv1_b = model.Graph.AddTensor<float>(
                name: "conv1_b",
                shape: [1, 3, 128, 128],
                value: new float[1 * 3 * 128 * 128]
            );

            var conv1_in = model.Graph.AddEdge("conv1_in");

            var conv = model.Graph.Conv(
                name: "conv1",
                options: new ConvInputOptions
                {
                    X = conv1_in,
                    W = conv1_w,
                    B = conv1_b,
                }
            );

            return;
        }

        static void Test1()
        {
            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq.onnx");
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq__test.onnx");
            var model = OnnxModel.FromFile(inputPath);

            var text = model.ToString();
            Console.WriteLine(text);

            model.Save(outputPath, true);
            return;
        }

        static void Test2()
        {
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "test.onnx");
            var model = OnnxModel.Create(new OnnxModelCreationOptions());

            var input = model.Graph.AddInput(
                name: "input_0",
                type: OnnxTensorType.Create<float>([256, 128])
            );

            var output = model.Graph.AddOutput(
                name: "probs_0",
                type: OnnxTensorType.Create<float>([128, 64])
            );

            var conv1_w = model.Graph.AddTensor<float>(
                name: "conv1_w",
                shape: [64, 3, 11, 11],
                value: new float[64 * 3 * 11 * 11]
            );

            var conv1_b = model.Graph.AddTensor<float>(
                name: "conv1_b",
                shape: [1, 3, 128, 128],
                value: new float[1 * 3 * 128 * 128]
            );

            var fc_w = model.Graph.AddTensor<float>(
                name: "fc_w",
                shape: [1000, 9216],
                value: new float[1000 * 9216]
            );

            var fc_b = model.Graph.AddTensor<float>(
                name: "fc_b",
                shape: [1000, 9216],
                value: new float[1000]
            );

            var conv1 = model.Graph.Conv(
                name: "conv1",
                options: new ConvInputOptions
                {
                    X = input,
                    W = conv1_w,
                    B = conv1_b,
                }
            );

            var relu1 = model.Graph.Relu(
                name: "relu1",
                options: new ReluInputOptions
                {
                    X = conv1
                }
            );

            var pool1 = model.Graph.MaxPool(
                name: "pool1",
                options: new MaxPoolInputOptions
                {
                    X = relu1,
                    KernelShape = [2, 2],
                    Strides = [2, 2]
                }
            );

            var flatten = model.Graph.Flatten(
                name: "flatten",
                options: new FlattenInputOptions
                {
                    Input = pool1.Y,
                }
            );

            model.Graph.Gemm(
                name: "fc",
                options: new GemmInputOutputOptions
                {
                    A = flatten,
                    B = fc_b,
                    C = fc_w,
                    Y = output,
                }
            );

            model.Save(outputPath, true);

            return;
        }

        static void Test4()
        {
            var model = new AlexNet("alexnet", 10);
            var onnxModel = model.Export();

            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "alexnet__test.onnx");
            onnxModel.Save(outputPath, true);
            return;
        }

        static void Test5()
        {
            var inputModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq.onnx");
            var model = OnnxModel.FromFile(inputModelPath);

            string GetOnnxTensorType(OnnxTensorType t)
            {
                if (t.Shape is null)
                {
                    return $"OnnxTensorType.Create<{t.Type.Name}>(\"{t.Denotation}\")";
                }

                var values = t.Shape.Dimensions
                    .Select(x =>
                    {
                        return x.GetValue() switch
                        {
                            long n => n.ToString(),
                            string s => $"\"{s}\"",
                            _ => throw new NotImplementedException($"Not implemented for '${x.GetValue()}'"),
                        };
                    })
                    .ToArray();

                return $"OnnxTensorType.Create<{t.Type.Name}>([{string.Join(", ", values)}], \"{t.Denotation}\")";
            }

            var flow = new StringBuilder();

            var edges = new Dictionary<IOnnxGraphEdge, string>();

            string GetEdgeName(IOnnxGraphEdge edge)
            {
                if (edges.TryGetValue(edge, out var name))
                {
                    return name;
                }
                else
                {
                    name = edge.Name
                        .Replace(".", "_")
                        .Replace("/", "_")
                        .Replace("-", "_")
                        .Replace("+", "_")
                        .Replace(" ", "_");

                    if (string.IsNullOrWhiteSpace(name))
                    {
                        name = $"{edge.GetType().Name}_{edges.Count}";
                    }

                    while (true)
                    {
                        if (!edges.Values.Contains(name))
                        {
                            edges.Add(edge, name);
                            break;
                        }

                        name += "_";
                    }

                    return name;
                }
            }

            foreach (var input in model.Graph.Inputs)
            {
                var type = input.Type switch
                {
                    OnnxTensorType t => GetOnnxTensorType(t),
                    _ => throw new NotImplementedException($"Not implemented for '${input.Type}'"),
                };

                flow.AppendLine($$"""
                var {{GetEdgeName(input)}} = model.Graph.AddInput(
                    name: "{{input.Name}}",
                    type: {{type}}
                );

                """);
            }

            foreach (var output in model.Graph.Outputs)
            {
                var type = output.Type switch
                {
                    OnnxTensorType t => GetOnnxTensorType(t),
                    _ => throw new NotImplementedException($"Not implemented for '${output.Type}'"),
                };

                flow.AppendLine($$"""
                var {{GetEdgeName(output)}} = model.Graph.AddOutput(
                    name: "{{output.Name}}",
                    type: {{type}}
                );

                """);
            }

            foreach (var node in model.Graph.Nodes)
            {
                flow.AppendLine($$"""
                model.Graph.AddNode(
                    name: "{{node.Name}}",
                    opType: "{{node.OpType}}",
                    domain: "{{node.Domain}}",
                    docString: "{{node.DocString}}",
                    inputs: [
                        {{string.Join(",\n", node.Inputs.Select(GetEdgeName)).Indent(2)}}
                    ],
                    outputs: [
                        {{string.Join(",\n", node.Outputs.Select(GetEdgeName)).Indent(2)}}
                    ],
                    attributes: []
                );

                """);
            }

            var text = $$"""
            using System;

            public class Program
            {
                public static void Main()
                {
                    var model = OnnxModel.Create(new OnnxModelCreationOptions());
                    
                    {{flow.ToString().Indent(2)}}
                }
            }
            """;
            Console.WriteLine(text);

            return;
        }

        static void Test6()
        {
            var inputModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "bvlcalexnet-12-qdq.onnx");
            var outputDirectoryPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "Generated", "AlexNetProject");

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(new ProjectGeneratorOptions
            {
                InputModelPath = inputModelPath,
                OutputDirectoryPath = outputDirectoryPath,
                ProjectName = "AlexNetGeneratedSample",
                Namespace = "Onnxify.Generated.AlexNet",
                Overwrite = true,
            });

            Console.WriteLine($"Generated Program: {result.ProgramFilePath}");
            Console.WriteLine($"Generated Project: {result.ProjectFilePath}");

            foreach (var tensorFilePath in result.TensorFilePaths)
            {
                Console.WriteLine($"Generated Tensor: {tensorFilePath}");
            }

            foreach (var warning in result.Warnings)
            {
                Console.WriteLine($"Generator Warning: {warning}");
            }

            return;
        }
    }

    public static class TextHelper
    {
        public static string Indent(this string text, int tabs)
        {
            var indent = new string(' ', tabs * 4);
            return text.Trim().Replace("\n", $"\n{indent}").Trim();
        }
    }

}

