using System.Globalization;
using System.Text;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Data;
using Onnxify.ProjectGenerator;
using Onnxify.Safetensors;

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
            Test6();
            Test7();
            Test8();
            Test9();

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
        
        static void Test9()
        {
            var inputModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "mobilenetv2-12.onnx");
            var outputDirectoryPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "Generated", "MobileNetProject");

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(new ProjectGeneratorOptions
            {
                InputModelPath = inputModelPath,
                OutputDirectoryPath = outputDirectoryPath,
                ProjectName = "MobileNetGeneratedSample",
                Namespace = "Onnxify.Generated.MobileNet",
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

        static void Test7()
        {
            var outputDirectoryPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets");
            Directory.CreateDirectory(outputDirectoryPath);

            var outputPath = Path.Combine(outputDirectoryPath, "sample.safetensors");

            var values = new float[] { 1.0f, 2.0f, 3.5f, 4.5f };
            var data = values
                .SelectMany(BitConverter.GetBytes)
                .ToArray();

            var tensor = new TensorView(
                dtype: DataType.F32,
                shape: [2, 2],
                data: data
            );

            Onnxify.Safetensors.SafeTensors.SerializeToFile(
                data: [new KeyValuePair<string, TensorView>("weights", tensor)],
                metadata: new Dictionary<string, string>
                {
                    ["framework"] = "onnxify-console-test",
                    ["purpose"] = "roundtrip-demo",
                },
                path: outputPath
            );

            var raw = File.ReadAllBytes(outputPath);
            var safetensors = Onnxify.Safetensors.SafeTensors.Deserialize(raw);
            var loadedTensor = safetensors.Tensor("weights");
            var loadedValues = loadedTensor.Data.ToArray()
                .Chunk(sizeof(float))
                .Select(chunk => BitConverter.ToSingle(chunk))
                .ToArray();

            Console.WriteLine($"Safetensors file: {outputPath}");
            Console.WriteLine($"Safetensors names: {string.Join(", ", safetensors.Names())}");
            Console.WriteLine($"Safetensors loaded shape: [{string.Join(", ", loadedTensor.Shape)}]");
            Console.WriteLine($"Safetensors loaded values: {string.Join(", ", loadedValues.Select(x => x.ToString(CultureInfo.InvariantCulture)))}");
            Console.WriteLine($"Safetensors round-trip ok: {values.SequenceEqual(loadedValues)}");

            return;
        }

        private static void Test8()
        {
            try
            {
                using var gptModel = new GptOssQ4f16Model();
                using var alexNetModel = new Bvlcalexnet12QdqModel();
                using var realEsrganModel = new RealEsrganX4plusModel();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            using var model = new Mobilenetv212Model();

            var input = new DenseTensor<float>([1, 3, 224, 224]);

            FillWithDummyImage(input);

            using var outputs = model.Run(input);

            var logits = outputs.Output;

            var top5 = GetTopK(logits, k: 5);

            Console.WriteLine("Top-5 predictions:");
            foreach (var item in top5)
            {
                Console.WriteLine($"Class #{item.Index}: score={item.Score:F6}");
            }
        }

        private static void FillWithDummyImage(DenseTensor<float> input)
        {
            for (var y = 0; y < 224; y++)
            {
                for (var x = 0; x < 224; x++)
                {
                    input[0, 0, y, x] = 0.5f; // R
                    input[0, 1, y, x] = 0.5f; // G
                    input[0, 2, y, x] = 0.5f; // B
                }
            }
        }

        private static IReadOnlyList<(int Index, float Score)> GetTopK(Tensor<float> output, int k)
        {
            var classCount = output.Dimensions[1];

            return Enumerable
                .Range(0, classCount)
                .Select(i => (Index: i, Score: output[0, i]))
                .OrderByDescending(x => x.Score)
                .Take(k)
                .ToArray();
        }
    }

    public class ReluInputOptionsX
    {
        /// <summary>
        /// <b>X (parameter):</b>
        /// 
        /// Input tensor
        /// 
        /// <para>Allowed types: <c>OnnxTensor&lt;BFloat16&gt;</c>, <c>OnnxTensor&lt;double&gt;</c>, <c>OnnxTensor&lt;float&gt;</c>, <c>OnnxTensor&lt;Half&gt;</c>, <c>OnnxTensor&lt;short&gt;</c>, <c>OnnxTensor&lt;int&gt;</c>, <c>OnnxTensor&lt;long&gt;</c>, <c>OnnxTensor&lt;sbyte&gt;</c></para>
        /// <para>Type: Single</para>
        /// </summary>
        [AcceptType<OnnxTensor<float>>]
        public required IOnnxGraphEdge X { get; init; }
    }
}

