using System.Globalization;
using System.Text;
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
                data: data);

            Onnxify.Safetensors.Safetensors.SerializeToFile(
                data: [new KeyValuePair<string, TensorView>("weights", tensor)],
                metadata: new Dictionary<string, string>
                {
                    ["framework"] = "onnxify-console-test",
                    ["purpose"] = "roundtrip-demo",
                },
                path: outputPath);

            var raw = File.ReadAllBytes(outputPath);
            var safetensors = Onnxify.Safetensors.Safetensors.Deserialize(raw);
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
    }
}

