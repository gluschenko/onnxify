using System.Globalization;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.HuggingFace;
using Onnxify.ProjectGenerator;
using Onnxify.Safetensors;
using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using TorchTensor = TorchSharp.torch.Tensor;

namespace Onnxify.ConsoleTest
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            Console.Title = nameof(Onnxify);
            Console.InputEncoding = Encoding.UTF8;
            Console.OutputEncoding = Encoding.UTF8;

            Test0();
            Test1();
            Test2();
            Test6();
            Test7();
            Test8();
            Test9();
            await Test10();
            Test11();

            if (!Console.IsInputRedirected)
            {
                Console.WriteLine("Press any key to pay respect...");
                Console.ReadKey();
            }
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
            var model = OnnxModel.FromFile(inputPath, new OnnxModelBaseOptions
            {
                NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.IgnoreIncompatible,
            });

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
            var model = OnnxModel.FromFile(inputModelPath, new OnnxModelBaseOptions
            {
                NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.IgnoreIncompatible,
            });

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(model, new ProjectGeneratorOptions
            {
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
            var model = OnnxModel.FromFile(inputModelPath, new OnnxModelBaseOptions
            {
                NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.IgnoreIncompatible,
            });

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(model, new ProjectGeneratorOptions
            {
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

        private static void Test11()
        {
            var originalModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "mobilenetv2-12.onnx");
            var roundtripModelPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "Assets",
                "Generated",
                "MobileNetTorchRoundtrip",
                "mobilenetv2-12-trained-roundtrip.onnx"
            );
            Directory.CreateDirectory(Path.GetDirectoryName(roundtripModelPath)!);

            torch.random.manual_seed(12_345);

            using var torchModule = new Mobilenetv212ModelTorchModule();
            torchModule.LoadWeightsFromOnnx(originalModelPath);
            torchModule.train();

            using var trainingInput = torch.randn([1, 3, 224, 224], dtype: ScalarType.Float32);
            using var trainingTarget = torch.randn([1, 1000], dtype: ScalarType.Float32);
            using var optimizer = torch.optim.SGD(torchModule.parameters(), 1e-5);

            for (var step = 0; step < 10; step++)
            {
                optimizer.zero_grad();
                using var prediction = torchModule.forward(trainingInput);
                using var loss = torch.nn.functional.mse_loss(prediction, trainingTarget);
                loss.backward();
                optimizer.step();

                Console.WriteLine($"MobileNet Torch training step {step + 1}: loss={loss.item<float>():F6}");
            }

            torchModule.eval();

            using var evaluationInput = torch.randn([1, 3, 224, 224], dtype: ScalarType.Float32);
            var denseInput = ToDenseTensor(evaluationInput);

            using var originalWrapper = new Mobilenetv212Model(originalModelPath);
            using var originalOutputs = originalWrapper.Run(denseInput);
            var original = originalOutputs.Output.ToArray();

            using var torchOutput = torchModule.forward(evaluationInput);
            var torchValues = ToFloatArray(torchOutput);

            var exported = torchModule.ExportOnnxModel(
                inputName: "input",
                outputName: "output",
                input: OnnxTensorType.Create<float>([1, 3, 224, 224]),
                output: OnnxTensorType.Create<float>([1, 1000]),
                options: new OnnxModelCreationOptions
                {
                    Opset = 22,
                    ProducerName = "onnxify-console-test",
                }
            );
            exported.Save(roundtripModelPath, overwrite: true);

            using var roundtripWrapper = new Mobilenetv212Model(roundtripModelPath);
            using var roundtripOutputs = roundtripWrapper.Run(denseInput);
            var roundtrip = roundtripOutputs.Output.ToArray();

            PrintDifference("original ONNX vs trained TorchModule", original, torchValues);
            PrintDifference("trained TorchModule vs exported ONNX", torchValues, roundtrip);
            PrintDifference("original ONNX vs exported ONNX", original, roundtrip);
            Console.WriteLine($"MobileNet trained roundtrip ONNX: {roundtripModelPath}");
        }

        private static DenseTensor<float> ToDenseTensor(TorchTensor tensor)
        {
            using var detached = tensor.detach();
            using var cpu = detached.cpu();
            return new DenseTensor<float>(
                cpu.data<float>().ToArray(),
                cpu.shape.Select(static x => (int)x).ToArray()
            );
        }

        private static float[] ToFloatArray(TorchTensor tensor)
        {
            using var detached = tensor.detach();
            using var cpu = detached.cpu();
            return cpu.data<float>().ToArray();
        }

        private static void PrintDifference(
            string label,
            IReadOnlyList<float> left,
            IReadOnlyList<float> right
        )
        {
            var count = Math.Min(left.Count, right.Count);
            var max = 0f;
            var sum = 0d;

            for (var index = 0; index < count; index++)
            {
                var difference = MathF.Abs(left[index] - right[index]);
                max = Math.Max(max, difference);
                sum += difference;
            }

            Console.WriteLine($"{label}: max_abs={max:E6}; mean_abs={(sum / count):E6}; count={count}");
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

        private static async Task Test10()
        {
            var repositoryId = "gluschenko/higgs-audio-v2-tokenizer-onnx";
            var outputDirectoryPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "Assets",
                "Generated",
                "HuggingFace",
                "higgs-audio-v2-tokenizer-onnx"
            );
            var revision = "main";
            var variant = ".onnx";
            var token = Environment.GetEnvironmentVariable("HF_TOKEN");

            var client = new HuggingFaceClient();

            var result = await client.DownloadRepositoryAsync(
                repositoryId,
                outputDirectoryPath,
                new HuggingFaceDownloadOptions
                {
                    Revision = revision,
                    AccessToken = string.IsNullOrWhiteSpace(token) ? null : token,
                    IncludePath = string.Equals(variant, "all", StringComparison.OrdinalIgnoreCase)
                        ? null
                        : path => path.Contains(variant, StringComparison.OrdinalIgnoreCase),
                    ProgressCallback = item =>
                    {
                        if (item.Completed)
                        {
                            Console.WriteLine($"Downloaded {item.FileIndex}/{item.FileCount}: {item.RepositoryPath}");
                        }
                    },
                    Overwrite = true,
                }
            );

            Console.WriteLine($"Hugging Face repository: {result.RepositoryId}@{result.Revision}");
            Console.WriteLine($"Output directory: {result.OutputDirectoryPath}");
            Console.WriteLine($"Downloaded files: {result.DownloadedFileCount}");
        }
    }
}

