using System.Globalization;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Examples.Models;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using Tensor = TorchSharp.torch.Tensor;

namespace Onnxify.Examples
{
    internal class Program
    {
        private sealed record ExampleMenuItem(string Key, string Title, Action Run);

        static void Main(string[] args)
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            Console.Title = nameof(Onnxify);
            Console.InputEncoding = Encoding.Unicode;
            Console.OutputEncoding = Encoding.Unicode;

            var items = new[]
            {
                new ExampleMenuItem("showcase", "TorchSharp export showcase", RunTorchSharpExportShowcase),
                new ExampleMenuItem("lstm", "Language LSTM export", ExportLanguageLstm),
                new ExampleMenuItem("gpt2", "Mini GPT-2-like decoder export", RunMiniGpt2LikeExample),
                new ExampleMenuItem("mobilenet", "MobileNetV1-like classifier export", RunMobileNetV1LikeExample),
                new ExampleMenuItem("yolo", "Tiny YOLO-like detector export", RunTinyYoloLikeExample),
                new ExampleMenuItem("alexnet", "AlexNet training and export", TrainAlexNet),
                new ExampleMenuItem("all", "Run all examples", RunAllExamples),
            };

            var selectedItem = SelectExample(items, args);
            selectedItem.Run();
        }

        static void RunTorchSharpExportShowcase()
        {
            torch.random.manual_seed(1234);

            var outputDirectory = EnsureAssetsDirectory();
            var outputPath = Path.Combine(outputDirectory, "torchsharp-export-showcase.onnx");

            var model = new TorchSharpExportShowcase();
            model.eval();

            using var input = torch.randn(new long[] { 2, 3, 16, 16 }, device: torch.CPU);
            using var torchOutput = model.forward(input).cpu();

            var onnxModel = model.Export();
            onnxModel.Save(outputPath, true);

            using var session = new InferenceSession(outputPath);
            var onnxOutput = RunOnnx(session, input);

            Console.WriteLine("TorchSharp export showcase");
            Console.WriteLine($"Saved ONNX model: {outputPath}");
            Console.WriteLine($"Output shape: [{string.Join(", ", torchOutput.shape)}]");
            Console.WriteLine($"Max abs diff Torch vs ONNX: {ComputeMaxAbsDiff(torchOutput, onnxOutput):G9}");
            Console.WriteLine();
            Console.WriteLine("Operators exercised by this example:");
            Console.WriteLine(
                "ReflectionPad2d, Conv2d, GELU, AvgPool2d, Mish, MaxPool2d, PixelUnshuffle, PReLU, SiLU, PixelShuffle, LayerNorm, AdaptiveAvgPool2d, Flatten, Linear, SELU, Softplus, LogSoftmax"
            );
        }

        static void ExportLanguageLstm()
        {
            var outputDirectory = EnsureAssetsDirectory();

            var charToIdx = new Dictionary<string, int>
            {
                { "PAD", 0 },
                { "a", 1 },
                { "b", 2 },
            };

            var langToIdx = new Dictionary<string, int>
            {
                { "en", 0 },
                { "fr", 1 },
            };

            var embeddingDim = 128;
            var hiddenDim = 256;
            var layers = 2;

            var model = new LSTMLIDModel(charToIdx, langToIdx, langToIdx.Count, embeddingDim, hiddenDim, layers);

            var sentences = torch.randint(0, charToIdx.Count, new long[] { 1, 10 }, device: torch.CPU);
            var output = model.forward(sentences);
            Console.WriteLine(output);

            model.eval();
            model.SaveModel("LSTMLIDModel.pt");

            var outputPath = Path.Combine(outputDirectory, "lang-lstm.onnx");
            var onnxModel = model.Export();
            Console.WriteLine(onnxModel.ToString());

            onnxModel.Save(outputPath, true);
        }

        static void RunMiniGpt2LikeExample()
        {
            torch.random.manual_seed(1234);

            var outputDirectory = EnsureAssetsDirectory();
            var outputPath = Path.Combine(outputDirectory, "mini-gpt2-like.onnx");

            var model = new MiniGpt2LikeModel();
            model.eval();

            using var tokenIds = torch.tensor(
                    new long[]
                    {
                        1, 5, 7, 9, 11, 13, 15, 17,
                        2, 4, 6, 8, 10, 12, 14, 16,
                    },
                    dtype: ScalarType.Int64,
                    device: torch.CPU
                )
                .reshape(2, model.MaxSequenceLength);

            using var torchOutput = model.forward(tokenIds).cpu();

            var onnxModel = model.Export();
            onnxModel.Save(outputPath, true);

            using var session = new InferenceSession(outputPath);
            var onnxOutput = RunOnnxInt64(session, tokenIds);

            Console.WriteLine("Mini GPT-2-like decoder export");
            Console.WriteLine($"Saved ONNX model: {outputPath}");
            Console.WriteLine($"Input shape: [{string.Join(", ", tokenIds.shape)}]");
            Console.WriteLine($"Output shape: [{string.Join(", ", torchOutput.shape)}]");
            Console.WriteLine($"Max abs diff Torch vs ONNX: {ComputeMaxAbsDiff(torchOutput, onnxOutput):G9}");
            Console.WriteLine();
            Console.WriteLine("Architecture:");
            Console.WriteLine("Token embedding, position embedding, 2 GPT-2-style transformer blocks, fused QKV causal self-attention, GELU MLP, tied output projection");
        }

        static void RunTinyYoloLikeExample()
        {
            torch.random.manual_seed(1234);

            var outputDirectory = EnsureAssetsDirectory();
            var outputPath = Path.Combine(outputDirectory, "tiny-yolo-like.onnx");

            var model = new TinyYoloLikeDetector();
            model.eval();

            using var input = torch.randn(new long[] { 2, 3, 64, 64 }, device: torch.CPU);
            using var torchOutput = model.forward(input).cpu();

            var onnxModel = model.Export();
            onnxModel.Save(outputPath, true);

            using var session = new InferenceSession(outputPath);
            var onnxOutput = RunOnnx(session, input);

            Console.WriteLine("Tiny YOLO-like detector export");
            Console.WriteLine($"Saved ONNX model: {outputPath}");
            Console.WriteLine($"Input shape: [{string.Join(", ", input.shape)}]");
            Console.WriteLine($"Output shape: [{string.Join(", ", torchOutput.shape)}]");
            Console.WriteLine($"Max abs diff Torch vs ONNX: {ComputeMaxAbsDiff(torchOutput, onnxOutput):G9}");
            Console.WriteLine();
            Console.WriteLine("Detection layout:");
            Console.WriteLine($"[batch, {model.PredictionCount}, {model.AttributeCount}] where attrs = (x, y, w, h, objectness, classes...)");
        }

        static void RunMobileNetV1LikeExample()
        {
            torch.random.manual_seed(1234);

            var outputDirectory = EnsureAssetsDirectory();
            var outputPath = Path.Combine(outputDirectory, "mobilenet-v1-like.onnx");

            var model = new MobileNetV1LikeClassifier();
            model.eval();

            using var input = torch.randn(new long[] { 2, 3, 96, 96 }, device: torch.CPU);
            using var torchOutput = model.forward(input).cpu();

            var onnxModel = model.Export();
            onnxModel.Save(outputPath, true);

            using var session = new InferenceSession(outputPath);
            var onnxOutput = RunOnnx(session, input);

            Console.WriteLine("MobileNetV1-like classifier export");
            Console.WriteLine($"Saved ONNX model: {outputPath}");
            Console.WriteLine($"Input shape: [{string.Join(", ", input.shape)}]");
            Console.WriteLine($"Output shape: [{string.Join(", ", torchOutput.shape)}]");
            Console.WriteLine($"Max abs diff Torch vs ONNX: {ComputeMaxAbsDiff(torchOutput, onnxOutput):G9}");
            Console.WriteLine();
            Console.WriteLine("Architecture:");
            Console.WriteLine("Conv-BN-ReLU6 stem, depthwise-separable MobileNet blocks, adaptive average pooling, linear classifier");
        }

        static void TrainAlexNet()
        {
            var datasetDirectory = @"D:\Backups\ML\Ararat";
            // var datasetDirectory = @"D:\Backups\ML\microsoft-catsvsdogs-dataset\PetImages";
            var outputDirectory = EnsureAssetsDirectory();
            var device = cuda.is_available() ? CUDA : CPU;

            var model = new AlexNet("alexnet", 2);

            var hflip = transforms.HorizontalFlip();
            var gray = transforms.Grayscale(3);
            var rotate = transforms.Rotate(90);
            var contrast = transforms.AdjustContrast(1.25);

            using (var dataset = new DataReader(datasetDirectory, [
                // hflip,
                // gray,
                // rotate,
                // contrast,
            ]))
            {
                dataset.Load(
                    width: 227,
                    height: 227,
                    channels: 3,
                    count: 5000
                );
                dataset.Split(testFraction: 0.2f, seed: 42);

                Console.WriteLine($"Train samples: {dataset.TrainSampleCount}");
                Console.WriteLine($"Test samples:  {dataset.TestSampleCount}");

                var trainer = new AlexNetTrainer(model, dataset);
                trainer.Train(
                    epochs: 65,
                    batchSize: 256,
                    learningRate: 0.0001f,
                    schedulerStepSize: 30,
                    schedulerGamma: 0.5f,
                    minLearningRate: 1e-5f,
                    device: device
                );

                var outputPath = Path.Combine(outputDirectory, "alexnet__test.onnx");
                var onnxModel = model.Export();
                onnxModel.Save(outputPath, true);

                var torchEvaluation = ModelEvaluator.EvaluateTorch(model, dataset, batchSize: 256, device);
                var onnxEvaluation = ModelEvaluator.EvaluateOnnx(outputPath, dataset, batchSize: 256);

                ModelEvaluator.PrintConfusionMatrix(
                    "Torch Confusion Matrix",
                    torchEvaluation,
                    dataset.LabelNames
                );

                ModelEvaluator.PrintConfusionMatrix(
                    "ONNX Confusion Matrix",
                    onnxEvaluation,
                    dataset.LabelNames
                );
            }
        }

        static void RunAllExamples()
        {
            RunTorchSharpExportShowcase();
            ExportLanguageLstm();
            RunMiniGpt2LikeExample();
            RunMobileNetV1LikeExample();
            RunTinyYoloLikeExample();
            TrainAlexNet();
        }

        static ExampleMenuItem SelectExample(
            IReadOnlyList<ExampleMenuItem> items,
            IReadOnlyList<string> args
        )
        {
            var selector = args.FirstOrDefault()?.Trim();
            if (!string.IsNullOrWhiteSpace(selector))
            {
                var item = FindExample(items, selector);
                if (item is not null)
                {
                    return item;
                }
            }

            while (true)
            {
                Console.WriteLine("Select example:");
                for (var i = 0; i < items.Count; i++)
                {
                    Console.WriteLine($"{i + 1}. {items[i].Title} ({items[i].Key})");
                }

                Console.Write("Enter number or key: ");
                var input = Console.ReadLine()?.Trim();
                var item = FindExample(items, input);
                if (item is not null)
                {
                    Console.WriteLine();
                    return item;
                }

                Console.WriteLine("Unknown selection. Try again.");
                Console.WriteLine();
            }
        }

        static ExampleMenuItem? FindExample(
            IReadOnlyList<ExampleMenuItem> items,
            string? selector
        )
        {
            if (string.IsNullOrWhiteSpace(selector))
            {
                return null;
            }

            if (int.TryParse(selector, out var index) && index >= 1 && index <= items.Count)
            {
                return items[index - 1];
            }

            return items.FirstOrDefault(item =>
                string.Equals(item.Key, selector, StringComparison.OrdinalIgnoreCase)
            );
        }

        static string EnsureAssetsDirectory()
        {
            var outputDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets");
            Directory.CreateDirectory(outputDirectory);
            return outputDirectory;
        }

        static float[] RunOnnx(InferenceSession session, Tensor input)
        {
            var inputData = input.cpu().data<float>().ToArray();
            var inputTensor = new DenseTensor<float>(
                inputData,
                new[] { (int)input.shape[0], (int)input.shape[1], (int)input.shape[2], (int)input.shape[3] }
            );

            var inputValue = NamedOnnxValue.CreateFromTensor("input", inputTensor);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(new[] { inputValue });
            return results.Single().AsTensor<float>().ToArray();
        }

        static float[] RunOnnxInt64(InferenceSession session, Tensor input)
        {
            var inputData = input.cpu().data<long>().ToArray();
            var inputTensor = new DenseTensor<long>(
                inputData,
                new[] { (int)input.shape[0], (int)input.shape[1] }
            );

            var inputValue = NamedOnnxValue.CreateFromTensor("input", inputTensor);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(new[] { inputValue });
            return results.Single().AsTensor<float>().ToArray();
        }

        static float ComputeMaxAbsDiff(Tensor torchOutput, IReadOnlyList<float> onnxOutput)
        {
            var torchData = torchOutput.data<float>().ToArray();
            if (torchData.Length != onnxOutput.Count)
            {
                throw new InvalidOperationException(
                    $"Torch output length {torchData.Length} does not match ONNX output length {onnxOutput.Count}."
                );
            }

            var maxAbsDiff = 0f;
            for (var i = 0; i < torchData.Length; i++)
            {
                maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(torchData[i] - onnxOutput[i]));
            }

            return maxAbsDiff;
        }
    }
}
