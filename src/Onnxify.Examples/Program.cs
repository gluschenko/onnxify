using System.Globalization;
using System.Text;
using Onnxify.Examples.Models;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace Onnxify.Examples
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

            A();
            return;

            var datasetDirectory = @"D:\Backups\ML\Ararat";
            // var datasetDirectory = @"D:\Backups\ML\microsoft-catsvsdogs-dataset\PetImages";
            var outputDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets");
            var device = cuda.is_available() ? CUDA : CPU;

            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

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

            Console.WriteLine("Press any key to pay respect...");
            Console.ReadKey();
        }

        static void A()
        {
            var outputDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets");

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

            // Example input tensor
            var sentences = torch.randint(0, charToIdx.Count, new long[] { 1, 10 }, device: torch.CPU);

            // Forward pass
            var output = model.forward(sentences);
            Console.WriteLine(output);

            // Save model
            model.eval();
            model.SaveModel("LSTMLIDModel.pt");

            var outputPath = Path.Combine(outputDirectory, "lang-lstm.onnx");
            var onnxModel = model.Export();
            onnxModel.Save(outputPath, true);
        }
    }
}
