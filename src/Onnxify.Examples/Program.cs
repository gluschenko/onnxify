using System.Globalization;
using System.Text;
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

            var datasetDirectory = @"D:\Backups\ML\microsoft-catsvsdogs-dataset\PetImages";
            var outputDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets");

            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            var model = new AlexNet("alexnet", 2);
            var onnxModel = model.Export();

            var hflip = transforms.HorizontalFlip();
            var gray = transforms.Grayscale(3);
            var rotate = transforms.Rotate(90);
            var contrast = transforms.AdjustContrast(1.25);

            using (var dataset = new DataReader(datasetDirectory, [
                /*hflip,
                gray,
                rotate,
                contrast,*/
            ]))
            {
                dataset.Load(
                    width: 244,
                    height: 244,
                    channels: 3,
                    count: 5000
                );

                var trainer = new AlexNetTrainer(model, dataset);
                trainer.Train(
                    epochs: 10,
                    batchSize: 16,
                    learningRate: 0.001f
                );
            }

            var outputPath = Path.Combine(outputDirectory, "alexnet__test.onnx");
            onnxModel.Save(outputPath, true);

            Console.WriteLine("Press any key to pay respect...");
            Console.ReadKey();
        }
    }
}

