using System.Globalization;
using System.Text;

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

            Test4();

            Console.WriteLine("Press any key to pay respect...");
            Console.ReadKey();
        }

        static void Test4()
        {
            var model = new AlexNet("alexnet", 10);
            var onnxModel = model.Export();

            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "alexnet__test.onnx");
            onnxModel.Save(outputPath, true);
            return;
        }
    }
}

