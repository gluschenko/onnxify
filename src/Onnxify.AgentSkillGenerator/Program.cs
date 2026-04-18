using System.Globalization;
using System.Text;

namespace Onnxify.AgentSkillGenerator;

internal static class Program
{
    private static int Main(string[] args)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        Console.Title = nameof(Onnxify);
        Console.InputEncoding = Encoding.UTF8;
        Console.OutputEncoding = Encoding.UTF8;

        int operatorExitCode = OperatorSkillGenerator.Run(args);
        if (operatorExitCode != 0)
        {
            return operatorExitCode;
        }

        return TorchSharpConverterSkillGenerator.Run(args);
    }
}
