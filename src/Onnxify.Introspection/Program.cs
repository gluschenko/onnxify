namespace Onnxify.Introspection;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            if (args.Length == 0)
            {
                PrintHelp();
                return 1;
            }

            var mode = args[0];
            var modeArgs = args[1..];

            return mode switch
            {
                "onnxify" => OnnxifyMode.Run(modeArgs),
                "torchsharp" => TorchSharpMode.Run(modeArgs),
                "--help" or "-h" => PrintHelpAndReturnSuccess(),
                _ => throw new ArgumentException($"Unknown mode '{mode}'. Supported modes: onnxify, torchsharp."),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static int PrintHelpAndReturnSuccess()
    {
        PrintHelp();
        return 0;
    }

    private static void PrintHelp()
    {
        Console.WriteLine(
            """
            Unified operator markdown generator

            Usage:
              Onnxify.Introspection onnxify [options]
              Onnxify.Introspection torchsharp [options]

            Modes:
              onnxify     Generate operator docs from src/Onnxify/Assets/onnx_operators.json
              torchsharp  Generate operator docs from TorchSharp reflection and XML docs
            """
        );
    }
}
