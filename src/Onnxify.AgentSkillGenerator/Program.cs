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

        string repoRoot = SkillGeneratorPaths.FindRepositoryRoot(Directory.GetCurrentDirectory())
            ?? SkillGeneratorPaths.FindRepositoryRoot(AppContext.BaseDirectory)
            ?? throw new DirectoryNotFoundException("Repository root was not found.");

        var packageInventory = PackageInventory.Load(repoRoot);
        WritePackageInventory(repoRoot, packageInventory);

        int operatorExitCode = OperatorSkillGenerator.Run(args, packageInventory);
        if (operatorExitCode != 0)
        {
            return operatorExitCode;
        }

        return TorchSharpConverterSkillGenerator.Run(args, packageInventory);
    }

    private static void WritePackageInventory(string repoRoot, PackageInventory packageInventory)
    {
        string skillRoot = SkillGeneratorPaths.ResolveSkillRoot(repoRoot);
        string outputPath = Path.Combine(skillRoot, "references", "packages.md");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllText(
            outputPath,
            SkillGeneratorPaths.NormalizeLineEndings(packageInventory.BuildFullMarkdown()),
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)
        );
    }
}
