using System.Reflection;

namespace Onnxify.ProjectGenerator;

public sealed class ProjectGeneratorOptions
{
    public required string InputModelPath { get; init; }
    public required string OutputDirectoryPath { get; init; }
    public string? ProjectName { get; init; }
    public string? Namespace { get; init; }
    public string ProgramClassName { get; init; } = "Program";
    public string FactoryMethodName { get; init; } = "CreateModel";
    public string ProgramFileName { get; init; } = "Program.cs";
    public string TensorDirectoryName { get; init; } = "Assets";
    public bool GenerateProjectFile { get; init; } = true;
    public string? ProjectFileName { get; init; }
    public string? OnnxifyPackageVersion { get; init; }
    public bool Overwrite { get; init; }

    internal string GetProjectName()
    {
        var inputFileName = Path.GetFileNameWithoutExtension(InputModelPath);
        var fallback = string.IsNullOrWhiteSpace(inputFileName) ? "GeneratedOnnxifyModel" : inputFileName;
        return SanitizeIdentifier(ProjectName ?? fallback, "GeneratedOnnxifyModel");
    }

    internal string GetNamespace()
    {
        return SanitizeIdentifier(Namespace ?? GetProjectName(), "GeneratedOnnxifyModel");
    }

    internal string GetProjectFileName()
    {
        return string.IsNullOrWhiteSpace(ProjectFileName)
            ? $"{GetProjectName()}.csproj"
            : ProjectFileName;
    }

    internal string GetOnnxifyPackageVersion()
    {
        if (!string.IsNullOrWhiteSpace(OnnxifyPackageVersion))
        {
            return OnnxifyPackageVersion;
        }

        var informationalVersion = typeof(ProjectGeneratorOptions)
            .Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion;

        if (!string.IsNullOrWhiteSpace(informationalVersion))
        {
            var plusIndex = informationalVersion.IndexOf('+');
            return plusIndex >= 0
                ? informationalVersion[..plusIndex]
                : informationalVersion;
        }

        return typeof(ProjectGeneratorOptions).Assembly.GetName().Version?.ToString() ?? "1.0.0";
    }

    private static string SanitizeIdentifier(string value, string fallback)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        var chars = value
            .Select((ch, index) =>
            {
                if (char.IsLetterOrDigit(ch) || ch == '_')
                {
                    if (index == 0 && char.IsDigit(ch))
                    {
                        return '_';
                    }

                    return ch;
                }

                return '_';
            })
            .ToArray();

        var result = new string(chars).Trim('_');
        return string.IsNullOrWhiteSpace(result) ? fallback : result;
    }
}
