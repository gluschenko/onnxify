using System.Reflection;
using System.Text.RegularExpressions;
using Onnxify.HuggingFace;
using Onnxify.ProjectGenerator;

namespace Onnxify.CLI;

public static class App
{
    public static int Run(string[] args)
    {
        return Run(args, Console.Out, Console.Error);
    }

    public static int Run(string[] args, TextWriter standardOutput, TextWriter standardError)
    {
        return Run(args, standardOutput, standardError, null);
    }

    public static int Run(
        string[] args,
        TextWriter standardOutput,
        TextWriter standardError,
        HuggingFaceClient? huggingFaceClient)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(standardOutput);
        ArgumentNullException.ThrowIfNull(standardError);

        try
        {
            if (args.Length == 0)
            {
                WriteHelp(standardError);
                return 1;
            }

            if (IsHelp(args[0]))
            {
                WriteHelp(standardOutput);
                return 0;
            }

            if (IsVersion(args[0]))
            {
                standardOutput.WriteLine(GetToolVersion());
                return 0;
            }

            return args[0].ToLowerInvariant() switch
            {
                "onnx" => RunOnnx(args[1..], standardOutput, standardError),
                "safetensors" => RunSafetensors(args[1..], standardOutput, standardError),
                "project" => RunProject(args[1..], standardOutput, standardError),
                "hf" or "huggingface" => RunHuggingFace(args[1..], standardOutput, standardError, huggingFaceClient),
                _ => Fail(standardError, $"Unknown command '{args[0]}'.", WriteHelp),
            };
        }
        catch (Exception ex)
        {
            standardError.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunOnnx(string[] args, TextWriter standardOutput, TextWriter standardError)
    {
        if (args.Length == 0)
        {
            WriteOnnxHelp(standardError);
            return 1;
        }

        if (IsHelp(args[0]))
        {
            WriteOnnxHelp(standardOutput);
            return 0;
        }

        if (args.Length != 2)
        {
            return Fail(standardError, "The onnx command expects a subcommand and a model path.", WriteOnnxHelp);
        }

        var model = OnnxModel.FromFile(args[1]);

        switch (args[0].ToLowerInvariant())
        {
            case "show":
                standardOutput.WriteLine(model);
                return 0;
            case "io":
            case "inputs-outputs":
                standardOutput.WriteLine(FormatInputsOutputs(model));
                return 0;
            default:
                return Fail(standardError, $"Unknown onnx subcommand '{args[0]}'.", WriteOnnxHelp);
        }
    }

    private static int RunSafetensors(string[] args, TextWriter standardOutput, TextWriter standardError)
    {
        if (args.Length == 0)
        {
            WriteSafetensorsHelp(standardError);
            return 1;
        }

        if (IsHelp(args[0]))
        {
            WriteSafetensorsHelp(standardOutput);
            return 0;
        }

        if (args.Length != 2 || !args[0].Equals("show", StringComparison.OrdinalIgnoreCase))
        {
            return Fail(standardError, "The safetensors command supports only 'show <model.safetensors>'.", WriteSafetensorsHelp);
        }

        var safetensors = Safetensors.SafeTensors.Deserialize(File.ReadAllBytes(args[1]));
        standardOutput.WriteLine(safetensors);
        return 0;
    }

    private static int RunHuggingFace(
        string[] args,
        TextWriter standardOutput,
        TextWriter standardError,
        HuggingFaceClient? client)
    {
        if (args.Length == 0)
        {
            WriteHuggingFaceHelp(standardError);
            return 1;
        }

        if (IsHelp(args[0]))
        {
            WriteHuggingFaceHelp(standardOutput);
            return 0;
        }

        if (!args[0].Equals("download", StringComparison.OrdinalIgnoreCase))
        {
            return Fail(standardError, $"Unknown Hugging Face subcommand '{args[0]}'.", WriteHuggingFaceHelp);
        }

        if (args.Length < 3)
        {
            return Fail(standardError, "The huggingface download command expects a repository id and output directory.", WriteHuggingFaceHelp);
        }

        var repositoryId = args[1];
        var outputDirectoryPath = args[2];
        var revision = "main";
        string? accessToken = null;
        var tokenEnvironmentVariable = "HF_TOKEN";
        var includePatterns = new List<string>();
        var excludePatterns = new List<string>();
        string? variant = null;
        var overwrite = false;
        var quiet = false;

        for (var i = 3; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--revision":
                    revision = ReadOptionValue(args, ref i, "--revision");
                    break;
                case "--token":
                    accessToken = ReadOptionValue(args, ref i, "--token");
                    break;
                case "--token-env":
                    tokenEnvironmentVariable = ReadOptionValue(args, ref i, "--token-env");
                    break;
                case "--include":
                    includePatterns.Add(ReadOptionValue(args, ref i, "--include"));
                    break;
                case "--exclude":
                    excludePatterns.Add(ReadOptionValue(args, ref i, "--exclude"));
                    break;
                case "--variant":
                    variant = ReadOptionValue(args, ref i, "--variant");
                    break;
                case "--overwrite":
                    overwrite = true;
                    break;
                case "--quiet":
                    quiet = true;
                    break;
                default:
                    return Fail(standardError, $"Unknown option '{args[i]}'.", WriteHuggingFaceHelp);
            }
        }

        if (string.IsNullOrWhiteSpace(accessToken) && !string.IsNullOrWhiteSpace(tokenEnvironmentVariable))
        {
            accessToken = Environment.GetEnvironmentVariable(tokenEnvironmentVariable);
        }

        var includeMatchers = includePatterns.Select(CreatePathMatcher).ToArray();
        var excludeMatchers = excludePatterns.Select(CreatePathMatcher).ToArray();
        var selectedVariant = variant;

        client ??= new HuggingFaceClient();
        var result = client.DownloadRepositoryAsync(
            repositoryId,
            outputDirectoryPath,
            new HuggingFaceDownloadOptions
            {
                Revision = revision,
                AccessToken = string.IsNullOrWhiteSpace(accessToken) ? null : accessToken,
                IncludePath = path => ShouldIncludeHuggingFacePath(path, includeMatchers, selectedVariant),
                ExcludePath = path => excludeMatchers.Any(match => match(path)),
                Overwrite = overwrite,
                ProgressCallback = progress =>
                {
                    if (!quiet && progress.Completed)
                    {
                        standardOutput.WriteLine($"Downloaded {progress.FileIndex}/{progress.FileCount}: {progress.RepositoryPath}");
                    }
                },
            }).GetAwaiter().GetResult();

        standardOutput.WriteLine(FormatHuggingFaceDownloadResult(result));
        return 0;
    }

    private static int RunProject(string[] args, TextWriter standardOutput, TextWriter standardError)
    {
        if (args.Length == 0)
        {
            WriteProjectHelp(standardError);
            return 1;
        }

        if (IsHelp(args[0]))
        {
            WriteProjectHelp(standardOutput);
            return 0;
        }

        if (!args[0].Equals("generate", StringComparison.OrdinalIgnoreCase))
        {
            return Fail(standardError, $"Unknown project subcommand '{args[0]}'.", WriteProjectHelp);
        }

        if (args.Length < 3)
        {
            return Fail(standardError, "The project generate command expects an input model path and output directory path.", WriteProjectHelp);
        }

        var inputModelPath = args[1];
        var outputDirectoryPath = args[2];
        string? projectName = null;
        string? namespaceName = null;
        string? packageVersion = null;
        string? programClassName = null;
        string? factoryMethodName = null;
        string? programFileName = null;
        string? tensorDirectoryName = null;
        string? projectFileName = null;
        var generateProjectFile = true;
        var overwrite = false;

        for (var i = 3; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--project-name":
                    projectName = ReadOptionValue(args, ref i, "--project-name");
                    break;
                case "--namespace":
                    namespaceName = ReadOptionValue(args, ref i, "--namespace");
                    break;
                case "--package-version":
                    packageVersion = ReadOptionValue(args, ref i, "--package-version");
                    break;
                case "--program-class-name":
                    programClassName = ReadOptionValue(args, ref i, "--program-class-name");
                    break;
                case "--factory-method-name":
                    factoryMethodName = ReadOptionValue(args, ref i, "--factory-method-name");
                    break;
                case "--program-file-name":
                    programFileName = ReadOptionValue(args, ref i, "--program-file-name");
                    break;
                case "--tensor-directory-name":
                    tensorDirectoryName = ReadOptionValue(args, ref i, "--tensor-directory-name");
                    break;
                case "--project-file-name":
                    projectFileName = ReadOptionValue(args, ref i, "--project-file-name");
                    break;
                case "--no-project-file":
                    generateProjectFile = false;
                    break;
                case "--overwrite":
                    overwrite = true;
                    break;
                default:
                    return Fail(standardError, $"Unknown option '{args[i]}'.", WriteProjectHelp);
            }
        }

        var generator = new OnnxProjectGenerator();
        var result = generator.Generate(new ProjectGeneratorOptions
        {
            InputModelPath = inputModelPath,
            OutputDirectoryPath = outputDirectoryPath,
            ProjectName = projectName,
            Namespace = namespaceName,
            OnnxifyPackageVersion = packageVersion,
            ProgramClassName = programClassName ?? "Program",
            FactoryMethodName = factoryMethodName ?? "CreateModel",
            ProgramFileName = programFileName ?? "Program.cs",
            TensorDirectoryName = tensorDirectoryName ?? "Assets",
            ProjectFileName = projectFileName,
            GenerateProjectFile = generateProjectFile,
            Overwrite = overwrite,
        });

        standardOutput.WriteLine(FormatProjectGenerationResult(result));
        return 0;
    }

    private static string FormatInputsOutputs(OnnxModel model)
    {
        var producerName = string.IsNullOrWhiteSpace(model.ProducerName) ? "<unknown>" : model.ProducerName;
        var domain = string.IsNullOrWhiteSpace(model.Domain) ? "<default>" : model.Domain;
        var graphName = string.IsNullOrWhiteSpace(model.Graph.Name) ? "<unnamed>" : model.Graph.Name;

        return $"""
            OnnxModelInputsOutputs(
                Producer={producerName},
                Version={model.ProducerVersion},
                ModelVersion={model.ModelVersion},
                IrVersion={model.IrVersion},
                Domain={domain},
                GraphName={graphName},
                Inputs={FormatCollection(model.Graph.Inputs).Indent(1)},
                Outputs={FormatCollection(model.Graph.Outputs).Indent(1)}
            )
            """;
    }

    private static string FormatProjectGenerationResult(ProjectGenerationResult result)
    {
        return $"""
            ProjectGenerationResult(
                OutputDirectory={result.OutputDirectoryPath},
                ProgramFile={result.ProgramFilePath},
                ProjectFile={result.ProjectFilePath ?? "<none>"},
                TensorFiles={FormatCollection(result.TensorFilePaths).Indent(1)},
                Warnings={FormatCollection(result.Warnings).Indent(1)}
            )
            """;
    }

    private static string FormatHuggingFaceDownloadResult(HuggingFaceDownloadResult result)
    {
        return $"""
            HuggingFaceDownloadResult(
                Repository={result.RepositoryId},
                Revision={result.Revision},
                OutputDirectory={result.OutputDirectoryPath},
                Files={result.Files.Count},
                DownloadedFiles={result.DownloadedFileCount}
            )
            """;
    }

    private static string FormatCollection<T>(IEnumerable<T> values)
    {
        var items = values.Select(static x => x?.ToString() ?? string.Empty).ToArray();
        if (items.Length == 0)
        {
            return "[]";
        }

        return $"""
            [
                {string.Join(",\n", items).Indent(1)}
            ]
            """;
    }

    private static string ReadOptionValue(IReadOnlyList<string> args, ref int index, string optionName)
    {
        if (index + 1 >= args.Count)
        {
            throw new ArgumentException($"Option '{optionName}' requires a value.");
        }

        index++;
        return args[index];
    }

    private static bool ShouldIncludeHuggingFacePath(
        string path,
        IReadOnlyList<Func<string, bool>> includeMatchers,
        string? variant)
    {
        if (includeMatchers.Count > 0)
        {
            return includeMatchers.Any(match => match(path));
        }

        if (!string.IsNullOrWhiteSpace(variant) && !variant.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            return IsHuggingFaceSupportFile(path) || path.Contains(variant, StringComparison.OrdinalIgnoreCase);
        }

        return true;
    }

    private static bool IsHuggingFaceSupportFile(string path)
    {
        var fileName = Path.GetFileName(path);
        var extension = Path.GetExtension(path);

        return string.Equals(fileName, "README.md", StringComparison.OrdinalIgnoreCase)
            || string.Equals(extension, ".json", StringComparison.OrdinalIgnoreCase)
            || string.Equals(extension, ".txt", StringComparison.OrdinalIgnoreCase)
            || string.Equals(extension, ".yaml", StringComparison.OrdinalIgnoreCase)
            || string.Equals(extension, ".yml", StringComparison.OrdinalIgnoreCase)
            || string.Equals(extension, ".model", StringComparison.OrdinalIgnoreCase);
    }

    private static Func<string, bool> CreatePathMatcher(string pattern)
    {
        if (string.IsNullOrWhiteSpace(pattern))
        {
            throw new ArgumentException("Path filter pattern cannot be empty.");
        }

        var normalizedPattern = pattern.Replace('\\', '/');
        var regexPattern = "^" + Regex.Escape(normalizedPattern)
            .Replace("\\*", ".*", StringComparison.Ordinal)
            .Replace("\\?", ".", StringComparison.Ordinal) + "$";
        var regex = new Regex(regexPattern, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

        return path => regex.IsMatch(path.Replace('\\', '/'));
    }

    private static bool IsHelp(string value)
    {
        return value.Equals("--help", StringComparison.OrdinalIgnoreCase)
            || value.Equals("-h", StringComparison.OrdinalIgnoreCase)
            || value.Equals("help", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsVersion(string value)
    {
        return value.Equals("--version", StringComparison.OrdinalIgnoreCase);
    }

    private static string GetToolVersion()
    {
        var assembly = typeof(App).Assembly;
        var informationalVersion = assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion;

        if (!string.IsNullOrWhiteSpace(informationalVersion))
        {
            var plusIndex = informationalVersion.IndexOf('+');
            return plusIndex >= 0
                ? informationalVersion[..plusIndex]
                : informationalVersion;
        }

        return assembly.GetName().Version?.ToString() ?? "unknown";
    }

    private static int Fail(TextWriter standardError, string message, Action<TextWriter> writeHelp)
    {
        standardError.WriteLine(message);
        standardError.WriteLine();
        writeHelp(standardError);
        return 1;
    }

    private static void WriteHelp(TextWriter output)
    {
        output.WriteLine(
            """
            Onnxify CLI

            Usage:
              onnxify --version
              onnxify onnx show <model.onnx>
              onnxify onnx io <model.onnx>
              onnxify safetensors show <model.safetensors>
              onnxify project generate <model.onnx> <output-directory> [options]
              onnxify hf download <repository-id> <output-directory> [options]

            Run 'onnxify <command> --help' for command-specific help.
            """);
    }

    private static void WriteOnnxHelp(TextWriter output)
    {
        output.WriteLine(
            """
            ONNX commands

            Usage:
              onnxify onnx show <model.onnx>
              onnxify onnx io <model.onnx>
              onnxify onnx inputs-outputs <model.onnx>
            """);
    }

    private static void WriteSafetensorsHelp(TextWriter output)
    {
        output.WriteLine(
            """
            Safetensors commands

            Usage:
              onnxify safetensors show <model.safetensors>
            """);
    }

    private static void WriteProjectHelp(TextWriter output)
    {
        output.WriteLine(
            """
            Project generation commands

            Usage:
              onnxify project generate <model.onnx> <output-directory> [options]

            Options:
              --project-name <name>
              --namespace <name>
              --package-version <version>
              --program-class-name <name>
              --factory-method-name <name>
              --program-file-name <name>
              --tensor-directory-name <name>
              --project-file-name <name>
              --no-project-file
              --overwrite
            """);
    }

    private static void WriteHuggingFaceHelp(TextWriter output)
    {
        output.WriteLine(
            """
            Hugging Face commands

            Usage:
              onnxify hf download <repository-id> <output-directory> [options]
              onnxify huggingface download <repository-id> <output-directory> [options]

            Options:
              --revision <revision>       Defaults to main.
              --token <token>             Hugging Face access token.
              --token-env <name>          Token environment variable. Defaults to HF_TOKEN.
              --include <pattern>         Wildcard path include filter. Can be repeated.
              --exclude <pattern>         Wildcard path exclude filter. Can be repeated.
              --variant <name>            Include support files and paths containing this value, for example bf16.
              --overwrite                 Replace existing files.
              --quiet                     Do not print per-file progress.

            Pattern examples:
              --include "*bf16*"
              --include "*.json" --include "*.model" --exclude "*.md5"
            """);
    }
}

internal static class CliTextExtensions
{
    public static string Indent(this string text, int tabs)
    {
        var indent = new string(' ', tabs * 4);
        return text.Trim().Replace("\n", $"\n{indent}", StringComparison.Ordinal).Trim();
    }
}
