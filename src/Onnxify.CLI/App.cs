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

        switch (args[0].ToLowerInvariant())
        {
            case "show":
                return RunOnnxShow(args[1..], standardOutput, standardError);
            case "diff":
                return RunOnnxDiff(args[1..], standardOutput, standardError);
            case "io":
            case "inputs-outputs":
                if (args.Length != 2)
                {
                    return Fail(standardError, "The onnx inputs-outputs command expects a model path.", WriteOnnxHelp);
                }

                var model = OnnxModel.FromFile(args[1]);
                standardOutput.WriteLine(FormatInputsOutputs(model));
                return 0;
            default:
                return Fail(standardError, $"Unknown onnx subcommand '{args[0]}'.", WriteOnnxHelp);
        }
    }

    private static int RunOnnxShow(string[] args, TextWriter standardOutput, TextWriter standardError)
    {
        if (args.Length == 0)
        {
            return Fail(standardError, "The onnx show command expects a model path.", WriteOnnxHelp);
        }

        var options = OnnxShowOptions.None;
        string? modelPath = null;

        foreach (var arg in args)
        {
            switch (arg.ToLowerInvariant())
            {
                case "--inputs":
                    options |= OnnxShowOptions.Inputs;
                    break;
                case "--outputs":
                    options |= OnnxShowOptions.Outputs;
                    break;
                case "--values":
                    options |= OnnxShowOptions.Values;
                    break;
                case "--nodes":
                    options |= OnnxShowOptions.Nodes;
                    break;
                default:
                    if (arg.StartsWith("-", StringComparison.Ordinal))
                    {
                        return Fail(standardError, $"Unknown onnx show option '{arg}'.", WriteOnnxHelp);
                    }

                    if (modelPath is not null)
                    {
                        return Fail(standardError, "The onnx show command expects one model path.", WriteOnnxHelp);
                    }

                    modelPath = arg;
                    break;
            }
        }

        if (modelPath is null)
        {
            return Fail(standardError, "The onnx show command expects a model path.", WriteOnnxHelp);
        }

        var model = OnnxModel.FromFile(modelPath);
        standardOutput.WriteLine(options == OnnxShowOptions.None
            ? model.ToString()
            : FormatOnnxShow(model, options));

        return 0;
    }

    private static int RunOnnxDiff(string[] args, TextWriter standardOutput, TextWriter standardError)
    {
        if (args.Length == 0)
        {
            return Fail(standardError, "The onnx diff command expects two model paths.", WriteOnnxHelp);
        }

        var options = OnnxShowOptions.None;
        var modelPaths = new List<string>();

        foreach (var arg in args)
        {
            switch (arg.ToLowerInvariant())
            {
                case "--inputs":
                    options |= OnnxShowOptions.Inputs;
                    break;
                case "--outputs":
                    options |= OnnxShowOptions.Outputs;
                    break;
                case "--values":
                    options |= OnnxShowOptions.Values;
                    break;
                case "--nodes":
                    options |= OnnxShowOptions.Nodes;
                    break;
                default:
                    if (arg.StartsWith("-", StringComparison.Ordinal))
                    {
                        return Fail(standardError, $"Unknown onnx diff option '{arg}'.", WriteOnnxHelp);
                    }

                    modelPaths.Add(arg);
                    break;
            }
        }

        if (modelPaths.Count != 2)
        {
            return Fail(standardError, "The onnx diff command expects two model paths.", WriteOnnxHelp);
        }

        var left = OnnxModel.FromFile(modelPaths[0]);
        var right = OnnxModel.FromFile(modelPaths[1]);

        standardOutput.WriteLine(FormatOnnxDiff(left, right, modelPaths[0], modelPaths[1], options));
        return 0;
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

    private static string FormatOnnxShow(OnnxModel model, OnnxShowOptions options)
    {
        var sections = new List<string>
        {
            FormatOnnxSummary(model),
        };

        if (options.HasFlag(OnnxShowOptions.Inputs))
        {
            sections.Add(FormatNamedSection("Inputs", model.Graph.Inputs.Select(FormatValue)));
        }

        if (options.HasFlag(OnnxShowOptions.Outputs))
        {
            sections.Add(FormatNamedSection("Outputs", model.Graph.Outputs.Select(FormatValue)));
        }

        if (options.HasFlag(OnnxShowOptions.Values))
        {
            sections.Add(FormatNamedSection("Initializers", model.Graph.Initializers.Select(x => FormatTensor(x, includeValues: true))));
            sections.Add(FormatNamedSection("IntermediateValues", model.Graph.IntermediateValues.Select(FormatValue)));
        }

        if (options.HasFlag(OnnxShowOptions.Nodes))
        {
            sections.Add(FormatNamedSection("Nodes", model.Graph.Nodes.Select((node, index) => FormatNode(index, node))));
        }

        return string.Join("\n", sections);
    }

    private static string FormatOnnxDiff(
        OnnxModel left,
        OnnxModel right,
        string leftPath,
        string rightPath,
        OnnxShowOptions options
    )
    {
        var includeInputs = options == OnnxShowOptions.None || options.HasFlag(OnnxShowOptions.Inputs);
        var includeOutputs = options == OnnxShowOptions.None || options.HasFlag(OnnxShowOptions.Outputs);
        var includeValues = options == OnnxShowOptions.None || options.HasFlag(OnnxShowOptions.Values);
        var includeNodes = options == OnnxShowOptions.None || options.HasFlag(OnnxShowOptions.Nodes);
        var lines = new List<string>
        {
            "OnnxModelDiff(",
            $"    Left={leftPath},",
            $"    Right={rightPath},",
            "    Metadata=[",
        };

        AddDiff(lines, "ProducerName", FormatScalar(left.ProducerName), FormatScalar(right.ProducerName), indent: 2);
        AddDiff(lines, "ProducerVersion", FormatScalar(left.ProducerVersion), FormatScalar(right.ProducerVersion), indent: 2);
        AddDiff(lines, "ModelVersion", left.ModelVersion.ToString(), right.ModelVersion.ToString(), indent: 2);
        AddDiff(lines, "IrVersion", left.IrVersion.ToString(), right.IrVersion.ToString(), indent: 2);
        AddDiff(lines, "Domain", FormatDomain(left.Domain), FormatDomain(right.Domain), indent: 2);
        AddDiff(lines, "GraphName", FormatScalar(left.Graph.Name), FormatScalar(right.Graph.Name), indent: 2);
        AddDiff(lines, "Inputs", left.Graph.Inputs.Count.ToString(), right.Graph.Inputs.Count.ToString(), indent: 2);
        AddDiff(lines, "Outputs", left.Graph.Outputs.Count.ToString(), right.Graph.Outputs.Count.ToString(), indent: 2);
        AddDiff(lines, "Initializers", left.Graph.Initializers.Count.ToString(), right.Graph.Initializers.Count.ToString(), indent: 2);
        AddDiff(lines, "IntermediateValues", left.Graph.IntermediateValues.Count.ToString(), right.Graph.IntermediateValues.Count.ToString(), indent: 2);
        AddDiff(lines, "Nodes", left.Graph.Nodes.Count.ToString(), right.Graph.Nodes.Count.ToString(), indent: 2);
        lines.Add("    ],");

        lines.Add("    OpCounts=[");
        foreach (var diff in CompareMaps(GetOpCounts(left), GetOpCounts(right)))
        {
            lines.Add($"        {diff},");
        }
        lines.Add("    ],");

        if (includeInputs)
        {
            lines.Add("    Inputs=[");
            foreach (var diff in CompareOrdered(
                left.Graph.Inputs.Select(FormatValue).ToArray(),
                right.Graph.Inputs.Select(FormatValue).ToArray()))
            {
                lines.Add($"        {diff},");
            }
            lines.Add("    ],");
        }

        if (includeOutputs)
        {
            lines.Add("    Outputs=[");
            foreach (var diff in CompareOrdered(
                left.Graph.Outputs.Select(FormatValue).ToArray(),
                right.Graph.Outputs.Select(FormatValue).ToArray()))
            {
                lines.Add($"        {diff},");
            }
            lines.Add("    ],");
        }

        if (includeValues)
        {
            lines.Add("    Initializers=[");
            foreach (var diff in CompareOrdered(
                left.Graph.Initializers.Select(x => FormatTensor(x, includeValues: false)).ToArray(),
                right.Graph.Initializers.Select(x => FormatTensor(x, includeValues: false)).ToArray()))
            {
                lines.Add($"        {diff},");
            }
            lines.Add("    ],");

            lines.Add("    IntermediateValues=[");
            foreach (var diff in CompareOrdered(
                left.Graph.IntermediateValues.Select(FormatValue).ToArray(),
                right.Graph.IntermediateValues.Select(FormatValue).ToArray()))
            {
                lines.Add($"        {diff},");
            }
            lines.Add("    ],");
        }

        if (includeNodes)
        {
            lines.Add("    Nodes=[");
            foreach (var diff in CompareOrdered(
                left.Graph.Nodes.Select(FormatNodeSignature).ToArray(),
                right.Graph.Nodes.Select(FormatNodeSignature).ToArray()))
            {
                lines.Add($"        {diff},");
            }
            lines.Add("    ]");
        }

        lines.Add(")");

        return string.Join("\n", lines);
    }

    private static string FormatOnnxSummary(OnnxModel model)
    {
        return $"""
            OnnxModelSummary(
                Producer={FormatScalar(model.ProducerName)},
                Version={FormatScalar(model.ProducerVersion)},
                ModelVersion={model.ModelVersion},
                IrVersion={model.IrVersion},
                Domain={FormatDomain(model.Domain)},
                GraphName={FormatScalar(model.Graph.Name)},
                OpsetImports={FormatCollection(model.OpsetImport.Select(FormatOpsetImport)).Indent(1)},
                Inputs={model.Graph.Inputs.Count},
                Outputs={model.Graph.Outputs.Count},
                Initializers={model.Graph.Initializers.Count},
                IntermediateValues={model.Graph.IntermediateValues.Count},
                Nodes={model.Graph.Nodes.Count},
                OpCounts={FormatCollection(GetOpCounts(model).Select(x => $"{x.Key}={x.Value}")).Indent(1)}
            )
            """;
    }

    private static string FormatNamedSection(string name, IEnumerable<string> values)
    {
        return $"""
            {name}={FormatCollection(values).Indent(1)}
            """;
    }

    private static string FormatNode(int index, OnnxNode node)
    {
        return $"{index}: {FormatNodeSignature(node)}";
    }

    private static string FormatNodeSignature(OnnxNode node)
    {
        var domain = FormatDomain(node.Domain);
        var inputs = string.Join(", ", node.Inputs.Select(FormatEdgeReference));
        var outputs = string.Join(", ", node.Outputs.Select(FormatEdgeReference));
        var attributes = string.Join(", ", node.Attributes
            .OrderBy(x => x.Name, StringComparer.Ordinal)
            .Select(FormatAttribute));

        return $"{node.Name}: {domain}:{node.OpType}({inputs}) -> [{outputs}] attrs=[{attributes}]";
    }

    private static string FormatAttribute(OnnxAttribute attribute)
    {
        return $"{attribute.Name}={FormatObject(attribute.GetValue())}";
    }

    private static string FormatValue(OnnxValue value)
    {
        return $"{value.Name}: {value.Type}";
    }

    private static string FormatTensor(OnnxTensor tensor, bool includeValues)
    {
        var result = $"{tensor.Name}: {tensor.DataType.Name}[{string.Join(", ", tensor.Shape)}]";

        if (!includeValues)
        {
            return result;
        }

        return tensor switch
        {
            OnnxTensor<float> value => $"{result} = {FormatPreview(value.Value)}",
            OnnxTensor<double> value => $"{result} = {FormatPreview(value.Value)}",
            OnnxTensor<long> value => $"{result} = {FormatPreview(value.Value)}",
            OnnxTensor<int> value => $"{result} = {FormatPreview(value.Value)}",
            OnnxTensor<short> value => $"{result} = {FormatPreview(value.Value)}",
            OnnxTensor<byte> value => $"{result} = {FormatPreview(value.Value)}",
            OnnxTensor<sbyte> value => $"{result} = {FormatPreview(value.Value)}",
            OnnxTensor<bool> value => $"{result} = {FormatPreview(value.Value)}",
            OnnxTensor<string> value => $"{result} = {FormatPreview(value.Value)}",
            _ => result,
        };
    }

    private static string FormatPreview<T>(IEnumerable<T> values)
    {
        const int PREVIEW_COUNT = 8;

        var materialized = values.Take(PREVIEW_COUNT + 1).ToArray();
        var preview = materialized.Take(PREVIEW_COUNT).Select(static x => x?.ToString() ?? "null");
        var suffix = materialized.Length > PREVIEW_COUNT ? ", ..." : string.Empty;

        return $"[{string.Join(", ", preview)}{suffix}]";
    }

    private static string FormatEdgeReference(IOnnxGraphEdge edge)
    {
        return edge switch
        {
            OnnxValue value => FormatValue(value),
            OnnxTensor tensor => FormatTensor(tensor, includeValues: false),
            _ => edge.Name,
        };
    }

    private static string FormatObject(object? value)
    {
        return value switch
        {
            null => "<null>",
            string text => text,
            IEnumerable<long> values => $"[{string.Join(", ", values)}]",
            IEnumerable<int> values => $"[{string.Join(", ", values)}]",
            IEnumerable<float> values => $"[{string.Join(", ", values)}]",
            IEnumerable<double> values => $"[{string.Join(", ", values)}]",
            IEnumerable<string> values => $"[{string.Join(", ", values)}]",
            System.Collections.IEnumerable values when value is not string => $"[{string.Join(", ", values.Cast<object?>())}]",
            _ => value.ToString() ?? string.Empty,
        };
    }

    private static string FormatScalar(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? "<empty>" : value;
    }

    private static string FormatDomain(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? "<default>" : value;
    }

    private static string FormatOpsetImport(OperationSet opset)
    {
        return $"{FormatDomain(opset.Domain)}={opset.Version}";
    }

    private static Dictionary<string, int> GetOpCounts(OnnxModel model)
    {
        return model.Graph.Nodes
            .GroupBy(x => string.IsNullOrWhiteSpace(x.Domain) ? x.OpType : $"{x.Domain}:{x.OpType}")
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .ToDictionary(x => x.Key, x => x.Count(), StringComparer.Ordinal);
    }

    private static IEnumerable<string> CompareMaps(
        IReadOnlyDictionary<string, int> left,
        IReadOnlyDictionary<string, int> right
    )
    {
        var keys = left.Keys.Concat(right.Keys).Distinct(StringComparer.Ordinal).OrderBy(x => x, StringComparer.Ordinal);

        foreach (var key in keys)
        {
            left.TryGetValue(key, out var leftValue);
            right.TryGetValue(key, out var rightValue);

            if (leftValue == rightValue)
            {
                yield return $"same {key}: {leftValue}";
            }
            else
            {
                yield return $"diff {key}: left={leftValue}, right={rightValue}";
            }
        }
    }

    private static IEnumerable<string> CompareOrdered(string[] left, string[] right)
    {
        var count = Math.Max(left.Length, right.Length);

        for (var i = 0; i < count; i++)
        {
            var leftValue = i < left.Length ? left[i] : "<missing>";
            var rightValue = i < right.Length ? right[i] : "<missing>";

            if (StringComparer.Ordinal.Equals(leftValue, rightValue))
            {
                yield return $"same [{i}]: {leftValue}";
            }
            else
            {
                yield return $"diff [{i}]: left={leftValue}; right={rightValue}";
            }
        }
    }

    private static void AddDiff(List<string> lines, string name, string left, string right, int indent)
    {
        var prefix = new string(' ', indent * 4);
        var status = StringComparer.Ordinal.Equals(left, right) ? "same" : "diff";
        lines.Add($"{prefix}{status} {name}: left={left}, right={right},");
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
              onnxify onnx show [options] <model.onnx>
              onnxify onnx diff [options] <left.onnx> <right.onnx>
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
              onnxify onnx show [options] <model.onnx>
              onnxify onnx diff [options] <left.onnx> <right.onnx>
              onnxify onnx io <model.onnx>
              onnxify onnx inputs-outputs <model.onnx>

            Show options:
              --inputs     Include graph inputs.
              --outputs    Include graph outputs.
              --values     Include initializer previews and intermediate value-info entries.
              --nodes      Include compact node signatures.

            Diff options:
              --inputs     Include graph input differences.
              --outputs    Include graph output differences.
              --values     Include initializer and intermediate value differences.
              --nodes      Include compact node signature differences.
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

[Flags]
internal enum OnnxShowOptions
{
    None = 0,
    Inputs = 1,
    Outputs = 2,
    Values = 4,
    Nodes = 8,
}

internal static class CliTextExtensions
{
    public static string Indent(this string text, int tabs)
    {
        var indent = new string(' ', tabs * 4);
        return text.Trim().Replace("\n", $"\n{indent}", StringComparison.Ordinal).Trim();
    }
}
