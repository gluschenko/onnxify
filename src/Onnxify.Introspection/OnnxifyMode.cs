using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Onnxify.Introspection;

internal static class OnnxifyMode
{
    public static int Run(string[] args)
    {
        var options = OnnxifyOptions.Parse(args);
        var schemaPath = ResolveSchemaPath(options.SchemaPath);
        var root = LoadSchema(schemaPath);
        var markdown = OnnxifyMarkdownRenderer.Render(root);

        var outputPath = Path.GetFullPath(options.OutputPath);
        var outputDirectory = Path.GetDirectoryName(outputPath);

        if (!string.IsNullOrWhiteSpace(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        File.WriteAllText(outputPath, markdown, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        var operatorCount = root.Operators
            .GroupBy(op => $"{op.Domain ?? string.Empty}::{op.Name}", StringComparer.Ordinal)
            .Count();

        Console.WriteLine($"Saved Markdown to: {outputPath}");
        Console.WriteLine($"Schema: {schemaPath}");
        Console.WriteLine($"Operator entries: {root.Operators.Count}");
        Console.WriteLine($"Rendered operators: {operatorCount}");
        return 0;
    }

    private static string ResolveSchemaPath(string? schemaPath)
    {
        if (!string.IsNullOrWhiteSpace(schemaPath))
        {
            return Path.GetFullPath(schemaPath);
        }

        var current = new DirectoryInfo(AppContext.BaseDirectory);

        while (current is not null)
        {
            var candidate = Path.Combine(current.FullName, "src", "Onnxify", "Assets", "onnx_operators.json");

            if (File.Exists(candidate))
            {
                return candidate;
            }

            current = current.Parent;
        }

        throw new FileNotFoundException("Could not find src\\Onnxify\\Assets\\onnx_operators.json. Pass it explicitly via --schema.");
    }

    private static OnnxifyOperatorSchemaRoot LoadSchema(string schemaPath)
    {
        var json = File.ReadAllText(schemaPath);
        var root = JsonSerializer.Deserialize<OnnxifyOperatorSchemaRoot>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        });

        return root ?? throw new InvalidOperationException($"Failed to deserialize operator schema from '{schemaPath}'.");
    }
}

internal sealed record OnnxifyOptions(string OutputPath, string? SchemaPath)
{
    public static OnnxifyOptions Parse(string[] args)
    {
        string outputPath = Path.Combine(Environment.CurrentDirectory, "artifacts", "onnxify-operators.md");
        string? schemaPath = null;

        for (var index = 0; index < args.Length; index++)
        {
            var arg = args[index];

            switch (arg)
            {
                case "--output":
                case "-o":
                    outputPath = RequireValue(args, ref index, arg);
                    break;
                case "--schema":
                case "-s":
                    schemaPath = RequireValue(args, ref index, arg);
                    break;
                case "--help":
                case "-h":
                    PrintHelp();
                    Environment.Exit(0);
                    break;
                default:
                    throw new ArgumentException($"Unknown argument '{arg}'. Use --help to see supported options.");
            }
        }

        return new OnnxifyOptions(outputPath, schemaPath);
    }

    private static string RequireValue(string[] args, ref int index, string optionName)
    {
        if (index + 1 >= args.Length)
        {
            throw new ArgumentException($"Option '{optionName}' requires a value.");
        }

        index++;
        return args[index];
    }

    private static void PrintHelp()
    {
        Console.WriteLine(
            """
            Onnxify operator markdown generator

            Options:
              -o, --output <path>   Markdown output path. Default: ./artifacts/onnxify-operators.md
              -s, --schema <path>   Optional path to src/Onnxify/Assets/onnx_operators.json
              -h, --help            Show this help
            """
        );
    }
}

internal static class OnnxifyMarkdownRenderer
{
    public static string Render(OnnxifyOperatorSchemaRoot root)
    {
        var builder = new StringBuilder();

        var operators = root.Operators
            .GroupBy(op => $"{op.Domain ?? string.Empty}::{op.Name}", StringComparer.Ordinal)
            .Select(group => group.OrderByDescending(op => op.SinceVersion).First())
            .OrderBy(op => op.Domain, StringComparer.Ordinal)
            .ThenBy(op => op.Name, StringComparer.Ordinal)
            .ToArray();

        for (var index = 0; index < operators.Length; index++)
        {
            var op = operators[index];
            builder.AppendLine(GetTitle(op));
            builder.AppendLine(Normalize(op.Doc));
            builder.AppendLine();
            AppendSection(builder, "Inputs", op.Inputs.Select(OnnxifyParameterDescriptor.FromInput).ToArray());
            builder.AppendLine();
            AppendSection(builder, "Outputs", op.Outputs.Select(OnnxifyParameterDescriptor.FromOutput).ToArray());
            builder.AppendLine();
            AppendSection(builder, "Attributes", op.Attributes.Select(OnnxifyParameterDescriptor.FromAttribute).ToArray());

            if (index < operators.Length - 1)
            {
                builder.AppendLine();
            }
        }

        return builder.ToString();
    }

    private static string GetTitle(OnnxifyOperatorSchema op)
    {
        return string.IsNullOrWhiteSpace(op.Domain)
            ? $"# {op.Name}"
            : $"# {op.Name} [{op.Domain}]";
    }

    private static void AppendSection(StringBuilder builder, string title, IReadOnlyList<OnnxifyParameterDescriptor> parameters)
    {
        builder.AppendLine($"## {title}");

        if (parameters.Count == 0)
        {
            builder.AppendLine("_None._");
            return;
        }

        foreach (var parameter in parameters)
        {
            builder.AppendLine($"* `{parameter.Name}` ({parameter.TypeDisplay}) - {parameter.Description}");
        }
    }

    private static string Normalize(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return "No description.";
        }

        var lines = text
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(line => !string.IsNullOrWhiteSpace(line));

        var normalized = string.Join(" ", lines);
        return string.IsNullOrWhiteSpace(normalized) ? "No description." : normalized;
    }
}

internal sealed record OnnxifyParameterDescriptor(string Name, string TypeDisplay, string Description)
{
    public static OnnxifyParameterDescriptor FromInput(OnnxifyOperatorParameter parameter)
    {
        return new OnnxifyParameterDescriptor(
            parameter.Name,
            FormatParameterType(parameter.Type, parameter.Option, parameter.MinArity),
            Normalize(parameter.Description)
        );
    }

    public static OnnxifyParameterDescriptor FromOutput(OnnxifyOperatorParameter parameter)
    {
        return new OnnxifyParameterDescriptor(
            parameter.Name,
            FormatParameterType(parameter.Type, parameter.Option, parameter.MinArity),
            Normalize(parameter.Description)
        );
    }

    public static OnnxifyParameterDescriptor FromAttribute(OnnxifyOperatorAttribute attribute)
    {
        var parts = new List<string> { FormatAttributeType(attribute.Type) };
        parts.Add(attribute.Required ? "required" : "optional");

        if (attribute.Default is not null)
        {
            parts.Add($"default: {FormatDefaultValue(attribute.Default)}");
        }

        return new OnnxifyParameterDescriptor(
            attribute.Name,
            string.Join(", ", parts),
            Normalize(attribute.Description)
        );
    }

    private static string Normalize(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return "No description.";
        }

        var lines = text
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(line => !string.IsNullOrWhiteSpace(line));

        var normalized = string.Join(" ", lines);
        return string.IsNullOrWhiteSpace(normalized) ? "No description." : normalized;
    }

    private static string FormatParameterType(string type, OnnxifyFormalParameterOption option, int minArity)
    {
        var parts = new List<string> { type };

        switch (option)
        {
            case OnnxifyFormalParameterOption.Optional:
                parts.Add("optional");
                break;
            case OnnxifyFormalParameterOption.Variadic:
                parts.Add($"variadic, min arity: {minArity}");
                break;
        }

        return string.Join(", ", parts);
    }

    private static string FormatAttributeType(int type)
    {
        return type switch
        {
            1 => "float",
            2 => "int",
            3 => "string",
            4 => "tensor",
            5 => "graph",
            6 => "float[]",
            7 => "int[]",
            8 => "string[]",
            9 => "tensor[]",
            10 => "graph[]",
            11 => "sparse_tensor",
            12 => "sparse_tensor[]",
            13 => "type_proto",
            14 => "type_proto[]",
            _ => "undefined",
        };
    }

    private static string FormatDefaultValue(object value)
    {
        return value switch
        {
            JsonElement element => element.ValueKind switch
            {
                JsonValueKind.String => element.GetString() ?? string.Empty,
                JsonValueKind.Number => element.ToString(),
                JsonValueKind.True => "true",
                JsonValueKind.False => "false",
                JsonValueKind.Array => $"[{string.Join(", ", element.EnumerateArray().Select(item => FormatDefaultValue(item)))}]",
                JsonValueKind.Null => "null",
                _ => element.ToString(),
            },
            _ => value.ToString() ?? string.Empty,
        };
    }
}

internal sealed class OnnxifyOperatorSchemaRoot
{
    [JsonPropertyName("operators")]
    public required List<OnnxifyOperatorSchema> Operators { get; set; }
}

internal sealed class OnnxifyOperatorSchema
{
    [JsonPropertyName("name")]
    public required string Name { get; set; }

    [JsonPropertyName("domain")]
    public required string Domain { get; set; }

    [JsonPropertyName("sinceVersion")]
    public required int SinceVersion { get; set; }

    [JsonPropertyName("doc")]
    public required string? Doc { get; set; }

    [JsonPropertyName("attributes")]
    public required List<OnnxifyOperatorAttribute> Attributes { get; set; }

    [JsonPropertyName("inputs")]
    public required List<OnnxifyOperatorParameter> Inputs { get; set; }

    [JsonPropertyName("outputs")]
    public required List<OnnxifyOperatorParameter> Outputs { get; set; }
}

internal sealed class OnnxifyOperatorAttribute
{
    [JsonPropertyName("name")]
    public required string Name { get; set; }

    [JsonPropertyName("description")]
    public string? Description { get; set; }

    [JsonPropertyName("required")]
    public required bool Required { get; set; }

    [JsonPropertyName("type")]
    public required int Type { get; set; }

    [JsonPropertyName("default")]
    public object? Default { get; set; }
}

internal sealed class OnnxifyOperatorParameter
{
    [JsonPropertyName("name")]
    public required string Name { get; set; }

    [JsonPropertyName("description")]
    public string? Description { get; set; }

    [JsonPropertyName("option")]
    public required OnnxifyFormalParameterOption Option { get; set; }

    [JsonPropertyName("minArity")]
    public required int MinArity { get; set; }

    [JsonPropertyName("type")]
    public required string Type { get; set; }
}

internal enum OnnxifyFormalParameterOption : byte
{
    Single = 0,
    Optional = 1,
    Variadic = 2,
}
