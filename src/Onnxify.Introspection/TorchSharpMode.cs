using System.Reflection;
using System.Text;
using System.Xml.Linq;
using TorchSharp;

namespace Onnxify.Introspection;

internal static class TorchSharpMode
{
    public static int Run(string[] args)
    {
        var options = TorchSharpOptions.Parse(args);
        var assembly = typeof(torch).Assembly;
        var xmlDocumentation = TorchXmlDocumentationIndex.TryLoad(options.XmlPath ?? TryFindAdjacentXml(assembly.Location));
        var catalog = TorchSharpCatalog.Create(assembly, xmlDocumentation);
        var markdown = TorchSharpMarkdownRenderer.Render(catalog);

        var outputPath = Path.GetFullPath(options.OutputPath);
        var outputDirectory = Path.GetDirectoryName(outputPath);

        if (!string.IsNullOrWhiteSpace(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        File.WriteAllText(outputPath, markdown, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        Console.WriteLine($"Saved Markdown to: {outputPath}");
        Console.WriteLine($"Operators: {catalog.Operators.Count}");
        Console.WriteLine($"Source methods: {catalog.MethodCount}");
        Console.WriteLine($"XML docs: {(xmlDocumentation is null ? "not found" : xmlDocumentation.SourcePath)}");
        return 0;
    }

    private static string? TryFindAdjacentXml(string assemblyPath)
    {
        var xmlPath = Path.ChangeExtension(assemblyPath, ".xml");
        if (File.Exists(xmlPath))
        {
            return xmlPath;
        }

        return TryFindNuGetXmlFallback();
    }

    private static string? TryFindNuGetXmlFallback()
    {
        var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

        if (string.IsNullOrWhiteSpace(userProfile))
        {
            return null;
        }

        var packageRoot = Path.Combine(userProfile, ".nuget", "packages", "torchsharp");

        if (!Directory.Exists(packageRoot))
        {
            return null;
        }

        var candidates = Directory.GetDirectories(packageRoot)
            .Select(path => new
            {
                Path = Path.Combine(path, "lib", "net6.0", "TorchSharp.xml"),
                VersionText = Path.GetFileName(path),
            })
            .Where(entry => File.Exists(entry.Path))
            .Select(entry =>
            {
                var parsed = Version.TryParse(entry.VersionText, out var version);
                return new
                {
                    entry.Path,
                    Version = parsed ? version : new Version(0, 0),
                    Parsed = parsed,
                };
            })
            .OrderByDescending(entry => entry.Parsed)
            .ThenByDescending(entry => entry.Version)
            .ToArray();

        return candidates.FirstOrDefault()?.Path;
    }
}

internal sealed record TorchSharpOptions(string OutputPath, string? XmlPath)
{
    public static TorchSharpOptions Parse(string[] args)
    {
        string outputPath = Path.Combine(Environment.CurrentDirectory, "artifacts", "torchsharp-operators.md");
        string? xmlPath = null;

        for (var index = 0; index < args.Length; index++)
        {
            var arg = args[index];

            switch (arg)
            {
                case "--output":
                case "-o":
                    outputPath = RequireValue(args, ref index, arg);
                    break;
                case "--xml":
                    xmlPath = RequireValue(args, ref index, arg);
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

        return new TorchSharpOptions(outputPath, xmlPath);
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
            TorchSharp operator markdown generator

            Options:
              -o, --output <path>   Markdown output path. Default: ./artifacts/torchsharp-operators.md
                  --xml <path>      Optional XML documentation file for TorchSharp.dll
              -h, --help            Show this help
            """
        );
    }
}

internal static class TorchSharpCatalog
{
    public static TorchCatalog Create(Assembly assembly, TorchXmlDocumentationIndex? xmlDocumentation)
    {
        var methods = assembly.GetExportedTypes()
            .Where(IsTorchOperatorContainer)
            .SelectMany(GetDeclaredPublicStaticMethods)
            .Where(method => !method.IsSpecialName)
            .Where(IsOperatorMethod)
            .ToArray();

        var operators = methods
            .GroupBy(method => method.Name, StringComparer.Ordinal)
            .Select(group => TorchOperatorDescriptor.Create(group.Key, group.ToArray(), xmlDocumentation))
            .OrderBy(op => op.Name, StringComparer.Ordinal)
            .ToArray();

        return new TorchCatalog(operators, methods.Length);
    }

    private static IEnumerable<MethodInfo> GetDeclaredPublicStaticMethods(Type type)
    {
        return type.GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly);
    }

    private static bool IsTorchOperatorContainer(Type type)
    {
        if (type == typeof(torch))
        {
            return true;
        }

        for (var current = type.DeclaringType; current is not null; current = current.DeclaringType)
        {
            if (current == typeof(torch))
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsOperatorMethod(MethodInfo method)
    {
        return TorchOperatorDescriptor.IsTensorLikeReturnType(method.ReturnType);
    }
}

internal sealed record TorchCatalog(IReadOnlyList<TorchOperatorDescriptor> Operators, int MethodCount);

internal sealed record TorchOperatorDescriptor(
    string Name,
    string Description,
    IReadOnlyList<TorchFieldDescriptor> Inputs,
    IReadOnlyList<TorchFieldDescriptor> Outputs,
    IReadOnlyList<TorchFieldDescriptor> Attributes)
{
    public static TorchOperatorDescriptor Create(string name, IReadOnlyList<MethodInfo> methods, TorchXmlDocumentationIndex? xmlDocumentation)
    {
        var description = methods
            .Select(method => xmlDocumentation?.GetSummary(method))
            .FirstOrDefault(summary => !string.IsNullOrWhiteSpace(summary))
            ?? "No description.";

        var inputs = MergeFields(
            methods.SelectMany(method => method.GetParameters())
                .Where(IsInputParameter)
                .Select(parameter => TorchFieldDescriptor.FromInput(parameter, xmlDocumentation))
        );

        var outputs = MergeFields(
            methods.Select(method => TorchFieldDescriptor.FromOutput(method, xmlDocumentation))
        );

        var attributes = MergeFields(
            methods.SelectMany(method => method.GetParameters())
                .Where(parameter => !IsInputParameter(parameter))
                .Select(parameter => TorchFieldDescriptor.FromAttribute(parameter, xmlDocumentation))
        );

        return new TorchOperatorDescriptor(name, description, inputs, outputs, attributes);
    }

    private static IReadOnlyList<TorchFieldDescriptor> MergeFields(IEnumerable<TorchFieldDescriptor> fields)
    {
        return fields
            .GroupBy(field => field.Name, StringComparer.Ordinal)
            .Select(group =>
            {
                var types = group.Select(field => field.TypeDisplay).Distinct(StringComparer.Ordinal).OrderBy(x => x, StringComparer.Ordinal).ToArray();
                var descriptions = group.Select(field => field.Description).Where(x => !string.IsNullOrWhiteSpace(x)).Distinct(StringComparer.Ordinal).ToArray();

                return new TorchFieldDescriptor(
                    group.Key,
                    string.Join(" | ", types),
                    descriptions.Length == 0 ? "No description." : descriptions[0]
                );
            })
            .OrderBy(field => field.Name, StringComparer.Ordinal)
            .ToArray();
    }

    private static bool IsInputParameter(ParameterInfo parameter)
    {
        return IsTensorLikeParameter(parameter.ParameterType);
    }

    internal static bool IsTensorLikeParameter(Type type)
    {
        if (type.IsArray)
        {
            return IsTensorLikeParameter(type.GetElementType()!);
        }

        if (Nullable.GetUnderlyingType(type) is Type underlyingType)
        {
            return IsTensorLikeParameter(underlyingType);
        }

        var fullName = type.FullName ?? type.Name;
        return fullName.Contains("Tensor", StringComparison.Ordinal)
            || (type.IsGenericType && type.GetGenericArguments().Any(IsTensorLikeParameter));
    }

    internal static bool IsTensorLikeReturnType(Type type)
    {
        return IsTensorLikeParameter(type);
    }
}

internal sealed record TorchFieldDescriptor(string Name, string TypeDisplay, string Description)
{
    public static TorchFieldDescriptor FromInput(ParameterInfo parameter, TorchXmlDocumentationIndex? xmlDocumentation)
    {
        return new TorchFieldDescriptor(
            parameter.Name ?? "input",
            TorchSignatureFormatter.FormatParameterType(parameter),
            xmlDocumentation?.GetParameterComment(parameter.Member as MethodInfo, parameter.Name) ?? "No description."
        );
    }

    public static TorchFieldDescriptor FromAttribute(ParameterInfo parameter, TorchXmlDocumentationIndex? xmlDocumentation)
    {
        var typeDisplay = TorchSignatureFormatter.FormatParameterType(parameter);

        if (parameter.IsOptional)
        {
            typeDisplay += $", optional, default: {TorchSignatureFormatter.FormatConstant(parameter.DefaultValue)}";
        }

        return new TorchFieldDescriptor(
            parameter.Name ?? "attribute",
            typeDisplay,
            xmlDocumentation?.GetParameterComment(parameter.Member as MethodInfo, parameter.Name) ?? "No description."
        );
    }

    public static TorchFieldDescriptor FromOutput(MethodInfo method, TorchXmlDocumentationIndex? xmlDocumentation)
    {
        return new TorchFieldDescriptor(
            GetOutputName(method.ReturnType),
            TorchSignatureFormatter.FormatTypeName(method.ReturnType),
            xmlDocumentation?.GetReturns(method) ?? "No description."
        );
    }

    private static string GetOutputName(Type returnType)
    {
        if (returnType == typeof(void))
        {
            return "result";
        }

        if (returnType.IsArray)
        {
            return "outputs";
        }

        var fullName = returnType.FullName ?? returnType.Name;
        return fullName.Contains("Tensor", StringComparison.Ordinal) ? "output" : "result";
    }
}

internal static class TorchSharpMarkdownRenderer
{
    public static string Render(TorchCatalog catalog)
    {
        var builder = new StringBuilder();

        for (var index = 0; index < catalog.Operators.Count; index++)
        {
            var op = catalog.Operators[index];
            builder.AppendLine($"# {op.Name}");
            builder.AppendLine(op.Description);
            builder.AppendLine();

            AppendSection(builder, "Inputs", op.Inputs);
            builder.AppendLine();
            AppendSection(builder, "Outputs", op.Outputs);
            builder.AppendLine();
            AppendSection(builder, "Attributes", op.Attributes);

            if (index < catalog.Operators.Count - 1)
            {
                builder.AppendLine();
            }
        }

        return builder.ToString();
    }

    private static void AppendSection(StringBuilder builder, string title, IReadOnlyList<TorchFieldDescriptor> fields)
    {
        builder.AppendLine($"## {title}");

        if (fields.Count == 0)
        {
            builder.AppendLine("_None._");
            return;
        }

        foreach (var field in fields)
        {
            builder.AppendLine($"* `{field.Name}` ({field.TypeDisplay}) - {field.Description}");
        }
    }
}

internal static class TorchSignatureFormatter
{
    public static string FormatParameterType(ParameterInfo parameter)
    {
        if (parameter.ParameterType.IsByRef)
        {
            var prefix = parameter.IsOut ? "out " : "ref ";
            return prefix + FormatTypeName(parameter.ParameterType.GetElementType()!);
        }

        return FormatTypeName(parameter.ParameterType);
    }

    public static string FormatTypeName(Type type)
    {
        if (type.IsByRef)
        {
            return $"{FormatTypeName(type.GetElementType()!)}&";
        }

        if (type.IsArray)
        {
            var commas = new string(',', type.GetArrayRank() - 1);
            return $"{FormatTypeName(type.GetElementType()!)}[{commas}]";
        }

        if (Nullable.GetUnderlyingType(type) is Type underlyingType)
        {
            return $"{FormatTypeName(underlyingType)}?";
        }

        if (type.IsGenericParameter)
        {
            return type.Name;
        }

        if (type.IsGenericType)
        {
            var genericTypeDefinition = type.GetGenericTypeDefinition();
            var typeName = genericTypeDefinition.Name;
            var tickIndex = typeName.IndexOf('`');

            if (tickIndex >= 0)
            {
                typeName = typeName[..tickIndex];
            }

            var prefix = type.IsNested
                ? $"{FormatTypeName(type.DeclaringType!)}."
                : string.IsNullOrWhiteSpace(type.Namespace) ? string.Empty : $"{type.Namespace}.";

            return $"{prefix}{typeName}<{string.Join(", ", type.GetGenericArguments().Select(FormatTypeName))}>";
        }

        if (type.IsNested)
        {
            return $"{FormatTypeName(type.DeclaringType!)}.{type.Name}";
        }

        return type.FullName ?? type.Name;
    }

    public static string FormatConstant(object? value)
    {
        return value switch
        {
            null => "null",
            string text => $"\"{text}\"",
            char ch => $"'{ch}'",
            bool boolean => boolean ? "true" : "false",
            Enum @enum => $"{FormatTypeName(@enum.GetType())}.{Enum.GetName(@enum.GetType(), @enum)}",
            float number => number.ToString("R"),
            double number => number.ToString("R"),
            decimal number => number.ToString(System.Globalization.CultureInfo.InvariantCulture),
            _ => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? value.ToString() ?? "null",
        };
    }

    public static string FormatDocumentationTypeName(Type type)
    {
        if (type.IsByRef)
        {
            return $"{FormatDocumentationTypeName(type.GetElementType()!)}@";
        }

        if (type.IsArray)
        {
            if (type.GetArrayRank() == 1)
            {
                return $"{FormatDocumentationTypeName(type.GetElementType()!)}[]";
            }

            var dimensions = string.Join(",", Enumerable.Repeat("0:", type.GetArrayRank()));
            return $"{FormatDocumentationTypeName(type.GetElementType()!)}[{dimensions}]";
        }

        if (type.IsGenericParameter)
        {
            var prefix = type.DeclaringMethod is null ? "`" : "``";
            return $"{prefix}{type.GenericParameterPosition}";
        }

        if (type.IsGenericType)
        {
            var genericTypeDefinition = type.GetGenericTypeDefinition();
            var definitionName = GetDocumentationTypeName(genericTypeDefinition);
            var arguments = string.Join(",", type.GetGenericArguments().Select(FormatDocumentationTypeName));
            return $"{definitionName}{{{arguments}}}";
        }

        return GetDocumentationTypeName(type);
    }

    private static string GetDocumentationTypeName(Type type)
    {
        var fullName = type.FullName ?? type.Name;
        return fullName.Replace('+', '.');
    }
}

internal sealed class TorchXmlDocumentationIndex
{
    private readonly Dictionary<string, XElement> _members;

    private TorchXmlDocumentationIndex(string sourcePath, Dictionary<string, XElement> members)
    {
        SourcePath = sourcePath;
        _members = members;
    }

    public string SourcePath { get; }

    public static TorchXmlDocumentationIndex? TryLoad(string? xmlPath)
    {
        if (string.IsNullOrWhiteSpace(xmlPath) || !File.Exists(xmlPath))
        {
            return null;
        }

        var document = XDocument.Load(xmlPath, LoadOptions.PreserveWhitespace);
        var members = document.Root?
            .Element("members")?
            .Elements("member")
            .Select(member => new
            {
                Name = member.Attribute("name")?.Value,
                Member = member,
            })
            .Where(entry => !string.IsNullOrWhiteSpace(entry.Name))
            .ToDictionary(entry => entry.Name!, entry => entry.Member, StringComparer.Ordinal);

        return members is null ? null : new TorchXmlDocumentationIndex(Path.GetFullPath(xmlPath), members);
    }

    public string? GetSummary(MethodInfo method)
    {
        return TryGetMember(method, out var member) ? Normalize(member.Element("summary")?.Value) : null;
    }

    public string? GetReturns(MethodInfo method)
    {
        return TryGetMember(method, out var member) ? Normalize(member.Element("returns")?.Value) : null;
    }

    public string? GetParameterComment(MethodInfo? method, string? parameterName)
    {
        if (method is null || string.IsNullOrWhiteSpace(parameterName))
        {
            return null;
        }

        if (!TryGetMember(method, out var member))
        {
            return null;
        }

        return Normalize(
            member.Elements("param")
                .FirstOrDefault(param => string.Equals(param.Attribute("name")?.Value, parameterName, StringComparison.Ordinal))?
                .Value
        );
    }

    private bool TryGetMember(MethodInfo method, out XElement member)
    {
        return _members.TryGetValue(BuildMethodDocumentationId(method), out member!);
    }

    private static string BuildMethodDocumentationId(MethodInfo method)
    {
        var builder = new StringBuilder();
        builder.Append("M:");
        builder.Append(TorchSignatureFormatter.FormatDocumentationTypeName(method.DeclaringType!));
        builder.Append('.');
        builder.Append(method.Name);

        if (method.IsGenericMethodDefinition)
        {
            builder.Append("``");
            builder.Append(method.GetGenericArguments().Length);
        }

        var parameters = method.GetParameters();

        if (parameters.Length > 0)
        {
            builder.Append('(');
            builder.Append(string.Join(",", parameters.Select(parameter => TorchSignatureFormatter.FormatDocumentationTypeName(parameter.ParameterType))));
            builder.Append(')');
        }

        return builder.ToString();
    }

    private static string? Normalize(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        var parts = text
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(part => !string.IsNullOrWhiteSpace(part));

        var normalized = string.Join(" ", parts).Trim();
        return string.IsNullOrWhiteSpace(normalized) ? null : normalized;
    }
}
