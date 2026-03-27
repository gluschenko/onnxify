using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Xml.Linq;
using Onnxify;

namespace Onnxify.Introspection;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            var options = Options.Parse(args);
            var assembly = typeof(OnnxModel).Assembly;
            var xmlDocumentation = XmlDocumentationIndex.TryLoad(options.XmlPath ?? TryFindAdjacentXml(assembly.Location));
            var catalog = ApiCatalog.Create(assembly, xmlDocumentation);
            var markdown = MarkdownRenderer.Render(catalog, assembly, xmlDocumentation, options);

            var outputPath = Path.GetFullPath(options.OutputPath);
            var outputDirectory = Path.GetDirectoryName(outputPath);

            if (!string.IsNullOrWhiteSpace(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            File.WriteAllText(outputPath, markdown, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            Console.WriteLine($"Saved Markdown to: {outputPath}");
            Console.WriteLine($"Exported types: {catalog.Types.Count}");
            Console.WriteLine($"Method groups: {catalog.MethodGroups.Count}");
            Console.WriteLine($"Static methods: {catalog.StaticMethods.Count}");
            Console.WriteLine($"Instance methods: {catalog.InstanceMethods.Count}");
            Console.WriteLine($"Extension methods: {catalog.ExtensionMethods.Count}");
            Console.WriteLine($"XML docs: {(xmlDocumentation is null ? "not found" : xmlDocumentation.SourcePath)}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static string? TryFindAdjacentXml(string assemblyPath)
    {
        var xmlPath = Path.ChangeExtension(assemblyPath, ".xml");
        return File.Exists(xmlPath) ? xmlPath : null;
    }
}

internal sealed record Options(string OutputPath, string? XmlPath)
{
    public static Options Parse(string[] args)
    {
        string outputPath = Path.Combine(Environment.CurrentDirectory, "artifacts", "onnxify-public-api.md");
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

        return new Options(outputPath, xmlPath);
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
            Onnxify public API catalog generator

            Options:
              -o, --output <path>   Markdown output path. Default: ./artifacts/onnxify-public-api.md
                  --xml <path>      Optional XML documentation file for Onnxify.dll
              -h, --help            Show this help
            """
        );
    }
}

internal static class ApiCatalog
{
    public static Catalog Create(Assembly assembly, XmlDocumentationIndex? xmlDocumentation)
    {
        var exportedTypes = assembly.GetExportedTypes()
            .OrderBy(type => type.FullName, StringComparer.Ordinal)
            .ToArray();

        var staticMethods = new List<MethodDescriptor>();
        var instanceMethods = new List<MethodDescriptor>();
        var extensionMethods = new List<MethodDescriptor>();

        foreach (var type in exportedTypes)
        {
            foreach (var method in GetDeclaredPublicMethods(type))
            {
                if (IsExtensionMethod(method))
                {
                    extensionMethods.Add(MethodDescriptor.Create(method, xmlDocumentation, MethodCategory.Extension));
                }
                else if (method.IsStatic)
                {
                    staticMethods.Add(MethodDescriptor.Create(method, xmlDocumentation, MethodCategory.Static));
                }
                else
                {
                    instanceMethods.Add(MethodDescriptor.Create(method, xmlDocumentation, MethodCategory.Instance));
                }
            }
        }

        var typeDescriptors = exportedTypes
            .Select(type => TypeDescriptor.Create(type, xmlDocumentation))
            .ToArray();

        var methodGroups = staticMethods
            .Concat(instanceMethods)
            .Concat(extensionMethods)
            .GroupBy(method => method.Name, StringComparer.Ordinal)
            .Select(group => new MethodGroup(
                group.Key,
                group.Where(method => method.Category == MethodCategory.Static).OrderBy(method => method, MethodDescriptorComparer.Instance).ToArray(),
                group.Where(method => method.Category == MethodCategory.Instance).OrderBy(method => method, MethodDescriptorComparer.Instance).ToArray(),
                group.Where(method => method.Category == MethodCategory.Extension).OrderBy(method => method, MethodDescriptorComparer.Instance).ToArray()))
            .OrderBy(group => group.Name, StringComparer.Ordinal)
            .ToArray();

        return new Catalog(
            typeDescriptors,
            methodGroups,
            staticMethods.OrderBy(method => method, MethodDescriptorComparer.Instance).ToArray(),
            instanceMethods.OrderBy(method => method, MethodDescriptorComparer.Instance).ToArray(),
            extensionMethods.OrderBy(method => method, MethodDescriptorComparer.Instance).ToArray()
        );
    }

    private static IEnumerable<MethodInfo> GetDeclaredPublicMethods(Type type)
    {
        return type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Where(method => !method.IsSpecialName);
    }

    private static bool IsExtensionMethod(MethodInfo method)
    {
        return method.IsDefined(typeof(ExtensionAttribute), inherit: false);
    }
}

internal sealed record Catalog(
    IReadOnlyList<TypeDescriptor> Types,
    IReadOnlyList<MethodGroup> MethodGroups,
    IReadOnlyList<MethodDescriptor> StaticMethods,
    IReadOnlyList<MethodDescriptor> InstanceMethods,
    IReadOnlyList<MethodDescriptor> ExtensionMethods);

internal sealed record TypeDescriptor(
    string DisplayName,
    string Kind,
    string? BaseType,
    IReadOnlyList<string> Interfaces,
    string? Summary)
{
    public static TypeDescriptor Create(Type type, XmlDocumentationIndex? xmlDocumentation)
    {
        return new TypeDescriptor(
            SignatureFormatter.FormatTypeName(type),
            GetKind(type),
            type.BaseType is null || type.BaseType == typeof(object) ? null : SignatureFormatter.FormatTypeName(type.BaseType),
            type.GetInterfaces().Select(SignatureFormatter.FormatTypeName).OrderBy(name => name, StringComparer.Ordinal).ToArray(),
            xmlDocumentation?.GetSummary(type)
        );
    }

    private static string GetKind(Type type)
    {
        if (type.IsInterface)
        {
            return "interface";
        }

        if (type.IsEnum)
        {
            return "enum";
        }

        if (type.IsValueType)
        {
            return "struct";
        }

        if (type.IsAbstract && type.IsSealed)
        {
            return "static class";
        }

        if (type.IsAbstract)
        {
            return "abstract class";
        }

        return "class";
    }
}

internal sealed record MethodGroup(
    string Name,
    IReadOnlyList<MethodDescriptor> StaticMethods,
    IReadOnlyList<MethodDescriptor> InstanceMethods,
    IReadOnlyList<MethodDescriptor> ExtensionMethods);

internal enum MethodCategory
{
    Static,
    Instance,
    Extension,
}

internal sealed record MethodDescriptor(
    string Name,
    string Signature,
    string DeclaringTypeDisplayName,
    string ReturnTypeDisplayName,
    MethodCategory Category,
    IReadOnlyList<ParameterDescriptor> Parameters,
    IReadOnlyList<string> GenericArguments,
    string? Summary,
    IReadOnlyDictionary<string, string> ParameterComments)
{
    public static MethodDescriptor Create(MethodInfo method, XmlDocumentationIndex? xmlDocumentation, MethodCategory category)
    {
        var parameters = method.GetParameters();
        var summary = xmlDocumentation?.GetSummary(method);
        var parameterComments = parameters
            .Select(parameter => new KeyValuePair<string, string?>(parameter.Name ?? string.Empty, xmlDocumentation?.GetParameterComment(method, parameter.Name)))
            .Where(pair => !string.IsNullOrWhiteSpace(pair.Value))
            .ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.Ordinal);

        return new MethodDescriptor(
            method.Name,
            SignatureFormatter.FormatMethod(method),
            SignatureFormatter.FormatTypeName(method.DeclaringType!),
            SignatureFormatter.FormatTypeName(method.ReturnType),
            category,
            parameters.Select((parameter, index) => ParameterDescriptor.Create(parameter, category == MethodCategory.Extension && index == 0)).ToArray(),
            method.IsGenericMethodDefinition
                ? method.GetGenericArguments().Select(arg => arg.Name).ToArray()
                : Array.Empty<string>(),
            summary,
            parameterComments
        );
    }
}

internal sealed record ParameterDescriptor(
    string Name,
    string TypeDisplayName,
    bool IsExtensionReceiver,
    bool IsOptional,
    bool IsParams,
    string? DefaultValue)
{
    public static ParameterDescriptor Create(ParameterInfo parameter, bool isExtensionReceiver)
    {
        return new ParameterDescriptor(
            parameter.Name ?? string.Empty,
            SignatureFormatter.FormatParameterType(parameter),
            isExtensionReceiver,
            parameter.IsOptional,
            parameter.GetCustomAttribute<ParamArrayAttribute>() is not null,
            parameter.IsOptional ? SignatureFormatter.FormatConstant(parameter.DefaultValue) : null
        );
    }
}

internal static class MarkdownRenderer
{
    public static string Render(Catalog catalog, Assembly assembly, XmlDocumentationIndex? xmlDocumentation, Options options)
    {
        var builder = new StringBuilder();

        builder.AppendLine("# Onnxify Public API Catalog");
        builder.AppendLine();
        builder.AppendLine($"- Generated at: `{DateTimeOffset.Now:O}`");
        builder.AppendLine($"- Assembly: `{assembly.Location}`");
        builder.AppendLine($"- Assembly version: `{assembly.GetName().Version}`");
        builder.AppendLine($"- Output: `{Path.GetFullPath(options.OutputPath)}`");
        builder.AppendLine($"- XML docs: `{xmlDocumentation?.SourcePath ?? "not found"}`");
        builder.AppendLine($"- Exported types: `{catalog.Types.Count}`");
        builder.AppendLine($"- Method groups: `{catalog.MethodGroups.Count}`");
        builder.AppendLine($"- Static methods: `{catalog.StaticMethods.Count}`");
        builder.AppendLine($"- Instance methods: `{catalog.InstanceMethods.Count}`");
        builder.AppendLine($"- Extension methods: `{catalog.ExtensionMethods.Count}`");
        builder.AppendLine();
        builder.AppendLine("## Types");
        builder.AppendLine();

        foreach (var type in catalog.Types)
        {
            builder.AppendLine($"### `{type.DisplayName}`");
            builder.AppendLine();
            builder.AppendLine($"- Kind: `{type.Kind}`");

            if (!string.IsNullOrWhiteSpace(type.BaseType))
            {
                builder.AppendLine($"- Base type: `{type.BaseType}`");
            }

            if (type.Interfaces.Count > 0)
            {
                builder.AppendLine($"- Interfaces: `{string.Join("`, `", type.Interfaces)}`");
            }

            if (!string.IsNullOrWhiteSpace(type.Summary))
            {
                builder.AppendLine($"- Summary: {type.Summary}");
            }

            builder.AppendLine();
        }

        builder.AppendLine("## Method Index");
        builder.AppendLine();

        foreach (var group in catalog.MethodGroups)
        {
            builder.AppendLine($"- [{group.Name}](#{Anchorize(group.Name)})");
        }

        foreach (var group in catalog.MethodGroups)
        {
            builder.AppendLine();
            builder.AppendLine($"## {group.Name}");
            builder.AppendLine();
            builder.AppendLine($"- Static overloads: `{group.StaticMethods.Count}`");
            builder.AppendLine($"- Instance overloads: `{group.InstanceMethods.Count}`");
            builder.AppendLine($"- Extension overloads: `{group.ExtensionMethods.Count}`");

            AppendSection(builder, "Static API", group.StaticMethods);
            AppendSection(builder, "Instance API", group.InstanceMethods);
            AppendSection(builder, "Extension API", group.ExtensionMethods);
        }

        return builder.ToString();
    }

    private static void AppendSection(StringBuilder builder, string title, IReadOnlyList<MethodDescriptor> methods)
    {
        builder.AppendLine();
        builder.AppendLine($"### {title}");
        builder.AppendLine();

        if (methods.Count == 0)
        {
            builder.AppendLine("_No methods found._");
            return;
        }

        for (var index = 0; index < methods.Count; index++)
        {
            var method = methods[index];
            builder.AppendLine($"#### `{method.Signature}`");
            builder.AppendLine();
            builder.AppendLine($"- Declaring type: `{method.DeclaringTypeDisplayName}`");
            builder.AppendLine($"- Return type: `{method.ReturnTypeDisplayName}`");

            if (method.GenericArguments.Count > 0)
            {
                builder.AppendLine($"- Generic arguments: `{string.Join("`, `", method.GenericArguments)}`");
            }

            if (!string.IsNullOrWhiteSpace(method.Summary))
            {
                builder.AppendLine($"- Summary: {method.Summary}");
            }

            if (method.Parameters.Count == 0)
            {
                builder.AppendLine("- Parameters: none");
            }
            else
            {
                builder.AppendLine("- Parameters:");

                foreach (var parameter in method.Parameters)
                {
                    var suffix = new List<string>();

                    if (parameter.IsExtensionReceiver)
                    {
                        suffix.Add("extension receiver");
                    }

                    if (parameter.IsParams)
                    {
                        suffix.Add("params");
                    }

                    if (parameter.IsOptional && parameter.DefaultValue is not null)
                    {
                        suffix.Add($"optional = {parameter.DefaultValue}");
                    }

                    var description = suffix.Count == 0
                        ? string.Empty
                        : $" ({string.Join(", ", suffix)})";

                    builder.AppendLine($"  - `{parameter.TypeDisplayName} {parameter.Name}`{description}");

                    if (method.ParameterComments.TryGetValue(parameter.Name, out var comment))
                    {
                        builder.AppendLine($"    - {comment}");
                    }
                }
            }

            if (index < methods.Count - 1)
            {
                builder.AppendLine();
            }
        }
    }

    private static string Anchorize(string text)
    {
        return text
            .Trim()
            .ToLowerInvariant()
            .Replace(" ", "-")
            .Replace(".", string.Empty)
            .Replace("_", "-");
    }
}

internal static class SignatureFormatter
{
    public static string FormatMethod(MethodInfo method)
    {
        var builder = new StringBuilder();

        builder.Append(FormatTypeName(method.DeclaringType!));
        builder.Append('.');
        builder.Append(method.Name);

        if (method.IsGenericMethodDefinition)
        {
            builder.Append('<');
            builder.Append(string.Join(", ", method.GetGenericArguments().Select(arg => arg.Name)));
            builder.Append('>');
        }

        builder.Append('(');
        builder.Append(string.Join(", ", method.GetParameters().Select(FormatParameter)));
        builder.Append(')');
        return builder.ToString();
    }

    public static string FormatParameter(ParameterInfo parameter)
    {
        var builder = new StringBuilder();

        if (parameter.GetCustomAttribute<ParamArrayAttribute>() is not null)
        {
            builder.Append("params ");
        }

        builder.Append(FormatParameterType(parameter));
        builder.Append(' ');
        builder.Append(parameter.Name);

        if (parameter.IsOptional)
        {
            builder.Append(" = ");
            builder.Append(FormatConstant(parameter.DefaultValue));
        }

        return builder.ToString();
    }

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

        if (type.IsPointer)
        {
            return $"{FormatTypeName(type.GetElementType()!)}*";
        }

        if (type.IsArray)
        {
            var commas = new string(',', type.GetArrayRank() - 1);
            return $"{FormatTypeName(type.GetElementType()!)}[{commas}]";
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

        if (type.IsPointer)
        {
            return $"{FormatDocumentationTypeName(type.GetElementType()!)}*";
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

internal sealed class MethodDescriptorComparer : IComparer<MethodDescriptor>
{
    public static MethodDescriptorComparer Instance { get; } = new();

    public int Compare(MethodDescriptor? x, MethodDescriptor? y)
    {
        if (ReferenceEquals(x, y))
        {
            return 0;
        }

        if (x is null)
        {
            return -1;
        }

        if (y is null)
        {
            return 1;
        }

        var result = StringComparer.Ordinal.Compare(x.DeclaringTypeDisplayName, y.DeclaringTypeDisplayName);

        if (result != 0)
        {
            return result;
        }

        result = x.Parameters.Count.CompareTo(y.Parameters.Count);

        if (result != 0)
        {
            return result;
        }

        for (var index = 0; index < x.Parameters.Count && index < y.Parameters.Count; index++)
        {
            result = StringComparer.Ordinal.Compare(x.Parameters[index].TypeDisplayName, y.Parameters[index].TypeDisplayName);

            if (result != 0)
            {
                return result;
            }
        }

        return StringComparer.Ordinal.Compare(x.Signature, y.Signature);
    }
}

internal sealed class XmlDocumentationIndex
{
    private readonly Dictionary<string, XElement> _members;

    private XmlDocumentationIndex(string sourcePath, Dictionary<string, XElement> members)
    {
        SourcePath = sourcePath;
        _members = members;
    }

    public string SourcePath { get; }

    public static XmlDocumentationIndex? TryLoad(string? xmlPath)
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

        if (members is null)
        {
            return null;
        }

        return new XmlDocumentationIndex(Path.GetFullPath(xmlPath), members);
    }

    public string? GetSummary(Type type)
    {
        return _members.TryGetValue($"T:{SignatureFormatter.FormatDocumentationTypeName(type)}", out var member)
            ? Normalize(member.Element("summary")?.Value)
            : null;
    }

    public string? GetSummary(MethodInfo method)
    {
        return TryGetMember(method, out var member)
            ? Normalize(member.Element("summary")?.Value)
            : null;
    }

    public string? GetParameterComment(MethodInfo method, string? parameterName)
    {
        if (string.IsNullOrWhiteSpace(parameterName))
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
        var key = BuildMethodDocumentationId(method);
        return _members.TryGetValue(key, out member!);
    }

    private static string BuildMethodDocumentationId(MethodInfo method)
    {
        var builder = new StringBuilder();
        builder.Append("M:");
        builder.Append(SignatureFormatter.FormatDocumentationTypeName(method.DeclaringType!));
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
            builder.Append(string.Join(",", parameters.Select(parameter => SignatureFormatter.FormatDocumentationTypeName(parameter.ParameterType))));
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
