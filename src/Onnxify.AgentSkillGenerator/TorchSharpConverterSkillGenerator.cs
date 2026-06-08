using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using Onnxify.TorchSharp;

namespace Onnxify.AgentSkillGenerator;

internal static class TorchSharpConverterSkillGenerator
{
    private const string NEW_LINE = "\n";
    private static readonly IReadOnlyDictionary<Type, string> SourceFilePaths =
        new Dictionary<Type, string>
        {
            [typeof(TorchModuleExtensions)] = "src/Onnxify.TorchSharp/TorchModuleExtensions.cs",
            [typeof(TorchTensorOperatorExtensions)] = "src/Onnxify.TorchSharp/TorchTensorOperatorExtensions.cs",
        };

    public static int Run(string[] args)
    {
        string repoRoot = FindRepositoryRoot(Directory.GetCurrentDirectory())
            ?? FindRepositoryRoot(AppContext.BaseDirectory)
            ?? throw new DirectoryNotFoundException("Repository root was not found.");

        string skillRoot = ResolveSkillRoot(repoRoot);
        string outputRoot = Path.Combine(skillRoot, "references", "torchsharp-converters");
        var generatedFiles = BuildGeneratedFiles(GetExistingGeneratedRelativePaths(outputRoot));

        RewriteGeneratedDirectory(outputRoot, generatedFiles);

        int converterCount = generatedFiles.Count - 1;
        int torchOpFileCount = generatedFiles.Keys.Count(path => path.StartsWith("torch-ops", StringComparison.Ordinal));

        Console.WriteLine($"Repository root: {MakeRelative(repoRoot, repoRoot)}");
        Console.WriteLine($"Skill root: {MakeRelative(repoRoot, skillRoot)}");
        Console.WriteLine(
            $"TorchSharp converter sources: {string.Join(", ", SourceFilePaths.Values.OrderBy(static x => x, StringComparer.Ordinal))}");
        Console.WriteLine($"Generated converter files: {converterCount}");
        Console.WriteLine($"Torch-op-backed converter files: {torchOpFileCount}");
        Console.WriteLine($"Index file: {MakeRelative(repoRoot, Path.Combine(outputRoot, "index.md"))}");

        return 0;
    }

    internal static IReadOnlyDictionary<string, string> BuildGeneratedFiles(
        IReadOnlySet<string>? existingRelativePaths = null
    )
    {
        var converterMethods = SourceFilePaths
            .SelectMany(static entry => entry.Key
                .GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly)
                .Where(IsPublicExportExtension)
                .Select(method => (Method: method, SourceFile: entry.Value)))
            .ToArray();

        var receiverCounts = converterMethods
            .GroupBy(static x => GetFriendlyTypeName(x.Method.GetParameters()[0].ParameterType))
            .ToDictionary(static group => group.Key, static group => group.Count(), StringComparer.Ordinal);

        var converters = converterMethods
            .Select(x => CreateConverterDoc(x.Method, x.SourceFile, receiverCounts, existingRelativePaths))
            .OrderBy(x => x.Kind)
            .ThenBy(x => x.ReceiverTypeName, StringComparer.Ordinal)
            .ThenBy(x => x.Signature, StringComparer.Ordinal)
            .ToArray();

        var files = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["index.md"] = BuildIndexMarkdown(converters),
        };

        foreach (ConverterDoc converter in converters)
        {
            files[converter.RelativePath] = BuildConverterMarkdown(converter);
        }

        return files;
    }

    private static bool IsPublicExportExtension(MethodInfo method)
    {
        if (!method.IsPublic ||
            !method.IsStatic ||
            !method.Name.StartsWith("Export", StringComparison.Ordinal) ||
            !method.IsDefined(typeof(ExtensionAttribute), inherit: false))
        {
            return false;
        }

        return method.GetParameters().Length > 0;
    }

    private static ConverterDoc CreateConverterDoc(
        MethodInfo method,
        string sourceFile,
        IReadOnlyDictionary<string, int> receiverCounts,
        IReadOnlySet<string>? existingRelativePaths
    )
    {
        ParameterInfo[] parameters = method.GetParameters();
        Type receiverType = parameters[0].ParameterType;
        string[] torchOps = method
            .GetCustomAttributes<TorchOpAttribute>(inherit: false)
            .Select(attribute => attribute.Name)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        ConverterKind kind = GetConverterKind(receiverType, torchOps);
        string receiverTypeName = GetFriendlyTypeName(receiverType);
        string fileName = receiverCounts.TryGetValue(receiverTypeName, out int count) && count > 1
            ? $"{SanitizeFileName(receiverTypeName)}__{SanitizeFileName(GetConverterSlug(method, torchOps, kind, receiverTypeName, existingRelativePaths))}.md"
            : $"{SanitizeFileName(receiverTypeName)}.md";
        string relativePath = Path.Combine(GetFolderName(kind), fileName);

        return new ConverterDoc(
            receiverTypeName,
            GetFriendlyTypeName(method.ReturnType),
            GetConverterDisplaySignature(method),
            sourceFile,
            kind,
            relativePath,
            parameters.Select((parameter, index) => CreateParameterDoc(parameter, index)).ToArray(),
            torchOps,
            GetReturnMembers(method.ReturnType));
    }

    private static ConverterKind GetConverterKind(Type receiverType, IReadOnlyList<string> torchOps)
    {
        if (torchOps.Count != 0)
        {
            return ConverterKind.TorchOpBacked;
        }

        if (string.Equals(receiverType.Name, "Sequential", StringComparison.Ordinal))
        {
            return ConverterKind.Composite;
        }

        if (string.Equals(receiverType.Name, "TorchModule", StringComparison.Ordinal) ||
            string.Equals(receiverType.FullName, "TorchSharp.torch+TorchModule", StringComparison.Ordinal))
        {
            return ConverterKind.DispatchEntryPoint;
        }

        return ConverterKind.Composite;
    }

    private static ParameterDoc CreateParameterDoc(ParameterInfo parameter, int index)
    {
        string role = index == 0 ? "receiver" : "argument";
        return new ParameterDoc(
            index,
            parameter.Name ?? $"arg{index.ToString(CultureInfo.InvariantCulture)}",
            GetFriendlyTypeName(parameter.ParameterType),
            role);
    }

    private static IReadOnlyList<ReturnMemberDoc> GetReturnMembers(Type returnType)
    {
        if (returnType == typeof(void))
        {
            return [];
        }

        if (returnType == typeof(IOnnxGraphEdge))
        {
            return [new ReturnMemberDoc("[single edge]", "IOnnxGraphEdge")];
        }

        var properties = returnType
            .GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .Where(property => property.GetIndexParameters().Length == 0)
            .Select(property => new ReturnMemberDoc(property.Name, GetFriendlyTypeName(property.PropertyType)));

        var fields = returnType
            .GetFields(BindingFlags.Instance | BindingFlags.Public)
            .Select(field => new ReturnMemberDoc(field.Name, GetFriendlyTypeName(field.FieldType)));

        var members = properties
            .Concat(fields)
            .Distinct()
            .OrderBy(member => member.Name, StringComparer.Ordinal)
            .ToArray();

        return members.Length == 0
            ? [new ReturnMemberDoc("[no public members detected]", GetFriendlyTypeName(returnType))]
            : members;
    }

    private static string BuildIndexMarkdown(IReadOnlyList<ConverterDoc> converters)
    {
        var builder = new StringBuilder();
        builder.AppendLine("# Onnxify TorchSharp Converter Instructions");
        builder.AppendLine();
        builder.AppendLine("Autogenerated by src/Onnxify.AgentSkillGenerator.");
        builder.AppendLine("Do not hand-edit files in this directory; rerun the generator instead.");
        builder.AppendLine();
        builder.AppendLine($"- Total public Export(...) signatures: {converters.Count}");
        builder.AppendLine($"- Torch-op-backed converters: {converters.Count(x => x.Kind == ConverterKind.TorchOpBacked)}");
        builder.AppendLine($"- Distinct Torch ops declared through [TorchOp]: {converters.SelectMany(x => x.TorchOps).Distinct(StringComparer.Ordinal).Count()}");
        builder.AppendLine(
            $"- Source files: {string.Join(", ", SourceFilePaths.Values.OrderBy(static x => x, StringComparer.Ordinal))}");
        builder.AppendLine();

        foreach (ConverterKind kind in Enum.GetValues<ConverterKind>())
        {
            ConverterDoc[] group = converters
                .Where(converter => converter.Kind == kind)
                .ToArray();

            if (group.Length == 0)
            {
                continue;
            }

            builder.AppendLine($"## {GetKindTitle(kind)}");
            builder.AppendLine();
            builder.AppendLine("| Receiver | Return | Torch ops | File |");
            builder.AppendLine("| --- | --- | --- | --- |");

            foreach (ConverterDoc converter in group)
            {
                string torchOps = converter.TorchOps.Count == 0
                    ? "[none]"
                    : string.Join(", ", converter.TorchOps.Select(EscapeMarkdownCell));

                builder.Append("| ")
                    .Append(EscapeMarkdownCell(converter.ReceiverTypeName))
                    .Append(" | ")
                    .Append(EscapeMarkdownCell(converter.ReturnTypeName))
                    .Append(" | ")
                    .Append(torchOps)
                    .Append(" | [")
                    .Append(EscapeMarkdownCell(Path.GetFileNameWithoutExtension(converter.RelativePath)))
                    .Append("](")
                    .Append(converter.RelativePath.Replace('\\', '/'))
                    .AppendLine(") |");
            }

            builder.AppendLine();
        }

        return builder.ToString().TrimEnd();
    }

    private static string BuildConverterMarkdown(ConverterDoc converter)
    {
        var builder = new StringBuilder();
        builder.Append("# ").Append(converter.ReceiverTypeName).AppendLine(" Converter");
        builder.AppendLine();
        builder.AppendLine("Autogenerated by src/Onnxify.AgentSkillGenerator.");
        builder.AppendLine("Do not hand-edit this file; rerun the generator instead.");
        builder.AppendLine();
        builder.Append("- Kind: ").Append(GetKindTitle(converter.Kind)).AppendLine();
        builder.Append("- Signature: ").Append(EscapeMarkdownCell(converter.Signature)).AppendLine();
        builder.Append("- Source: ").Append(converter.SourceFile).AppendLine();
        builder.Append("- Receiver: ").Append(converter.ReceiverTypeName).AppendLine();
        builder.Append("- Return type: ").Append(converter.ReturnTypeName).AppendLine();
        builder.Append("- Torch ops: ");
        if (converter.TorchOps.Count == 0)
        {
            builder.AppendLine("[none]");
        }
        else
        {
            builder.AppendLine(string.Join(", ", converter.TorchOps));
        }

        builder.AppendLine();
        builder.AppendLine("## Parameters");
        builder.AppendLine();
        builder.AppendLine("| Position | Name | Type | Role |");
        builder.AppendLine("| --- | --- | --- | --- |");
        foreach (ParameterDoc parameter in converter.Parameters)
        {
            builder.Append("| ")
                .Append(parameter.Position.ToString(CultureInfo.InvariantCulture))
                .Append(" | ")
                .Append(EscapeMarkdownCell(parameter.Name))
                .Append(" | ")
                .Append(EscapeMarkdownCell(parameter.TypeName))
                .Append(" | ")
                .Append(EscapeMarkdownCell(parameter.Role))
                .AppendLine(" |");
        }

        builder.AppendLine();
        builder.AppendLine("## Return Shape");
        builder.AppendLine();
        builder.AppendLine("| Member | Type |");
        builder.AppendLine("| --- | --- |");
        foreach (ReturnMemberDoc member in converter.ReturnMembers)
        {
            builder.Append("| ")
                .Append(EscapeMarkdownCell(member.Name))
                .Append(" | ")
                .Append(EscapeMarkdownCell(member.TypeName))
                .AppendLine(" |");
        }

        builder.AppendLine();
        builder.AppendLine("## Notes");
        builder.AppendLine();
        builder.AppendLine(GetKindDescription(converter.Kind));

        return builder.ToString().TrimEnd();
    }

    private static string GetKindTitle(ConverterKind kind)
    {
        return kind switch
        {
            ConverterKind.DispatchEntryPoint => "Dispatch Entry Points",
            ConverterKind.Composite => "Composite Converters",
            ConverterKind.TorchOpBacked => "Torch-Op-Backed Converters",
            _ => kind.ToString(),
        };
    }

    private static string GetKindDescription(ConverterKind kind)
    {
        return kind switch
        {
            ConverterKind.DispatchEntryPoint
                => "This signature is a public entry point that dispatches to a more specific converter based on the runtime TorchSharp module type.",
            ConverterKind.Composite
                => "This signature is public but does not declare a [TorchOp]; it composes or chains other converters rather than mapping a single Torch operator directly.",
            ConverterKind.TorchOpBacked
                => "This signature is decorated with one or more [TorchOp] attributes and represents a direct TorchSharp-to-ONNX conversion surface.",
            _ => string.Empty,
        };
    }

    private static string GetFolderName(ConverterKind kind)
    {
        return kind switch
        {
            ConverterKind.DispatchEntryPoint => "entry-points",
            ConverterKind.Composite => "composites",
            ConverterKind.TorchOpBacked => "torch-ops",
            _ => "misc",
        };
    }

    private static string SanitizeFileName(string value)
    {
        var builder = new StringBuilder(value.Length);
        foreach (char ch in value)
        {
            builder.Append(char.IsLetterOrDigit(ch) || ch == '.' || ch == '-' ? ch : '_');
        }

        return builder.ToString();
    }

    private static string GetConverterSlug(
        MethodInfo method,
        IReadOnlyList<string> torchOps,
        ConverterKind kind,
        string receiverTypeName,
        IReadOnlySet<string>? existingRelativePaths
    )
    {
        if (torchOps.Count != 0)
        {
            if (existingRelativePaths is not null)
            {
                string folderName = GetFolderName(kind);
                string receiverPrefix = SanitizeFileName(receiverTypeName);
                string? existingSlug = torchOps
                    .OrderByDescending(GetTorchOpSlugScore)
                    .ThenBy(op => op, StringComparer.Ordinal)
                    .FirstOrDefault(op => existingRelativePaths.Contains(
                        Path.Combine(
                            folderName,
                            $"{receiverPrefix}__{SanitizeFileName(op)}.md"
                        )
                    ));

                if (existingSlug is not null)
                {
                    return existingSlug;
                }
            }

            return torchOps
                .OrderByDescending(GetTorchOpSlugScore)
                .ThenBy(op => op, StringComparer.Ordinal)
                .First();
        }

        string parameterSuffix = string.Join(
            "_",
            method.GetParameters()
                .Skip(1)
                .Select(parameter => GetFriendlyTypeName(parameter.ParameterType)));

        return string.IsNullOrWhiteSpace(parameterSuffix)
            ? method.Name
            : $"{method.Name}_{parameterSuffix}";
    }

    private static IReadOnlySet<string> GetExistingGeneratedRelativePaths(string outputRoot)
    {
        if (!Directory.Exists(outputRoot))
        {
            return new HashSet<string>(StringComparer.Ordinal);
        }

        return Directory
            .EnumerateFiles(outputRoot, "*.md", SearchOption.AllDirectories)
            .Select(path => Path.GetRelativePath(outputRoot, path))
            .ToHashSet(StringComparer.Ordinal);
    }

    private static int GetTorchOpSlugScore(string torchOp)
    {
        int score = 0;

        int namespaceSeparatorIndex = torchOp.IndexOf("::", StringComparison.Ordinal);
        string opName = namespaceSeparatorIndex >= 0
            ? torchOp[(namespaceSeparatorIndex + 2)..]
            : torchOp;

        if (opName.Contains('.', StringComparison.Ordinal))
        {
            score += 100;
        }

        if (!opName.StartsWith("__", StringComparison.Ordinal))
        {
            score += 10;
        }

        if (torchOp.StartsWith("aten::", StringComparison.Ordinal))
        {
            score += 3;
        }
        else if (torchOp.StartsWith("prims::", StringComparison.Ordinal))
        {
            score += 2;
        }
        else if (!torchOp.StartsWith("_operator::", StringComparison.Ordinal))
        {
            score += 1;
        }

        return score;
    }

    private static string GetConverterDisplaySignature(MethodInfo method)
    {
        var declaringType = method.DeclaringType?.FullName?.Replace("+", ".", StringComparison.Ordinal) ?? "[unknown]";

        var parameters = string.Join(", ", method.GetParameters().Select((parameter, index) =>
        {
            string prefix = index == 0 && method.IsDefined(typeof(ExtensionAttribute), inherit: false) ? "this " : string.Empty;
            return $"{prefix}{GetFriendlyTypeName(parameter.ParameterType)} {parameter.Name}";
        }));

        return $"{declaringType}.{method.Name}({parameters}) -> {GetFriendlyTypeName(method.ReturnType)}";
    }

    private static string GetFriendlyTypeName(Type type)
    {
        if (type == typeof(void))
        {
            return "void";
        }

        if (type.IsArray)
        {
            return $"{GetFriendlyTypeName(type.GetElementType()!)}[]";
        }

        if (type.IsGenericType)
        {
            var genericTypeName = type.Name;
            int tickIndex = genericTypeName.IndexOf('`');
            if (tickIndex >= 0)
            {
                genericTypeName = genericTypeName[..tickIndex];
            }

            string genericArguments = string.Join(", ", type.GetGenericArguments().Select(GetFriendlyTypeName));
            return $"{GetFriendlyTypeNamePrefix(type)}{genericTypeName}<{genericArguments}>";
        }

        return type switch
        {
            _ when type == typeof(string) => "string",
            _ when type == typeof(bool) => "bool",
            _ when type == typeof(byte) => "byte",
            _ when type == typeof(sbyte) => "sbyte",
            _ when type == typeof(short) => "short",
            _ when type == typeof(ushort) => "ushort",
            _ when type == typeof(int) => "int",
            _ when type == typeof(uint) => "uint",
            _ when type == typeof(long) => "long",
            _ when type == typeof(ulong) => "ulong",
            _ when type == typeof(float) => "float",
            _ when type == typeof(double) => "double",
            _ => $"{GetFriendlyTypeNamePrefix(type)}{type.Name}",
        };
    }

    private static string GetFriendlyTypeNamePrefix(Type type)
    {
        if (type.DeclaringType is not null)
        {
            return GetFriendlyTypeNamePrefix(type.DeclaringType) + type.DeclaringType.Name + ".";
        }

        return string.Empty;
    }

    private static string EscapeMarkdownCell(string value)
    {
        return value
            .Replace("|", "\\|", StringComparison.Ordinal)
            .Replace("\r", " ", StringComparison.Ordinal)
            .Replace("\n", " ", StringComparison.Ordinal);
    }

    private static string NormalizeLineEndings(string value)
    {
        return value.Replace("\r\n", NEW_LINE, StringComparison.Ordinal).Replace("\r", NEW_LINE, StringComparison.Ordinal);
    }

    private static void RewriteGeneratedDirectory(string outputRoot, IReadOnlyDictionary<string, string> files)
    {
        if (Directory.Exists(outputRoot))
        {
            Directory.Delete(outputRoot, recursive: true);
        }

        Directory.CreateDirectory(outputRoot);

        foreach ((string relativePath, string content) in files.OrderBy(x => x.Key, StringComparer.Ordinal))
        {
            string fullPath = Path.Combine(outputRoot, relativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
            File.WriteAllText(fullPath, NormalizeLineEndings(content), new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        }
    }

    private static string ResolveSkillRoot(string repoRoot)
    {
        string[] relativeCandidates =
        [
            Path.Combine(".skills", "agents", "onnxify"),
            Path.Combine(".agents", "skills", "onnxify"),
        ];

        foreach (string relativeCandidate in relativeCandidates)
        {
            string candidate = Path.Combine(repoRoot, relativeCandidate);
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new DirectoryNotFoundException(
            "Onnxify skill root was not found. Expected either '.skills/agents/onnxify' or '.agents/skills/onnxify'.");
    }

    private static string? FindRepositoryRoot(string? currentDirectory)
    {
        DirectoryInfo? directory = currentDirectory is null ? null : new DirectoryInfo(currentDirectory);

        while (directory is not null)
        {
            bool hasGit = Directory.Exists(Path.Combine(directory.FullName, ".git"));
            bool hasSrc = Directory.Exists(Path.Combine(directory.FullName, "src"));

            if (hasGit && hasSrc)
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    private static string MakeRelative(string repoRoot, string path)
    {
        if (string.Equals(repoRoot, path, StringComparison.OrdinalIgnoreCase))
        {
            return ".";
        }

        return Path.GetRelativePath(repoRoot, path).Replace('\\', '/');
    }

    private sealed record ConverterDoc(
        string ReceiverTypeName,
        string ReturnTypeName,
        string Signature,
        string SourceFile,
        ConverterKind Kind,
        string RelativePath,
        IReadOnlyList<ParameterDoc> Parameters,
        IReadOnlyList<string> TorchOps,
        IReadOnlyList<ReturnMemberDoc> ReturnMembers);

    private sealed record ParameterDoc(int Position, string Name, string TypeName, string Role);

    private sealed record ReturnMemberDoc(string Name, string TypeName);

    private enum ConverterKind
    {
        DispatchEntryPoint,
        Composite,
        TorchOpBacked,
    }
}
