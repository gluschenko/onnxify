using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;

namespace Onnxify.TorchSharp.Observer;

internal static partial class Program
{
    private static readonly string[] _onnxScriptModules =
    [
        "core",
        "fft",
        "linalg",
        "nested",
        "nn",
        "prims",
        "quantized_decomposed",
        "sparse",
        "special",
        "vision",
    ];

    private static readonly Regex _torchOpRegex = TorchOpAttributeRegex();
    private static readonly Regex _stringLiteralRegex = StringLiteralPattern();

    private static int Main(string[] args)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        Console.Title = nameof(Onnxify);
        Console.InputEncoding = Encoding.UTF8;
        Console.OutputEncoding = Encoding.UTF8;

        var repoRoot = FindRepositoryRoot(AppContext.BaseDirectory)
            ?? throw new DirectoryNotFoundException("Repository root was not found.");

        var opsDirectory = Path.Combine(
            repoRoot,
            "third_party",
            "onnxscript",
            "onnxscript",
            "function_libs",
            "torch_lib",
            "ops"
        );

        var outputPath = args.Length > 0
            ? Path.GetFullPath(args[0])
            : Path.Combine(repoRoot, "src", "Onnxify.TorchSharp.Observer", "torchsharp-operator-report.md");

        var operators = LoadOperators(opsDirectory);
        var candidates = LoadTorchSharpCandidates();
        var torchSharpCoveredOperators = LoadTorchSharpCoveredOperators();
        var modelGeneratorCoveredOperators = LoadModelGeneratorCoveredOperators();

        var rows = operators
            .Select(op => CreateRow(op, candidates, torchSharpCoveredOperators, modelGeneratorCoveredOperators))
            .OrderBy(row => row.Operator, StringComparer.Ordinal)
            .ToArray();

        var markdown = BuildMarkdown(rows);

        Console.WriteLine(markdown);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllText(outputPath, markdown, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        Console.WriteLine($"Generated {rows.Length} rows.");
        Console.WriteLine(outputPath);
        return 0;
    }

    private static string? FindRepositoryRoot(string? currentDirectory)
    {
        var directory = currentDirectory is null ? null : new DirectoryInfo(currentDirectory);

        while (directory is not null)
        {
            var hasSrc = Directory.Exists(Path.Combine(directory.FullName, "src"));
            var hasThirdParty = Directory.Exists(Path.Combine(directory.FullName, "third_party"));

            if (hasSrc && hasThirdParty)
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    private static IReadOnlyList<OperatorRecord> LoadOperators(string opsDirectory)
    {
        var operators = new Dictionary<string, OperatorRecord>(StringComparer.Ordinal);

        foreach (var module in _onnxScriptModules)
        {
            var path = Path.Combine(opsDirectory, $"{module}.py");
            if (!File.Exists(path))
            {
                continue;
            }

            var content = File.ReadAllText(path);
            foreach (Match match in _torchOpRegex.Matches(content))
            {
                var arguments = match.Groups["args"].Value;
                foreach (Match literal in _stringLiteralRegex.Matches(arguments))
                {
                    var rawOperator = literal.Groups["value"].Value;
                    if (string.IsNullOrWhiteSpace(rawOperator))
                    {
                        continue;
                    }

                    if (!operators.ContainsKey(rawOperator))
                    {
                        operators.Add(rawOperator, new OperatorRecord(rawOperator, module));
                    }
                }
            }
        }

        return operators.Values.ToArray();
    }

    private static IReadOnlyList<TorchSharpCandidate> LoadTorchSharpCandidates()
    {
        const BindingFlags PUBLIC_STATIC = BindingFlags.Public | BindingFlags.Static;
        const BindingFlags PUBLIC_INSTANCE = BindingFlags.Public | BindingFlags.Instance;

        var assembly = typeof(global::TorchSharp.torch).Assembly;
        var candidates = new Dictionary<string, TorchSharpCandidate>(StringComparer.OrdinalIgnoreCase);

        foreach (var type in assembly.GetExportedTypes())
        {
            if (!type.FullName?.StartsWith("TorchSharp.", StringComparison.Ordinal) ?? true)
            {
                continue;
            }

            foreach (var method in type.GetMethods(PUBLIC_STATIC))
            {
                if (method.IsSpecialName)
                {
                    continue;
                }

                AddCandidate(candidates, NormalizeTorchSharpName(method.Name), $"{type.FullName}.{method.Name}");
            }

            if (IsModuleType(type))
            {
                if (type.FullName is not null)
                {
                    AddCandidate(candidates, NormalizeTorchSharpName(type.Name), type.FullName);
                }
            }

            foreach (var property in type.GetProperties(PUBLIC_STATIC))
            {
                if (property.GetMethod is null || property.GetMethod.IsSpecialName)
                {
                    continue;
                }

                AddCandidate(candidates, NormalizeTorchSharpName(property.Name), $"{type.FullName}.{property.Name}");
            }

            foreach (var nested in type.GetNestedTypes(BindingFlags.Public))
            {
                if (IsModuleType(nested))
                {
                    AddCandidate(candidates, NormalizeTorchSharpName(nested.Name), $"{type.FullName}.{nested.Name}");
                }

                foreach (var method in nested.GetMethods(PUBLIC_STATIC | PUBLIC_INSTANCE))
                {
                    if (method.IsSpecialName)
                    {
                        continue;
                    }

                    if (nested.FullName is not null)
                    {
                        AddCandidate(candidates, NormalizeTorchSharpName(method.Name), $"{nested.FullName}.{method.Name}");
                    }
                }
            }
        }

        return candidates.Values.ToArray();
    }

    private static IReadOnlySet<string> LoadTorchSharpCoveredOperators()
    {
        const BindingFlags ALL_MEMBERS =
            BindingFlags.Public |
            BindingFlags.NonPublic |
            BindingFlags.Static |
            BindingFlags.Instance;

        var assembly = typeof(global::Onnxify.TorchSharp.TorchOpAttribute).Assembly;
        var coveredOperators = new HashSet<string>(StringComparer.Ordinal);

        foreach (var type in assembly.GetTypes())
        {
            foreach (global::Onnxify.TorchSharp.TorchOpAttribute attribute in
                type.GetCustomAttributes<global::Onnxify.TorchSharp.TorchOpAttribute>(inherit: false))
            {
                if (!string.IsNullOrWhiteSpace(attribute.Name))
                {
                    coveredOperators.Add(attribute.Name);
                }
            }

            foreach (var method in type.GetMethods(ALL_MEMBERS))
            {
                foreach (global::Onnxify.TorchSharp.TorchOpAttribute attribute in
                    method.GetCustomAttributes<global::Onnxify.TorchSharp.TorchOpAttribute>(inherit: false))
                {
                    if (!string.IsNullOrWhiteSpace(attribute.Name))
                    {
                        coveredOperators.Add(attribute.Name);
                    }
                }
            }
        }

        return coveredOperators;
    }

    private static IReadOnlySet<string> LoadModelGeneratorCoveredOperators()
    {
        const BindingFlags ALL_MEMBERS =
            BindingFlags.Public |
            BindingFlags.NonPublic |
            BindingFlags.Static |
            BindingFlags.Instance;

        var assembly = typeof(global::Onnxify.ModelGenerator.TorchSharpOpAttribute).Assembly;
        var coveredOperators = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var type in GetLoadableTypes(assembly))
        {
            foreach (global::Onnxify.ModelGenerator.TorchSharpOpAttribute attribute in
                type.GetCustomAttributes<global::Onnxify.ModelGenerator.TorchSharpOpAttribute>(inherit: false))
            {
                AddModelGeneratorCoverage(coveredOperators, attribute.Name);
            }

            foreach (var method in type.GetMethods(ALL_MEMBERS))
            {
                foreach (global::Onnxify.ModelGenerator.TorchSharpOpAttribute attribute in
                    method.GetCustomAttributes<global::Onnxify.ModelGenerator.TorchSharpOpAttribute>(inherit: false))
                {
                    AddModelGeneratorCoverage(coveredOperators, attribute.Name);
                }
            }
        }

        return coveredOperators;
    }

    private static IEnumerable<Type> GetLoadableTypes(Assembly assembly)
    {
        try
        {
            return assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException ex)
        {
            return ex.Types.Where(static type => type is not null)!;
        }
    }

    private static void AddModelGeneratorCoverage(ISet<string> coveredOperators, string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return;
        }

        coveredOperators.Add(name);
        coveredOperators.Add(NormalizeTorchSharpName(name));
    }

    private static bool IsModuleType(Type type)
    {
        for (var current = type; current is not null; current = current.BaseType)
        {
            if (current.FullName?.StartsWith("TorchSharp.torch+nn+Module", StringComparison.Ordinal) == true)
            {
                return true;
            }
        }

        return false;
    }

    private static void AddCandidate(IDictionary<string, TorchSharpCandidate> candidates, string normalizedName, string path)
    {
        if (string.IsNullOrWhiteSpace(normalizedName))
        {
            return;
        }

        if (!candidates.TryGetValue(normalizedName, out TorchSharpCandidate? existing) ||
            string.CompareOrdinal(path, existing.Path) < 0)
        {
            candidates[normalizedName] = new TorchSharpCandidate(normalizedName, path);
        }
    }

    private static ReportRow CreateRow(
        OperatorRecord op,
        IReadOnlyList<TorchSharpCandidate> candidates,
        IReadOnlySet<string> torchSharpCoveredOperators,
        IReadOnlySet<string> modelGeneratorCoveredOperators)
    {
        string normalizedOperator = NormalizeOperatorName(op.Name, op.SourceModule);
        TorchSharpCandidate? match = candidates.FirstOrDefault(candidate =>
            string.Equals(candidate.NormalizedName, normalizedOperator, StringComparison.OrdinalIgnoreCase));
        var modelGeneratorCovered =
            modelGeneratorCoveredOperators.Contains(op.Name) ||
            modelGeneratorCoveredOperators.Contains(normalizedOperator) ||
            (match is not null && modelGeneratorCoveredOperators.Contains(match.NormalizedName));

        return new ReportRow(
            op.Name,
            match?.Path ?? string.Empty,
            match is not null,
            torchSharpCoveredOperators.Contains(op.Name),
            modelGeneratorCovered);
    }

    private static string NormalizeOperatorName(string operatorName, string sourceModule)
    {
        string name = operatorName;

        int separatorIndex = name.IndexOf("::", StringComparison.Ordinal);
        if (separatorIndex >= 0)
        {
            name = name[(separatorIndex + 2)..];
        }

        int overloadSeparator = name.IndexOf('.');
        if (overloadSeparator >= 0)
        {
            name = name[..overloadSeparator];
        }

        if (name.StartsWith('_'))
        {
            name = name[1..];
        }

        name = sourceModule switch
        {
            "fft" when name.StartsWith("fft_", StringComparison.Ordinal) => name["fft_".Length..],
            "linalg" when name.StartsWith("linalg_", StringComparison.Ordinal) => name["linalg_".Length..],
            "special" when name.StartsWith("special_", StringComparison.Ordinal) => name["special_".Length..],
            _ => name
        };

        if (name.StartsWith("aten_", StringComparison.Ordinal))
        {
            name = name["aten_".Length..];
        }

        if (name.StartsWith("torchvision_", StringComparison.Ordinal))
        {
            name = name["torchvision_".Length..];
        }

        if (name.StartsWith("prims_", StringComparison.Ordinal))
        {
            name = name["prims_".Length..];
        }

        return NormalizeTorchSharpName(name);
    }

    private static string NormalizeTorchSharpName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return string.Empty;
        }

        return name
            .Replace("_", string.Empty, StringComparison.Ordinal)
            .Replace(".", string.Empty, StringComparison.Ordinal)
            .ToLowerInvariant();
    }

    private static string BuildMarkdown(IEnumerable<ReportRow> rows)
    {
        ReportRow[] rowArray = rows.ToArray();
        int total = rowArray.Length;
        int foundCount = rowArray.Count(static row => row.Found);
        int torchSharpCoveredCount = rowArray.Count(static row => row.TorchSharpCovered);
        int modelGeneratorCoveredCount = rowArray.Count(static row => row.ModelGeneratorCovered);

        var builder = new StringBuilder();
        builder.AppendLine("# TorchSharp operator coverage");
        builder.AppendLine();
        builder.AppendLine($"Found: {FormatPercentage(foundCount, total)} ({foundCount}/{total})");
        builder.AppendLine($"Onnxify.TorchSharp coverage: {FormatPercentage(torchSharpCoveredCount, total)} ({torchSharpCoveredCount}/{total})");
        builder.AppendLine($"Onnxify.ModelGenerator coverage: {FormatPercentage(modelGeneratorCoveredCount, total)} ({modelGeneratorCoveredCount}/{total})");
        builder.AppendLine();
        builder.AppendLine("## Coverage Columns");
        builder.AppendLine();
        builder.AppendLine("- `Found` means the observer found a likely matching public TorchSharp API or module for the ONNXScript Torch operator name. This is a discovery signal, not an Onnxify implementation guarantee.");
        builder.AppendLine("- `Onnxify.TorchSharp coverage` means `Onnxify.TorchSharp` declares exporter support for that Torch operator through `[TorchOp(...)]`, so TorchSharp code can be exported to ONNX through that converter path.");
        builder.AppendLine("- `Onnxify.ModelGenerator coverage` means `Onnxify.ModelGenerator` declares reverse TorchModule reconstruction support through `[TorchSharpOp(...)]` for the matched TorchSharp API/module name or operator name, so an ONNX graph pattern can be regenerated as a TorchSharp module for that family.");
        builder.AppendLine("- `✅` means the category is covered/found. `❌` means it is not covered/found.");
        builder.AppendLine();
        builder.AppendLine("| ONNXScript operator | TorchSharp module | Found | Onnxify.TorchSharp coverage | Onnxify.ModelGenerator coverage |");
        builder.AppendLine("| --- | --- | --- | --- | --- |");

        foreach (ReportRow row in rowArray)
        {
            builder
                .Append("| ")
                .Append(EscapeMarkdown(row.Operator))
                .Append(" | ")
                .Append(EscapeMarkdown(row.TorchSharpModule))
                .Append(" | ")
                .Append(FormatMarker(row.Found))
                .Append(" | ")
                .Append(FormatMarker(row.TorchSharpCovered))
                .Append(" | ")
                .Append(FormatMarker(row.ModelGeneratorCovered))
                .AppendLine(" |");
        }

        return builder.ToString();
    }

    private static string FormatPercentage(int count, int total)
    {
        if (total == 0)
        {
            return "0.00%";
        }

        return (count * 100.0 / total).ToString("F2", CultureInfo.InvariantCulture) + "%";
    }

    private static string FormatMarker(bool value)
    {
        return value ? "✅" : "❌";
    }

    private static string EscapeMarkdown(string value)
    {
        return value.Replace("|", "\\|", StringComparison.Ordinal);
    }

    [GeneratedRegex(@"@torch_op\((?<args>.*?)\)", RegexOptions.Singleline | RegexOptions.Compiled)]
    private static partial Regex TorchOpAttributeRegex();

    [GeneratedRegex("""(?<quote>['"])(?<value>.*?)(\k<quote>)""", RegexOptions.Singleline | RegexOptions.Compiled)]
    private static partial Regex StringLiteralPattern();

    private sealed record OperatorRecord(string Name, string SourceModule);

    private sealed record TorchSharpCandidate(string NormalizedName, string Path);

    private sealed record ReportRow(
        string Operator,
        string TorchSharpModule,
        bool Found,
        bool TorchSharpCovered,
        bool ModelGeneratorCovered
    );
}
