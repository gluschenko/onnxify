using System.Collections.Frozen;
using System.Globalization;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Onnxify.TorchSharp;

namespace Onnxify.AgentSkillGenerator;

internal static class OperatorSkillGenerator
{
    private const string NewLine = "\n";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private static readonly IReadOnlyDictionary<short, OpCode> SingleByteOpCodes = CreateOpCodeMap(multiByte: false);
    private static readonly IReadOnlyDictionary<short, OpCode> MultiByteOpCodes = CreateOpCodeMap(multiByte: true);
    private static readonly FrozenDictionary<string, string> AliasFieldNames = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["Inputs"] = "In",
        ["Outputs"] = "Out",
    }.ToFrozenDictionary(StringComparer.Ordinal);

    public static int Run(string[] args)
    {
        string repoRoot = FindRepositoryRoot(Directory.GetCurrentDirectory())
            ?? FindRepositoryRoot(AppContext.BaseDirectory)
            ?? throw new DirectoryNotFoundException("Repository root was not found.");

        string schemaPath = ResolveSchemaPath(repoRoot);
        string skillRoot = ResolveSkillRoot(repoRoot);
        string outputRoot = Path.Combine(skillRoot, "references", "operators");

        var schemaRoot = JsonSerializer.Deserialize<OperatorSchemaRoot>(File.ReadAllText(schemaPath), JsonOptions)
            ?? throw new InvalidOperationException($"Failed to parse operator schema file '{schemaPath}'.");

        var schemaByKey = schemaRoot.Operators.ToDictionary(
            x => new OperatorKey(x.Domain ?? string.Empty, x.Name),
            x => x);

        var onnxOperators = BuildOnnxOperators(schemaByKey);
        var converterCoverage = BuildTorchSharpCoverage(onnxOperators);
        var generatedFiles = BuildGeneratedFiles(onnxOperators, converterCoverage);

        RewriteGeneratedDirectory(outputRoot, generatedFiles);

        Console.WriteLine($"Repository root: {MakeRelative(repoRoot, repoRoot)}");
        Console.WriteLine($"Skill root: {MakeRelative(repoRoot, skillRoot)}");
        Console.WriteLine($"Schema source: {MakeRelative(repoRoot, schemaPath)}");
        Console.WriteLine($"Generated operator files: {generatedFiles.Count - 1}");
        Console.WriteLine($"Index file: {MakeRelative(repoRoot, Path.Combine(outputRoot, "index.md"))}");
        Console.WriteLine(
            $"TorchSharp-covered operators: {onnxOperators.Count(x => converterCoverage.ContainsKey(x.Key))}/{onnxOperators.Count}");

        return 0;
    }

    private static string ResolveSchemaPath(string repoRoot)
    {
        string copiedAssetPath = Path.Combine(AppContext.BaseDirectory, "Assets", "onnx_operators.json");
        if (File.Exists(copiedAssetPath))
        {
            return copiedAssetPath;
        }

        string repoAssetPath = Path.Combine(repoRoot, "src", "Onnxify", "Assets", "onnx_operators.json");
        if (File.Exists(repoAssetPath))
        {
            return repoAssetPath;
        }

        throw new FileNotFoundException("onnx_operators.json was not found in the generator output or repository.");
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

    private static IReadOnlyList<OnnxOperatorDoc> BuildOnnxOperators(
        IReadOnlyDictionary<OperatorKey, OperatorSchema> schemaByKey)
    {
        Assembly assembly = typeof(OnnxModel).Assembly;

        var extensionMethods = SafeGetTypes(assembly)
            .Where(type => string.Equals(type.Name, "OnnxifyExtensions", StringComparison.Ordinal))
            .SelectMany(type => type.GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly))
            .ToArray();

        return SafeGetTypes(assembly)
            .Where(type =>
                type.IsPublic &&
                !type.IsAbstract &&
                type != typeof(OnnxNode) &&
                typeof(OnnxNode).IsAssignableFrom(type))
            .Select(type => CreateOnnxOperatorDoc(type, extensionMethods, schemaByKey))
            .Where(x => x is not null)
            .Cast<OnnxOperatorDoc>()
            .OrderBy(x => x.Key.Domain, StringComparer.Ordinal)
            .ThenBy(x => x.Key.Name, StringComparer.Ordinal)
            .ToArray();
    }

    private static OnnxOperatorDoc? CreateOnnxOperatorDoc(
        Type nodeType,
        IReadOnlyList<MethodInfo> extensionMethods,
        IReadOnlyDictionary<OperatorKey, OperatorSchema> schemaByKey)
    {
        string? domain = NamespaceToDomain(nodeType.Namespace);
        if (domain is null)
        {
            return null;
        }

        var key = new OperatorKey(domain, nodeType.Name);
        schemaByKey.TryGetValue(key, out OperatorSchema? schema);

        Type? inputOptionsType = nodeType.Assembly.GetType($"{nodeType.Namespace}.{nodeType.Name}InputOptions");
        Type? inputOutputOptionsType = nodeType.Assembly.GetType($"{nodeType.Namespace}.{nodeType.Name}InputOutputOptions");

        var wrapperMethods = extensionMethods
            .Where(method =>
                string.Equals(method.Name, nodeType.Name, StringComparison.Ordinal) &&
                method.GetParameters().Length == 3 &&
                (method.GetParameters()[2].ParameterType == inputOptionsType ||
                 method.GetParameters()[2].ParameterType == inputOutputOptionsType))
            .OrderBy(method => GetMethodDisplaySignature(method), StringComparer.Ordinal)
            .Select(method => new WrapperMethodDoc(method, GetWrapperDisplaySignature(method)))
            .ToArray();

        var inputs = schema is null || inputOptionsType is null
            ? Array.Empty<ParameterDoc>()
            : schema.Inputs
                .Select(parameter => CreateParameterDoc(nodeType.Name, schema, parameter, inputOptionsType, ParameterKind.Input))
                .ToArray();

        var outputs = schema is null || inputOutputOptionsType is null
            ? Array.Empty<ParameterDoc>()
            : schema.Outputs
                .Select(parameter => CreateParameterDoc(nodeType.Name, schema, parameter, inputOutputOptionsType, ParameterKind.Output))
                .ToArray();

        var attributes = schema is null || inputOptionsType is null
            ? Array.Empty<AttributeDoc>()
            : schema.Attributes
                .Select(attribute => CreateAttributeDoc(nodeType.Name, schema, attribute, inputOptionsType))
                .ToArray();

        return new OnnxOperatorDoc(
            key,
            schema,
            nodeType,
            inputOptionsType,
            inputOutputOptionsType,
            wrapperMethods,
            inputs,
            outputs,
            attributes);
    }

    private static ParameterDoc CreateParameterDoc(
        string operatorName,
        OperatorSchema schema,
        OperatorParameterSchema parameter,
        Type optionsType,
        ParameterKind kind)
    {
        string propertyName = kind switch
        {
            ParameterKind.Input => GetInputPropertyName(operatorName, schema, parameter.Name),
            ParameterKind.Output => GetOutputPropertyName(operatorName, schema, parameter.Name),
            _ => throw new InvalidOperationException($"Unsupported parameter kind '{kind}'."),
        };

        PropertyInfo? property = optionsType.GetProperty(propertyName, BindingFlags.Instance | BindingFlags.Public);

        return new ParameterDoc(
            parameter.Name,
            propertyName,
            property is null ? string.Empty : GetFriendlyTypeName(property.PropertyType),
            parameter.Option,
            parameter.MinArity,
            CollapseWhitespace(parameter.Description),
            property is not null && IsRequiredMember(property));
    }

    private static AttributeDoc CreateAttributeDoc(
        string operatorName,
        OperatorSchema schema,
        OperatorAttributeSchema attribute,
        Type optionsType)
    {
        string propertyName = GetAttributePropertyName(operatorName, schema, attribute.Name);
        PropertyInfo? property = optionsType.GetProperty(propertyName, BindingFlags.Instance | BindingFlags.Public);

        return new AttributeDoc(
            attribute.Name,
            propertyName,
            property is null ? string.Empty : GetFriendlyTypeName(property.PropertyType),
            CollapseWhitespace(attribute.Description),
            attribute.Required || (property is not null && IsRequiredMember(property)),
            FormatDefaultValue(attribute.Default));
    }

    private static IReadOnlyDictionary<OperatorKey, IReadOnlyList<TorchSharpConverterDoc>> BuildTorchSharpCoverage(
        IReadOnlyList<OnnxOperatorDoc> onnxOperators)
    {
        Assembly torchSharpAssembly = typeof(TorchOpAttribute).Assembly;

        var operatorConstructorsByType = onnxOperators.ToDictionary(
            x => x.NodeType.FullName ?? x.NodeType.Name,
            x => x.Key,
            StringComparer.Ordinal);

        var wrapperMethodsByHandle = onnxOperators
            .SelectMany(x => x.WrapperMethods.Select(method => new KeyValuePair<string, OperatorKey>(GetMethodKey(method.Method), x.Key)))
            .ToDictionary(x => x.Key, x => x.Value, StringComparer.Ordinal);

        MethodAnalysis[] analyses = SafeGetTypes(torchSharpAssembly)
            .SelectMany(type => type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            .Where(method => method.GetMethodBody() is not null)
            .Select(method => AnalyzeMethod(method, torchSharpAssembly, operatorConstructorsByType, wrapperMethodsByHandle))
            .ToArray();

        var analysesByKey = analyses.ToDictionary(x => x.MethodKey, StringComparer.Ordinal);
        var cache = new Dictionary<string, FrozenSet<OperatorKey>>(StringComparer.Ordinal);

        FrozenSet<OperatorKey> ResolveOperators(string methodKey, HashSet<string> visiting)
        {
            if (cache.TryGetValue(methodKey, out FrozenSet<OperatorKey>? cached))
            {
                return cached;
            }

            if (!analysesByKey.TryGetValue(methodKey, out MethodAnalysis? analysis))
            {
                return FrozenSet<OperatorKey>.Empty;
            }

            if (!visiting.Add(methodKey))
            {
                return FrozenSet<OperatorKey>.Empty;
            }

            var builder = new HashSet<OperatorKey>(analysis.DirectOperators);
            foreach (string internalCall in analysis.InternalCalls)
            {
                builder.UnionWith(ResolveOperators(internalCall, visiting));
            }

            visiting.Remove(methodKey);

            FrozenSet<OperatorKey> result = builder.ToFrozenSet();
            cache[methodKey] = result;
            return result;
        }

        var coverage = new Dictionary<OperatorKey, List<TorchSharpConverterDoc>>();

        foreach (MethodAnalysis analysis in analyses.Where(x => x.IsTopLevelConverter))
        {
            FrozenSet<OperatorKey> emittedOperators = ResolveOperators(analysis.MethodKey, new HashSet<string>(StringComparer.Ordinal));
            if (emittedOperators.Count == 0)
            {
                continue;
            }

            var converter = new TorchSharpConverterDoc(
                GetConverterDisplaySignature(analysis.Method),
                analysis.TorchOps.OrderBy(x => x, StringComparer.Ordinal).ToArray());

            foreach (OperatorKey key in emittedOperators.OrderBy(x => x.Domain, StringComparer.Ordinal).ThenBy(x => x.Name, StringComparer.Ordinal))
            {
                if (!coverage.TryGetValue(key, out List<TorchSharpConverterDoc>? list))
                {
                    list = [];
                    coverage.Add(key, list);
                }

                list.Add(converter);
            }
        }

        return coverage.ToDictionary(
            pair => pair.Key,
            pair => (IReadOnlyList<TorchSharpConverterDoc>)pair.Value
                .Distinct()
                .OrderBy(x => x.Signature, StringComparer.Ordinal)
                .ThenBy(x => string.Join(",", x.TorchOps), StringComparer.Ordinal)
                .ToArray());
    }

    private static MethodAnalysis AnalyzeMethod(
        MethodBase method,
        Assembly currentAssembly,
        IReadOnlyDictionary<string, OperatorKey> operatorConstructorsByType,
        IReadOnlyDictionary<string, OperatorKey> wrapperMethodsByHandle)
    {
        var directOperators = new HashSet<OperatorKey>();
        var internalCalls = new HashSet<string>(StringComparer.Ordinal);

        foreach (ResolvedCall call in ReadMethodCalls(method))
        {
            if (call.Target is null)
            {
                continue;
            }

            if (call.Target.DeclaringType?.Assembly == currentAssembly)
            {
                internalCalls.Add(GetMethodKey(call.Target));
            }

            if (wrapperMethodsByHandle.TryGetValue(GetMethodKey(call.Target), out OperatorKey? wrapperOperator))
            {
                directOperators.Add(wrapperOperator);
            }

            if (call.Target is ConstructorInfo constructor &&
                constructor.DeclaringType?.FullName is string fullName &&
                operatorConstructorsByType.TryGetValue(fullName, out OperatorKey? constructedOperator))
            {
                directOperators.Add(constructedOperator);
            }
        }

        string[] torchOps = method
            .GetCustomAttributes<TorchOpAttribute>(inherit: false)
            .Select(attribute => attribute.Name)
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToArray();

        return new MethodAnalysis(
            method,
            GetMethodKey(method),
            directOperators.ToFrozenSet(),
            internalCalls.ToFrozenSet(StringComparer.Ordinal),
            torchOps,
            IsTopLevelConverter(method));
    }

    private static IReadOnlyDictionary<string, string> BuildGeneratedFiles(
        IReadOnlyList<OnnxOperatorDoc> onnxOperators,
        IReadOnlyDictionary<OperatorKey, IReadOnlyList<TorchSharpConverterDoc>> converterCoverage)
    {
        var files = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (OnnxOperatorDoc op in onnxOperators)
        {
            string relativePath = Path.Combine(GetDomainDirectoryName(op.Key.Domain), $"{op.Key.Name}.md");
            files[relativePath] = BuildOperatorMarkdown(op, converterCoverage.GetValueOrDefault(op.Key, Array.Empty<TorchSharpConverterDoc>()));
        }

        files["index.md"] = BuildIndexMarkdown(onnxOperators, converterCoverage);
        return files;
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

    private static string BuildIndexMarkdown(
        IReadOnlyList<OnnxOperatorDoc> onnxOperators,
        IReadOnlyDictionary<OperatorKey, IReadOnlyList<TorchSharpConverterDoc>> converterCoverage)
    {
        var builder = new StringBuilder();
        builder.AppendLine("# Onnxify Operator Instructions");
        builder.AppendLine();
        builder.AppendLine("Autogenerated by `src/Onnxify.AgentSkillGenerator`.");
        builder.AppendLine("Do not hand-edit files in this directory; rerun the generator instead.");
        builder.AppendLine();
        builder.AppendLine($"- Total reflected Onnxify operators: `{onnxOperators.Count}`");
        builder.AppendLine($"- Operators with at least one Onnxify.TorchSharp converter path: `{onnxOperators.Count(x => converterCoverage.ContainsKey(x.Key))}`");
        builder.AppendLine("- Operator schema source: `src/Onnxify/Assets/onnx_operators.json`");
        builder.AppendLine();

        foreach (IGrouping<string, OnnxOperatorDoc> domainGroup in onnxOperators.GroupBy(x => x.Key.Domain).OrderBy(x => x.Key, StringComparer.Ordinal))
        {
            builder.AppendLine($"## {FormatDomain(domainGroup.Key)}");
            builder.AppendLine();
            builder.AppendLine("| Operator | Since | TorchSharp converter | File |");
            builder.AppendLine("| --- | --- | --- | --- |");

            foreach (OnnxOperatorDoc op in domainGroup.OrderBy(x => x.Key.Name, StringComparer.Ordinal))
            {
                string relativePath = Path.Combine(GetDomainDirectoryName(op.Key.Domain), $"{op.Key.Name}.md").Replace('\\', '/');
                string sinceVersion = op.Schema?.SinceVersion.ToString(CultureInfo.InvariantCulture) ?? "?";
                string converterStatus = converterCoverage.ContainsKey(op.Key) ? "yes" : "no";

                builder.Append("| ")
                    .Append(EscapeMarkdownCell(op.Key.Name))
                    .Append(" | ")
                    .Append(EscapeMarkdownCell(sinceVersion))
                    .Append(" | ")
                    .Append(converterStatus)
                    .Append(" | [")
                    .Append(EscapeMarkdownCell(op.Key.Name))
                    .Append("](")
                    .Append(relativePath)
                    .AppendLine(") |");
            }

            builder.AppendLine();
        }

        return builder.ToString().TrimEnd();
    }

    private static string BuildOperatorMarkdown(
        OnnxOperatorDoc op,
        IReadOnlyList<TorchSharpConverterDoc> converters)
    {
        var builder = new StringBuilder();
        builder.Append("# ").AppendLine(op.Key.Name);
        builder.AppendLine();
        builder.AppendLine("[Back to operator index](../index.md)");
        builder.AppendLine();
        builder.AppendLine("Autogenerated by `src/Onnxify.AgentSkillGenerator`.");
        builder.AppendLine();
        builder.AppendLine($"- Domain: `{FormatDomain(op.Key.Domain)}`");
        builder.AppendLine($"- Onnxify node type: `{GetFriendlyTypeName(op.NodeType)}`");
        builder.AppendLine($"- Since version: `{op.Schema?.SinceVersion.ToString(CultureInfo.InvariantCulture) ?? "?"}`");
        builder.AppendLine($"- Onnxify.TorchSharp converter coverage: `{(converters.Count > 0 ? "available" : "not detected")}`");
        builder.AppendLine();

        builder.AppendLine("## Description");
        builder.AppendLine();
        if (string.IsNullOrWhiteSpace(op.Schema?.Doc))
        {
            builder.AppendLine("Schema description was not found in `onnx_operators.json`.");
        }
        else
        {
            builder.AppendLine(NormalizeMarkdownBlock(op.Schema.Doc));
        }

        builder.AppendLine();
        builder.AppendLine("## Onnxify Surface");
        builder.AppendLine();
        builder.AppendLine($"- Input options type: `{(op.InputOptionsType is null ? "[not found]" : GetFriendlyTypeName(op.InputOptionsType))}`");
        builder.AppendLine($"- Input/output options type: `{(op.InputOutputOptionsType is null ? "[not found]" : GetFriendlyTypeName(op.InputOutputOptionsType))}`");

        if (op.WrapperMethods.Count == 0)
        {
            builder.AppendLine("- Wrapper overloads detected via reflection: `[none]`");
        }
        else
        {
            builder.AppendLine("- Wrapper overloads detected via reflection:");
            foreach (WrapperMethodDoc method in op.WrapperMethods)
            {
                builder.Append("  - `").Append(method.DisplaySignature).AppendLine("`");
            }
        }

        builder.AppendLine();
        builder.AppendLine("## Inputs");
        builder.AppendLine();
        builder.AppendLine("| JSON name | Onnxify property | Type | Semantics | Description |");
        builder.AppendLine("| --- | --- | --- | --- | --- |");
        if (op.Schema is null || op.Inputs.Count == 0)
        {
            builder.AppendLine("| [schema missing] |  |  |  |  |");
        }
        else
        {
            foreach (ParameterDoc input in op.Inputs)
            {
                builder.Append("| `")
                    .Append(EscapeMarkdownCell(input.SchemaName))
                    .Append("` | `")
                    .Append(EscapeMarkdownCell(input.PropertyName))
                    .Append("` | `")
                    .Append(EscapeMarkdownCell(input.PropertyType))
                    .Append("` | ")
                    .Append(EscapeMarkdownCell(FormatParameterSemantics(input.Option, input.MinArity, input.Required)))
                    .Append(" | ")
                    .Append(EscapeMarkdownCell(input.Description))
                    .AppendLine(" |");
            }
        }

        builder.AppendLine();
        builder.AppendLine("## Outputs");
        builder.AppendLine();
        builder.AppendLine("| JSON name | Onnxify property | Type | Semantics | Description |");
        builder.AppendLine("| --- | --- | --- | --- | --- |");
        if (op.Schema is null || op.Outputs.Count == 0)
        {
            builder.AppendLine("| [schema missing] |  |  |  |  |");
        }
        else
        {
            foreach (ParameterDoc output in op.Outputs)
            {
                builder.Append("| `")
                    .Append(EscapeMarkdownCell(output.SchemaName))
                    .Append("` | `")
                    .Append(EscapeMarkdownCell(output.PropertyName))
                    .Append("` | `")
                    .Append(EscapeMarkdownCell(output.PropertyType))
                    .Append("` | ")
                    .Append(EscapeMarkdownCell(FormatParameterSemantics(output.Option, output.MinArity, output.Required)))
                    .Append(" | ")
                    .Append(EscapeMarkdownCell(output.Description))
                    .AppendLine(" |");
            }
        }

        builder.AppendLine();
        builder.AppendLine("## Attributes");
        builder.AppendLine();
        builder.AppendLine("| JSON name | Onnxify property | Type | Required | Default | Description |");
        builder.AppendLine("| --- | --- | --- | --- | --- | --- |");
        if (op.Schema is null)
        {
            builder.AppendLine("| [schema missing] |  |  |  |  |  |");
        }
        else if (op.Attributes.Count == 0)
        {
            builder.AppendLine("| [none] |  |  |  |  |  |");
        }
        else
        {
            foreach (AttributeDoc attribute in op.Attributes)
            {
                builder.Append("| `")
                    .Append(EscapeMarkdownCell(attribute.SchemaName))
                    .Append("` | `")
                    .Append(EscapeMarkdownCell(attribute.PropertyName))
                    .Append("` | `")
                    .Append(EscapeMarkdownCell(attribute.PropertyType))
                    .Append("` | ")
                    .Append(attribute.Required ? "yes" : "no")
                    .Append(" | `")
                    .Append(EscapeMarkdownCell(attribute.DefaultValue))
                    .Append("` | ")
                    .Append(EscapeMarkdownCell(attribute.Description))
                    .AppendLine(" |");
            }
        }

        builder.AppendLine();
        builder.AppendLine("## TorchSharp Coverage");
        builder.AppendLine();
        if (converters.Count == 0)
        {
            builder.AppendLine("No `Onnxify.TorchSharp` converter path that emits this operator was detected via reflection and IL analysis.");
        }
        else
        {
            builder.AppendLine("| Converter | Torch ops |");
            builder.AppendLine("| --- | --- |");
            foreach (TorchSharpConverterDoc converter in converters)
            {
                string torchOps = converter.TorchOps.Length == 0
                    ? "[none declared]"
                    : string.Join(", ", converter.TorchOps.Select(x => $"`{x}`"));

                builder.Append("| `")
                    .Append(EscapeMarkdownCell(converter.Signature))
                    .Append("` | ")
                    .Append(torchOps)
                    .AppendLine(" |");
            }
        }

        return builder.ToString().TrimEnd();
    }

    private static IEnumerable<ResolvedCall> ReadMethodCalls(MethodBase method)
    {
        MethodBody? body = method.GetMethodBody();
        if (body is null)
        {
            yield break;
        }

        byte[]? il = body.GetILAsByteArray();
        if (il is null || il.Length == 0)
        {
            yield break;
        }

        Type[]? typeArguments = method.DeclaringType?.GetGenericArguments();
        Type[]? methodArguments = method is MethodInfo info ? info.GetGenericArguments() : null;

        int offset = 0;
        while (offset < il.Length)
        {
            OpCode opCode = ReadOpCode(il, ref offset);
            object? operand = ReadOperand(method.Module, il, ref offset, opCode.OperandType, typeArguments, methodArguments);

            if ((opCode == OpCodes.Call || opCode == OpCodes.Callvirt || opCode == OpCodes.Newobj) && operand is MethodBase target)
            {
                yield return new ResolvedCall(opCode, target);
            }
        }
    }

    private static OpCode ReadOpCode(byte[] il, ref int offset)
    {
        byte first = il[offset++];
        if (first != 0xFE)
        {
            return SingleByteOpCodes[(short)first];
        }

        byte second = il[offset++];
        short key = unchecked((short)(0xFE00 | second));
        return MultiByteOpCodes[key];
    }

    private static object? ReadOperand(
        Module module,
        byte[] il,
        ref int offset,
        OperandType operandType,
        Type[]? typeArguments,
        Type[]? methodArguments)
    {
        switch (operandType)
        {
            case OperandType.InlineNone:
                return null;
            case OperandType.ShortInlineBrTarget:
            case OperandType.ShortInlineI:
            case OperandType.ShortInlineVar:
                offset += 1;
                return null;
            case OperandType.InlineVar:
                offset += 2;
                return null;
            case OperandType.InlineI:
            case OperandType.InlineBrTarget:
            case OperandType.InlineField:
            case OperandType.InlineSig:
            case OperandType.InlineString:
            case OperandType.InlineType:
            case OperandType.ShortInlineR:
            case OperandType.InlineTok:
                offset += 4;
                return null;
            case OperandType.InlineI8:
            case OperandType.InlineR:
                offset += 8;
                return null;
            case OperandType.InlineSwitch:
            {
                int count = BitConverter.ToInt32(il, offset);
                offset += 4 + (count * 4);
                return null;
            }
            case OperandType.InlineMethod:
            {
                int token = BitConverter.ToInt32(il, offset);
                offset += 4;

                try
                {
                    return module.ResolveMethod(token, typeArguments, methodArguments);
                }
                catch
                {
                    return null;
                }
            }
            default:
                throw new NotSupportedException($"Unsupported IL operand type '{operandType}'.");
        }
    }

    private static IReadOnlyDictionary<short, OpCode> CreateOpCodeMap(bool multiByte)
    {
        var map = new Dictionary<short, OpCode>();

        foreach (FieldInfo field in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (field.GetValue(null) is not OpCode opCode)
            {
                continue;
            }

            bool isMultiByte = (opCode.Value & 0xFF00) == 0xFE00;
            if (isMultiByte == multiByte)
            {
                map[opCode.Value] = opCode;
            }
        }

        return map;
    }

    private static bool IsTopLevelConverter(MethodBase method)
    {
        if (!method.IsPublic || method.Name != "Export")
        {
            return false;
        }

        if (!method.IsDefined(typeof(ExtensionAttribute), inherit: false))
        {
            return false;
        }

        ParameterInfo[] parameters = method.GetParameters();
        if (parameters.Length == 0)
        {
            return false;
        }

        if (parameters[0].ParameterType.Assembly != typeof(global::TorchSharp.torch).Assembly)
        {
            return false;
        }

        return method.GetCustomAttributes<TorchOpAttribute>(inherit: false).Any();
    }

    private static bool IsRequiredMember(PropertyInfo property)
    {
        return property.CustomAttributes.Any(attribute =>
            string.Equals(attribute.AttributeType.FullName, typeof(RequiredMemberAttribute).FullName, StringComparison.Ordinal));
    }

    private static string GetMethodKey(MethodBase method)
    {
        return $"{method.Module.ModuleVersionId:D}:{method.MetadataToken.ToString(CultureInfo.InvariantCulture)}";
    }

    private static string GetWrapperDisplaySignature(MethodInfo method)
    {
        ParameterInfo[] parameters = method.GetParameters();
        string receiver = GetReceiverExpression(parameters[0].ParameterType);
        string arguments = string.Join(", ", parameters.Skip(1).Select(parameter =>
            $"{GetFriendlyTypeName(parameter.ParameterType)} {parameter.Name}"));

        return $"{receiver}.{method.Name}({arguments}) -> {GetFriendlyTypeName(method.ReturnType)}";
    }

    private static string GetConverterDisplaySignature(MethodBase method)
    {
        string declaringType = method.DeclaringType?.FullName?.Replace("+", ".", StringComparison.Ordinal)
            ?? "[unknown]";

        string parameters = string.Join(", ", method.GetParameters().Select((parameter, index) =>
        {
            string prefix = index == 0 && method.IsDefined(typeof(ExtensionAttribute), inherit: false) ? "this " : string.Empty;
            return $"{prefix}{GetFriendlyTypeName(parameter.ParameterType)} {parameter.Name}";
        }));

        string returnType = method is MethodInfo methodInfo ? GetFriendlyTypeName(methodInfo.ReturnType) : "void";
        return $"{declaringType}.{method.Name}({parameters}) -> {returnType}";
    }

    private static string GetMethodDisplaySignature(MethodBase method)
    {
        return $"{method.DeclaringType?.FullName}:{method}";
    }

    private static string GetReceiverExpression(Type type)
    {
        return type.Name switch
        {
            nameof(OnnxGraph) => "graph",
            nameof(MLDomain) => "graph.ML",
            nameof(MicrosoftDomain) => "graph.Microsoft",
            nameof(MicrosoftInternalDomain) => "graph.Microsoft.Internal",
            nameof(MicrosoftNHWCDomain) => "graph.Microsoft.NHWC",
            nameof(MicrosoftNCHWcDomain) => "graph.Microsoft.NCHWc",
            nameof(MicrosoftInternalNHWCDomain) => "graph.Microsoft.Internal.NHWC",
            _ => type.Name,
        };
    }

    private static IReadOnlyList<Type> SafeGetTypes(Assembly assembly)
    {
        try
        {
            return assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException exception)
        {
            return exception.Types.Where(type => type is not null).Cast<Type>().ToArray();
        }
    }

    private static string? NamespaceToDomain(string? @namespace)
    {
        return @namespace switch
        {
            "Onnxify" => string.Empty,
            "Onnxify.ML" => "ai.onnx.ml",
            "Onnxify.Microsoft" => "com.microsoft",
            "Onnxify.Microsoft.NHWC" => "com.microsoft.nhwc",
            "Onnxify.Microsoft.NCHWc" => "com.microsoft.nchwc",
            "Onnxify.Microsoft.Internal.NHWC" => "com.ms.internal.nhwc",
            _ => null,
        };
    }

    private static string GetDomainDirectoryName(string domain)
    {
        return string.IsNullOrWhiteSpace(domain) ? "ai.onnx" : domain;
    }

    private static string FormatDomain(string domain)
    {
        return string.IsNullOrWhiteSpace(domain) ? "ai.onnx" : domain;
    }

    private static string GetInputPropertyName(string operatorName, OperatorSchema schema, string parameterName)
    {
        return GetFieldName(operatorName, schema, parameterName, "Input");
    }

    private static string GetOutputPropertyName(string operatorName, OperatorSchema schema, string parameterName)
    {
        return GetFieldName(operatorName, schema, parameterName, "Output");
    }

    private static string GetAttributePropertyName(string operatorName, OperatorSchema schema, string attributeName)
    {
        return GetFieldName(operatorName, schema, attributeName, "Attribute");
    }

    private static string GetFieldName(string operatorName, OperatorSchema schema, string name, string prefix)
    {
        var reservedFieldNames = new HashSet<string>(StringComparer.Ordinal)
        {
            operatorName,
            "Inputs",
            "Outputs",
        };

        if (schema.Inputs.Any(x => x.Name == "shape") && schema.Attributes.Any(x => x.Name == "shape"))
        {
            reservedFieldNames.Add("Shape");
        }

        string fieldName = PascalCase(name);
        if (AliasFieldNames.TryGetValue(fieldName, out string? alias))
        {
            fieldName = alias;
        }

        if (reservedFieldNames.Contains(fieldName))
        {
            return prefix + fieldName;
        }

        if (!string.IsNullOrEmpty(prefix) && fieldName.Equals(prefix, StringComparison.OrdinalIgnoreCase))
        {
            return prefix;
        }

        return fieldName;
    }

    private static string PascalCase(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        string[] words = value.Split(['_'], StringSplitOptions.RemoveEmptyEntries);
        var builder = new StringBuilder();

        foreach (string word in words)
        {
            if (word.Length == 0)
            {
                continue;
            }

            builder.Append(char.ToUpperInvariant(word[0]));
            if (word.Length > 1)
            {
                builder.Append(word[1..]);
            }
        }

        return builder.ToString();
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
            string genericTypeName = type.Name;
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

    private static string FormatParameterSemantics(FormalParameterOption option, int minArity, bool required)
    {
        return option switch
        {
            FormalParameterOption.Single => required ? "single, required" : "single",
            FormalParameterOption.Optional => "optional",
            FormalParameterOption.Variadic => $"variadic, min arity {minArity.ToString(CultureInfo.InvariantCulture)}",
            _ => option.ToString(),
        };
    }

    private static string FormatDefaultValue(object? value)
    {
        if (value is null)
        {
            return "[null]";
        }

        if (value is JsonElement element)
        {
            return element.ValueKind switch
            {
                JsonValueKind.Array => "[" + string.Join(", ", element.EnumerateArray().Select(item => FormatDefaultValue(item))) + "]",
                JsonValueKind.String => element.GetString() ?? string.Empty,
                JsonValueKind.Number => element.ToString(),
                JsonValueKind.True => "true",
                JsonValueKind.False => "false",
                JsonValueKind.Null => "[null]",
                _ => element.ToString(),
            };
        }

        return Convert.ToString(value, CultureInfo.InvariantCulture) ?? value.ToString() ?? string.Empty;
    }

    private static string CollapseWhitespace(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        var builder = new StringBuilder(value.Length);
        bool previousWasWhitespace = false;

        foreach (char ch in value)
        {
            bool isWhitespace = char.IsWhiteSpace(ch);
            if (isWhitespace)
            {
                if (!previousWasWhitespace)
                {
                    builder.Append(' ');
                }

                previousWasWhitespace = true;
                continue;
            }

            builder.Append(ch);
            previousWasWhitespace = false;
        }

        return builder.ToString().Trim();
    }

    private static string NormalizeMarkdownBlock(string value)
    {
        string normalized = NormalizeLineEndings(value).Trim();
        return string.IsNullOrWhiteSpace(normalized) ? string.Empty : normalized;
    }

    private static string NormalizeLineEndings(string value)
    {
        return value.Replace("\r\n", NewLine, StringComparison.Ordinal).Replace("\r", NewLine, StringComparison.Ordinal);
    }

    private static string EscapeMarkdownCell(string value)
    {
        return value
            .Replace("|", "\\|", StringComparison.Ordinal)
            .Replace("\r", " ", StringComparison.Ordinal)
            .Replace("\n", " ", StringComparison.Ordinal);
    }

    private static string MakeRelative(string repoRoot, string path)
    {
        if (string.Equals(repoRoot, path, StringComparison.OrdinalIgnoreCase))
        {
            return ".";
        }

        return Path.GetRelativePath(repoRoot, path).Replace('\\', '/');
    }

    private sealed record OperatorKey(string Domain, string Name);

    private sealed record WrapperMethodDoc(MethodInfo Method, string DisplaySignature);

    private sealed record ParameterDoc(
        string SchemaName,
        string PropertyName,
        string PropertyType,
        FormalParameterOption Option,
        int MinArity,
        string Description,
        bool Required);

    private sealed record AttributeDoc(
        string SchemaName,
        string PropertyName,
        string PropertyType,
        string Description,
        bool Required,
        string DefaultValue);

    private sealed record OnnxOperatorDoc(
        OperatorKey Key,
        OperatorSchema? Schema,
        Type NodeType,
        Type? InputOptionsType,
        Type? InputOutputOptionsType,
        IReadOnlyList<WrapperMethodDoc> WrapperMethods,
        IReadOnlyList<ParameterDoc> Inputs,
        IReadOnlyList<ParameterDoc> Outputs,
        IReadOnlyList<AttributeDoc> Attributes);

    private sealed record TorchSharpConverterDoc(string Signature, string[] TorchOps);

    private sealed record MethodAnalysis(
        MethodBase Method,
        string MethodKey,
        FrozenSet<OperatorKey> DirectOperators,
        FrozenSet<string> InternalCalls,
        string[] TorchOps,
        bool IsTopLevelConverter);

    private sealed record ResolvedCall(OpCode OpCode, MethodBase? Target);

    private enum ParameterKind
    {
        Input,
        Output,
    }
}
