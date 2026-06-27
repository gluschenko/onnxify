using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;

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
    private static readonly Regex _testMethodRegex = TestMethodPattern();
    private static readonly Regex _wordRegex = WordPattern();
    private static readonly Regex _camelCaseBoundaryRegex = CamelCaseBoundaryPattern();
    private static readonly HashSet<string> _ignoredTestCoverageTokens = new(StringComparer.Ordinal)
    {
        "aten",
        "dim",
        "dims",
        "int",
        "intlist",
        "math",
        "operator",
        "other",
        "prims",
        "scalar",
        "self",
        "tensor",
    };
    private static readonly OperatorMapping[] _operatorMappings =
    [
        new()
        {
            OperatorNames = ["_operator::__lshift__", "aten::__lshift__.Scalar"],
            TorchSharpAliases = ["bitwise_left_shift", "left_shift"],
            DeepImportOnnxOps = ["BitShift"],
        },
        new()
        {
            OperatorNames = ["_operator::__rshift__", "aten::__rshift__.Scalar"],
            TorchSharpAliases = ["bitwise_right_shift", "right_shift"],
            DeepImportOnnxOps = ["BitShift"],
        },
        new()
        {
            OperatorNames = ["_operator::and_"],
            TorchSharpAliases = ["bitwise_and", "logical_and"],
            DeepImportOnnxOps = ["And"],
        },
        new()
        {
            OperatorNames = ["_operator::eq", "aten::eq.Tensor", "aten::eq.Scalar"],
            TestAliases = ["equal"],
            DeepImportOnnxOps = ["Equal"],
        },
        new()
        {
            OperatorNames = ["_operator::ge", "aten::ge.Tensor", "aten::ge.Scalar"],
            TestAliases = ["greater_equal", "greater_or_equal"],
            DeepImportOnnxOps = ["GreaterOrEqual"],
        },
        new()
        {
            OperatorNames = ["_operator::gt", "aten::gt.Tensor", "aten::gt.Scalar"],
            TestAliases = ["greater"],
            DeepImportOnnxOps = ["Greater"],
        },
        new()
        {
            OperatorNames = ["_operator::le", "aten::le.Tensor", "aten::le.Scalar"],
            TestAliases = ["less_equal", "less_or_equal"],
            DeepImportOnnxOps = ["LessOrEqual"],
        },
        new()
        {
            OperatorNames = ["_operator::lt", "aten::lt.Tensor", "aten::lt.Scalar"],
            TestAliases = ["less"],
            DeepImportOnnxOps = ["Less"],
        },
        new()
        {
            OperatorNames = ["_operator::mod"],
            TorchSharpAliases = ["remainder", "fmod"],
            DeepImportOnnxOps = ["Mod"],
        },
        new()
        {
            OperatorNames = ["_operator::mul", "aten::mul.Tensor", "aten::mul.Scalar"],
            TestAliases = ["multiply"],
            DeepImportOnnxOps = ["Mul"],
        },
        new()
        {
            OperatorNames = ["_operator::ne", "aten::ne.Tensor", "aten::ne.Scalar"],
            TestAliases = ["not_equal", "not_equal_to"],
            DeepImportOnnxOps = ["Equal", "Not"],
        },
        new()
        {
            OperatorNames = ["_operator::neg"],
            DeepImportOnnxOps = ["Neg"],
        },
        new()
        {
            OperatorNames = ["_operator::or_"],
            TorchSharpAliases = ["bitwise_or", "logical_or"],
            DeepImportOnnxOps = ["Or"],
        },
        new()
        {
            OperatorNames = ["_operator::pow"],
            DeepImportOnnxOps = ["Pow"],
        },
        new()
        {
            OperatorNames = ["_operator::sub", "aten::sub.Tensor", "aten::sub.Scalar"],
            TestAliases = ["subtract"],
            DeepImportOnnxOps = ["Sub"],
        },
        new()
        {
            OperatorNames = ["_operator::floordiv"],
            TorchSharpAliases = ["floor_divide"],
            DeepImportOnnxOps = ["Div", "Floor", "Sign", "Equal", "Not", "Mod", "Cast", "And", "Sub"],
        },
        new()
        {
            OperatorNames = ["_operator::truediv"],
            TorchSharpAliases = ["true_divide", "div", "divide"],
            DeepImportOnnxOps = ["Div"],
        },
        new()
        {
            OperatorNames = ["aten::_to_copy"],
            TorchSharpAliases = ["to", "to_type", "type_as"],
            DeepImportOnnxOps = ["Cast"],
        },
        new()
        {
            OperatorNames = ["aten::special_softmax"],
            TorchSharpAliases = ["softmax"],
            DeepImportOnnxOps = ["Softmax"],
        },
        new()
        {
            OperatorNames = ["aten::split_with_sizes"],
            TorchSharpAliases = ["split"],
            DeepImportOnnxOps = ["Split"],
        },
        new()
        {
            OperatorNameContains = ["native_batch_norm"],
            TorchSharpAliases = ["batch_norm", "batchnorm", "batchnorm1d", "batchnorm2d", "batchnorm3d"],
            DeepImportOnnxOps = ["BatchNormalization"],
        },
        new()
        {
            OperatorNames = ["aten::native_dropout"],
            TorchSharpAliases = ["dropout"],
            DeepImportOnnxOps = ["Dropout"],
        },
        new()
        {
            OperatorNames = ["aten::native_group_norm"],
            TorchSharpAliases = ["group_norm", "groupnorm"],
            DeepImportOnnxOps = ["GroupNormalization"],
        },
        new()
        {
            OperatorNames = ["aten::native_layer_norm"],
            TorchSharpAliases = ["layer_norm", "layernorm"],
            DeepImportOnnxOps = ["LayerNormalization"],
        },
        new()
        {
            NormalizedPrefixes = ["upsample"],
            TorchSharpAliases = ["upsample", "interpolate"],
            DeepImportOnnxOps = ["Resize"],
        },
        new() { NormalizedNames = ["abs"], DeepImportOnnxOps = ["Abs"] },
        new() { NormalizedNames = ["acos"], DeepImportOnnxOps = ["Acos"] },
        new() { NormalizedNames = ["acosh"], DeepImportOnnxOps = ["Acosh"] },
        new() { NormalizedNames = ["add"], DeepImportOnnxOps = ["Add"] },
        new() { NormalizedNames = ["argmax"], DeepImportOnnxOps = ["ArgMax"] },
        new() { NormalizedNames = ["argmin"], DeepImportOnnxOps = ["ArgMin"] },
        new() { NormalizedNames = ["asin"], DeepImportOnnxOps = ["Asin"] },
        new() { NormalizedNames = ["asinh"], DeepImportOnnxOps = ["Asinh"] },
        new() { NormalizedNames = ["atan"], DeepImportOnnxOps = ["Atan"] },
        new() { NormalizedNames = ["atanh"], DeepImportOnnxOps = ["Atanh"] },
        new() { NormalizedNames = ["ceil"], DeepImportOnnxOps = ["Ceil"] },
        new() { NormalizedNames = ["cat", "concat", "concatenate"], DeepImportOnnxOps = ["Concat"] },
        new() { NormalizedNames = ["celu"], DeepImportOnnxOps = ["Celu"] },
        new() { NormalizedNames = ["clamp"], OperatorNameContains = ["clamp"], DeepImportOnnxOps = ["Clip"] },
        new() { NormalizedNames = ["clampmax"], OperatorNameContains = ["clamp_max"], DeepImportOnnxOps = ["Min"] },
        new() { NormalizedNames = ["clampmin"], OperatorNameContains = ["clamp_min"], DeepImportOnnxOps = ["Max"] },
        new() { NormalizedNames = ["clone", "contiguous", "detach", "alias"], DeepImportOnnxOps = ["Identity"] },
        new() { NormalizedNames = ["cos"], DeepImportOnnxOps = ["Cos"] },
        new() { NormalizedNames = ["cosh"], DeepImportOnnxOps = ["Cosh"] },
        new() { NormalizedNames = ["cumsum"], DeepImportOnnxOps = ["CumSum"] },
        new() { NormalizedNames = ["det", "linalgdet"], DeepImportOnnxOps = ["Det"] },
        new() { NormalizedNames = ["div", "divide", "truediv"], DeepImportOnnxOps = ["Div"] },
        new() { NormalizedNames = ["dropout", "nativedropout"], DeepImportOnnxOps = ["Dropout"] },
        new() { NormalizedNames = ["elu"], DeepImportOnnxOps = ["Elu"] },
        new() { NormalizedNames = ["eq", "equal"], DeepImportOnnxOps = ["Equal"] },
        new() { NormalizedNames = ["erf", "specialerf"], DeepImportOnnxOps = ["Erf"] },
        new() { NormalizedNames = ["exp"], DeepImportOnnxOps = ["Exp"] },
        new() { NormalizedNames = ["expand"], DeepImportOnnxOps = ["Expand"] },
        new() { NormalizedNames = ["expandas", "broadcastto"], DeepImportOnnxOps = ["Expand"] },
        new() { NormalizedPrefixes = ["flatten"], DeepImportOnnxOps = ["Flatten"] },
        new() { NormalizedNames = ["floor"], DeepImportOnnxOps = ["Floor"] },
        new() { NormalizedNames = ["floordiv", "floordivide"], DeepImportOnnxOps = ["Div", "Floor", "Sign", "Equal", "Not", "Mod", "Cast", "And", "Sub"] },
        new() { NormalizedNames = ["gather"], DeepImportOnnxOps = ["Gather"] },
        new() { NormalizedNames = ["gatherelements"], DeepImportOnnxOps = ["GatherElements"] },
        new() { NormalizedNames = ["ge", "greaterequal"], DeepImportOnnxOps = ["GreaterOrEqual"] },
        new() { NormalizedNames = ["gelu"], DeepImportOnnxOps = ["Gelu"] },
        new() { NormalizedNames = ["groupnorm", "nativegroupnorm"], DeepImportOnnxOps = ["GroupNormalization"] },
        new() { NormalizedNames = ["gru"], DeepImportOnnxOps = ["GRU"] },
        new() { NormalizedNames = ["gt", "greater"], DeepImportOnnxOps = ["Greater"] },
        new() { NormalizedNames = ["hardsigmoid"], DeepImportOnnxOps = ["HardSigmoid"] },
        new() { NormalizedNames = ["hardswish"], DeepImportOnnxOps = ["HardSwish"] },
        new() { NormalizedNames = ["isinf"], DeepImportOnnxOps = ["IsInf"] },
        new() { NormalizedNames = ["isnan"], DeepImportOnnxOps = ["IsNaN"] },
        new() { NormalizedNames = ["layernorm", "nativelayernorm"], DeepImportOnnxOps = ["LayerNormalization"] },
        new() { NormalizedNames = ["le", "lessequal"], DeepImportOnnxOps = ["LessOrEqual"] },
        new() { NormalizedNames = ["leakyrelu"], DeepImportOnnxOps = ["LeakyRelu"] },
        new() { NormalizedNames = ["log"], DeepImportOnnxOps = ["Log"] },
        new() { NormalizedNames = ["logsoftmax", "speciallogsoftmax"], DeepImportOnnxOps = ["LogSoftmax"] },
        new() { NormalizedNames = ["logicaland"], DeepImportOnnxOps = ["And"] },
        new() { NormalizedNames = ["logicalnot"], DeepImportOnnxOps = ["Not"] },
        new() { NormalizedNames = ["logicalor"], DeepImportOnnxOps = ["Or"] },
        new() { NormalizedNames = ["logicalxor"], DeepImportOnnxOps = ["Xor"] },
        new() { NormalizedNames = ["lstm"], DeepImportOnnxOps = ["LSTM"] },
        new() { NormalizedNames = ["lt", "less"], DeepImportOnnxOps = ["Less"] },
        new() { NormalizedNames = ["matmul", "bmm", "mm", "mv", "dot"], TestAliases = ["mm", "bmm"], DeepImportOnnxOps = ["MatMul"] },
        new() { NormalizedNames = ["addmm"], DeepImportOnnxOps = ["Gemm"] },
        new() { NormalizedNames = ["max", "maximum"], DeepImportOnnxOps = ["Max"] },
        new() { NormalizedNames = ["amax"], DeepImportOnnxOps = ["ReduceMax"] },
        new() { NormalizedNames = ["mean"], DeepImportOnnxOps = ["ReduceMean"] },
        new() { NormalizedNames = ["min", "minimum"], DeepImportOnnxOps = ["Min"] },
        new() { NormalizedNames = ["amin"], DeepImportOnnxOps = ["ReduceMin"] },
        new() { NormalizedNames = ["mish"], DeepImportOnnxOps = ["Mish"] },
        new() { NormalizedNames = ["mod", "remainder", "fmod"], DeepImportOnnxOps = ["Mod"] },
        new() { NormalizedNames = ["mul", "multiply"], DeepImportOnnxOps = ["Mul"] },
        new() { NormalizedNames = ["ne", "notequal"], DeepImportOnnxOps = ["Equal", "Not"] },
        new() { NormalizedNames = ["neg"], DeepImportOnnxOps = ["Neg"] },
        new() { NormalizedNames = ["nonzero"], DeepImportOnnxOps = ["NonZero"] },
        new() { NormalizedNames = ["pad"], DeepImportOnnxOps = ["Pad"] },
        new() { NormalizedNames = ["pow"], DeepImportOnnxOps = ["Pow"] },
        new() { NormalizedNames = ["prelu"], DeepImportOnnxOps = ["PRelu"] },
        new() { NormalizedNames = ["prod"], DeepImportOnnxOps = ["ReduceProd"] },
        new() { NormalizedNames = ["reciprocal"], DeepImportOnnxOps = ["Reciprocal"] },
        new() { NormalizedNames = ["relu"], DeepImportOnnxOps = ["Relu"] },
        new() { NormalizedNames = ["relu6"], DeepImportOnnxOps = ["Clip"] },
        new() { NormalizedNames = ["reshape", "view"], DeepImportOnnxOps = ["Reshape"] },
        new() { NormalizedNames = ["round"], DeepImportOnnxOps = ["Round"] },
        new() { NormalizedNames = ["selu"], DeepImportOnnxOps = ["Selu"] },
        new() { NormalizedNames = ["sigmoid"], DeepImportOnnxOps = ["Sigmoid"] },
        new() { NormalizedNames = ["sign"], DeepImportOnnxOps = ["Sign"] },
        new() { NormalizedNames = ["sin"], DeepImportOnnxOps = ["Sin"] },
        new() { NormalizedNames = ["sinh"], DeepImportOnnxOps = ["Sinh"] },
        new() { NormalizedNames = ["slice"], DeepImportOnnxOps = ["Slice"] },
        new() { NormalizedNames = ["softmax", "specialsoftmax"], DeepImportOnnxOps = ["Softmax"] },
        new() { NormalizedNames = ["softplus"], DeepImportOnnxOps = ["Softplus"] },
        new() { NormalizedNames = ["split", "splitwithsizes"], DeepImportOnnxOps = ["Split"] },
        new() { NormalizedNames = ["sqrt"], DeepImportOnnxOps = ["Sqrt"] },
        new() { NormalizedNames = ["squeeze"], DeepImportOnnxOps = ["Squeeze"] },
        new() { NormalizedNames = ["sub", "subtract"], DeepImportOnnxOps = ["Sub"] },
        new() { NormalizedNames = ["sum"], DeepImportOnnxOps = ["ReduceSum"] },
        new() { NormalizedNames = ["tan"], DeepImportOnnxOps = ["Tan"] },
        new() { NormalizedNames = ["tanh"], DeepImportOnnxOps = ["Tanh"] },
        new() { NormalizedNames = ["t"], DeepImportOnnxOps = ["Transpose"] },
        new() { NormalizedNames = ["tile"], DeepImportOnnxOps = ["Tile"] },
        new() { NormalizedNames = ["topk"], DeepImportOnnxOps = ["TopK"] },
        new() { NormalizedNames = ["transpose", "permute"], DeepImportOnnxOps = ["Transpose"] },
        new() { NormalizedNames = ["tril", "triu"], DeepImportOnnxOps = ["Trilu"] },
        new() { NormalizedNames = ["truedivide"], DeepImportOnnxOps = ["Div"] },
        new() { NormalizedNames = ["typeas"], DeepImportOnnxOps = ["CastLike"] },
        new() { NormalizedNames = ["unsqueeze"], DeepImportOnnxOps = ["Unsqueeze"] },
        new() { NormalizedNames = ["viewas"], DeepImportOnnxOps = ["Reshape"] },
        new() { NormalizedNames = ["where"], DeepImportOnnxOps = ["Where"] },
        new() { NormalizedNames = ["bitwisenot"], DeepImportOnnxOps = ["BitwiseNot"] },
        new() { NormalizedNames = ["bitwiseand"], DeepImportOnnxOps = ["BitwiseAnd"] },
        new() { NormalizedNames = ["bitwiseor"], DeepImportOnnxOps = ["BitwiseOr"] },
        new() { NormalizedNames = ["bitwisexor"], DeepImportOnnxOps = ["BitwiseXor"] },
        new() { NormalizedNames = ["bitwiseleftshift"], DeepImportOnnxOps = ["BitShift"] },
        new() { NormalizedNames = ["bitwiserightshift"], DeepImportOnnxOps = ["BitShift"] },
        new() { TorchSharpPathContains = ["Conv"], DeepImportOnnxOps = ["Conv"] },
        new() { TorchSharpPathContains = ["BatchNorm"], DeepImportOnnxOps = ["BatchNormalization"] },
        new() { NormalizedNames = ["linear"], TorchSharpPathContains = ["Linear"], DeepImportOnnxOps = ["Gemm"] },
        new() { TorchSharpPathContains = ["MaxPool"], DeepImportOnnxOps = ["MaxPool"] },
        new() { TorchSharpPathContains = ["AvgPool", "AdaptiveAvgPool"], DeepImportOnnxOps = ["AveragePool"] },
    ];

    private static int Main()
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

        var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "TORCH_OPERATOR_COVERAGE.md");

        var operators = LoadOperators(opsDirectory);
        var candidates = LoadTorchSharpCandidates();
        var torchSharpCoveredOperators = LoadTorchSharpCoveredOperators();
        var modelGeneratorCoveredOperators = LoadModelGeneratorCoveredOperators();
        var deepImportSupportedOnnxOps = LoadDeepImportSupportedOnnxOps();
        var testMethods = LoadOnnxifyTestMethods(repoRoot);
        var packageReferences = LoadOnnxifyPackageReferences(repoRoot);

        var rows = operators
            .Select(op => CreateRow(
                op,
                candidates,
                torchSharpCoveredOperators,
                modelGeneratorCoveredOperators,
                deepImportSupportedOnnxOps,
                testMethods
            ))
            .OrderBy(row => row.Operator, StringComparer.Ordinal)
            .ToArray();

        var markdown = BuildMarkdown(rows, packageReferences);

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

    private static IReadOnlySet<string> LoadDeepImportSupportedOnnxOps()
    {
        var supportedOps = new HashSet<string>(StringComparer.Ordinal);
        AddRegistryKeys(
            supportedOps,
            "Onnxify.ModelGenerator.Services.TorchModuleInlineOperators.TorchModuleInlineOperatorRegistry"
        );
        AddRegistryKeys(
            supportedOps,
            "Onnxify.ModelGenerator.Services.TorchModuleOperators.TorchModuleOperatorRegistry"
        );
        return supportedOps;
    }

    private static void AddRegistryKeys(
        ISet<string> supportedOps,
        string registryTypeName
    )
    {
        var assembly = typeof(global::Onnxify.ModelGenerator.TorchSharpOpAttribute).Assembly;
        var registryType = assembly.GetType(registryTypeName)
            ?? throw new InvalidOperationException($"Could not find deep import registry '{registryTypeName}'.");

        var createMethod = registryType.GetMethod("Create", BindingFlags.Static | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Could not find Create() on '{registryTypeName}'.");

        var registry = createMethod.Invoke(null, null)
            ?? throw new InvalidOperationException($"Create() on '{registryTypeName}' returned null.");

        if (registry is not System.Collections.IEnumerable enumerable)
        {
            throw new InvalidOperationException(
                $"Unsupported deep import registry object type '{registry.GetType().FullName}'."
            );
        }

        foreach (var entry in enumerable)
        {
            var keyProperty = entry.GetType().GetProperty("Key")
                ?? throw new InvalidOperationException(
                    $"Unsupported deep import registry entry type '{entry.GetType().FullName}'."
                );

            if (keyProperty.GetValue(entry) is string key)
            {
                supportedOps.Add(key);
            }
        }
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
        IReadOnlySet<string> modelGeneratorCoveredOperators,
        IReadOnlySet<string> deepImportSupportedOnnxOps,
        IReadOnlyList<TestMethodRecord> testMethods)
    {
        string normalizedOperator = NormalizeOperatorName(op.Name, op.SourceModule);
        TorchSharpCandidate? match = FindTorchSharpCandidate(op, normalizedOperator, candidates);
        var deepImportSupported = IsDeepImportSupported(op, normalizedOperator, match, deepImportSupportedOnnxOps);
        var modelGeneratorCovered = IsModelGeneratorCovered(
            op,
            normalizedOperator,
            match,
            modelGeneratorCoveredOperators,
            deepImportSupported);

        return new ReportRow(
            op.Name,
            match?.Path ?? string.Empty,
            match is not null,
            torchSharpCoveredOperators.Contains(op.Name),
            IsDeepExportSupported(op, torchSharpCoveredOperators),
            modelGeneratorCovered,
            deepImportSupported,
            CountCoveringTests(op, normalizedOperator, match, testMethods));
    }

    private static bool IsModelGeneratorCovered(
        OperatorRecord op,
        string normalizedOperator,
        TorchSharpCandidate? match,
        IReadOnlySet<string> modelGeneratorCoveredOperators,
        bool deepImportSupported
    )
    {
        return deepImportSupported
            || modelGeneratorCoveredOperators.Contains(op.Name)
            || modelGeneratorCoveredOperators.Contains(normalizedOperator)
            || GetTorchSharpAliases(op.Name, normalizedOperator).Any(modelGeneratorCoveredOperators.Contains)
            || GetExpectedDeepImportOnnxOps(op, normalizedOperator, match).Any(modelGeneratorCoveredOperators.Contains)
            || (match is not null && modelGeneratorCoveredOperators.Contains(match.NormalizedName));
    }

    private static bool IsDeepExportSupported(
        OperatorRecord op,
        IReadOnlySet<string> torchSharpCoveredOperators
    )
    {
        return torchSharpCoveredOperators.Contains(op.Name);
    }

    private static bool IsDeepImportSupported(
        OperatorRecord op,
        string normalizedOperator,
        TorchSharpCandidate? match,
        IReadOnlySet<string> supportedOnnxOps
    )
    {
        var expectedOps = GetExpectedDeepImportOnnxOps(op, normalizedOperator, match).ToArray();
        return expectedOps.Length > 0 && expectedOps.All(supportedOnnxOps.Contains);
    }

    private static IEnumerable<string> GetExpectedDeepImportOnnxOps(
        OperatorRecord op,
        string normalizedOperator,
        TorchSharpCandidate? match
    )
    {
        return GetMatchingOperatorMappings(op.Name, normalizedOperator, match)
            .SelectMany(static mapping => mapping.DeepImportOnnxOps)
            .Distinct(StringComparer.Ordinal);
    }

    private static TorchSharpCandidate? FindTorchSharpCandidate(
        OperatorRecord op,
        string normalizedOperator,
        IReadOnlyList<TorchSharpCandidate> candidates)
    {
        foreach (var candidateName in GetTorchSharpCandidateNames(op, normalizedOperator))
        {
            var normalizedCandidateName = NormalizeTorchSharpName(candidateName);
            var match = candidates.FirstOrDefault(candidate =>
                string.Equals(candidate.NormalizedName, normalizedCandidateName, StringComparison.OrdinalIgnoreCase));

            if (match is not null)
            {
                return match;
            }
        }

        return null;
    }

    private static IEnumerable<string> GetTorchSharpCandidateNames(OperatorRecord op, string normalizedOperator)
    {
        yield return normalizedOperator;

        foreach (var alias in GetTorchSharpAliases(op.Name, normalizedOperator))
        {
            yield return alias;
        }
    }

    private static IEnumerable<string> GetTorchSharpAliases(string operatorName, string normalizedOperator)
    {
        return GetMatchingOperatorMappings(operatorName, normalizedOperator, match: null)
            .SelectMany(static mapping => mapping.TorchSharpAliases)
            .Prepend(normalizedOperator)
            .Distinct(StringComparer.Ordinal);
    }

    private static IEnumerable<OperatorMapping> GetMatchingOperatorMappings(
        string operatorName,
        string normalizedOperator,
        TorchSharpCandidate? match
    )
    {
        return _operatorMappings.Where(mapping => MatchesOperatorMapping(
            mapping,
            operatorName,
            normalizedOperator,
            match
        ));
    }

    private static bool MatchesOperatorMapping(
        OperatorMapping mapping,
        string operatorName,
        string normalizedOperator,
        TorchSharpCandidate? match
    )
    {
        return mapping.OperatorNames.Contains(operatorName, StringComparer.Ordinal)
            || mapping.NormalizedNames.Contains(normalizedOperator, StringComparer.Ordinal)
            || mapping.OperatorNameContains.Any(value => operatorName.Contains(value, StringComparison.Ordinal))
            || mapping.NormalizedPrefixes.Any(value => normalizedOperator.StartsWith(value, StringComparison.Ordinal))
            || (match is not null && mapping.NormalizedNames.Contains(match.NormalizedName, StringComparer.Ordinal))
            || (match is not null && mapping.TorchSharpPathContains.Any(value => match.Path.Contains(value, StringComparison.Ordinal)));
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

    private static IReadOnlyList<TestMethodRecord> LoadOnnxifyTestMethods(string repoRoot)
    {
        var testsDirectory = Path.Combine(repoRoot, "src", "Onnxify.Tests");
        if (!Directory.Exists(testsDirectory))
        {
            return [];
        }

        var tests = new List<TestMethodRecord>();
        foreach (var path in Directory.EnumerateFiles(testsDirectory, "*.cs", SearchOption.AllDirectories))
        {
            var content = File.ReadAllText(path);
            foreach (Match match in _testMethodRegex.Matches(content))
            {
                var bodyStart = content.IndexOf('{', match.Index + match.Length - 1);
                if (bodyStart < 0 || !TryFindMatchingBrace(content, bodyStart, out var bodyEnd))
                {
                    continue;
                }

                var name = match.Groups["name"].Value;
                var body = content[bodyStart..(bodyEnd + 1)];
                tests.Add(CreateTestMethodRecord(name, body));
            }
        }

        return tests;
    }

    private static TestMethodRecord CreateTestMethodRecord(string name, string body)
    {
        var tokens = new HashSet<string>(StringComparer.Ordinal);
        foreach (var token in Tokenize($"{name} {body}"))
        {
            tokens.Add(token);
        }

        var nameTokens = new HashSet<string>(Tokenize(name), StringComparer.Ordinal);
        var compactText = string.Concat(tokens.Order(StringComparer.Ordinal));
        return new TestMethodRecord(
            name,
            nameTokens,
            tokens,
            string.Concat(nameTokens.Order(StringComparer.Ordinal)),
            compactText,
            NormalizeSearchText(name),
            NormalizeSearchText($"{name} {body}")
        );
    }

    private static int CountCoveringTests(
        OperatorRecord op,
        string normalizedOperator,
        TorchSharpCandidate? match,
        IReadOnlyList<TestMethodRecord> testMethods
    )
    {
        var terms = GetTestCoverageTerms(op, normalizedOperator, match).ToArray();
        return testMethods.Count(test => terms.Any(term => TestMentionsTerm(test, term)));
    }

    private static IEnumerable<TestCoverageTerm> GetTestCoverageTerms(
        OperatorRecord op,
        string normalizedOperator,
        TorchSharpCandidate? match
    )
    {
        foreach (var term in CreateCoverageTerms(op.Name, allowBodySubstring: true))
        {
            yield return term;
        }

        foreach (var term in CreateCoverageTerms(normalizedOperator, allowBodySubstring: false))
        {
            yield return term;
        }

        if (match is not null)
        {
            foreach (var term in CreateCoverageTerms(match.NormalizedName, allowBodySubstring: false))
            {
                yield return term;
            }
        }

        foreach (var candidateName in GetTorchSharpCandidateNames(op, normalizedOperator))
        {
            foreach (var term in CreateCoverageTerms(candidateName, allowBodySubstring: false))
            {
                yield return term;
            }
        }

        foreach (var alias in GetTestCoverageAliases(op.Name, normalizedOperator))
        {
            foreach (var term in CreateCoverageTerms(alias, allowBodySubstring: false))
            {
                yield return term;
            }
        }
    }

    private static IEnumerable<TestCoverageTerm> CreateCoverageTerms(string value, bool allowBodySubstring)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            yield break;
        }

        var normalized = NormalizeSearchText(value);
        if (normalized.Length >= 4)
        {
            yield return new TestCoverageTerm(normalized, ExactToken: false, AllowBodySubstring: allowBodySubstring);
        }

        foreach (var token in Tokenize(value))
        {
            if (token.Length >= 2 && !_ignoredTestCoverageTokens.Contains(token))
            {
                yield return new TestCoverageTerm(token, ExactToken: token.Length <= 3, AllowBodySubstring: false);
            }
        }
    }

    private static IEnumerable<string> GetTestCoverageAliases(string operatorName, string normalizedOperator)
    {
        return GetMatchingOperatorMappings(operatorName, normalizedOperator, match: null)
            .SelectMany(static mapping => mapping.TestAliases)
            .Distinct(StringComparer.Ordinal);
    }

    private static bool TestMentionsTerm(TestMethodRecord test, TestCoverageTerm term)
    {
        if (term.ExactToken)
        {
            return test.NameTokens.Contains(term.Value)
                || test.RawText.Contains($"export{term.Value}", StringComparison.Ordinal);
        }

        return test.NameRawText.Contains(term.Value, StringComparison.Ordinal)
            || test.CompactNameTokenText.Contains(term.Value, StringComparison.Ordinal)
            || test.RawText.Contains($"export{term.Value}", StringComparison.Ordinal)
            || (term.AllowBodySubstring && test.RawText.Contains(term.Value, StringComparison.Ordinal))
            || (term.AllowBodySubstring && test.CompactTokenText.Contains(term.Value, StringComparison.Ordinal));
    }

    private static IEnumerable<string> Tokenize(string value)
    {
        foreach (Match match in _wordRegex.Matches(value))
        {
            var word = _camelCaseBoundaryRegex.Replace(match.Value, " ");
            foreach (var part in word.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                yield return part.ToLowerInvariant();
            }
        }
    }

    private static string NormalizeSearchText(string value)
    {
        return string.Concat(
            value
                .Where(static character => char.IsLetterOrDigit(character))
                .Select(static character => char.ToLowerInvariant(character))
        );
    }

    private static bool TryFindMatchingBrace(string content, int openBraceIndex, out int closeBraceIndex)
    {
        var depth = 0;
        for (var index = openBraceIndex; index < content.Length; index++)
        {
            if (content[index] == '{')
            {
                depth++;
            }
            else if (content[index] == '}')
            {
                depth--;
                if (depth == 0)
                {
                    closeBraceIndex = index;
                    return true;
                }
            }
        }

        closeBraceIndex = -1;
        return false;
    }

    private static IReadOnlyList<OnnxifyPackageReference> LoadOnnxifyPackageReferences(string repoRoot)
    {
        var srcDirectory = Path.Combine(repoRoot, "src");
        var projectPaths = Directory
            .EnumerateFiles(srcDirectory, "*.csproj", SearchOption.AllDirectories)
            .OrderBy(static path => path, StringComparer.Ordinal)
            .ToArray();

        var projects = projectPaths
            .Select(LoadProjectMetadata)
            .ToArray();

        var publishableOnnxifyProjects = projects
            .Where(static project => project.PackageId.StartsWith("Onnxify.", StringComparison.Ordinal)
                || string.Equals(project.PackageId, "Onnxify", StringComparison.Ordinal))
            .Where(static project => project.IsPackable)
            .Where(static project => !string.IsNullOrWhiteSpace(project.Version))
            .ToArray();

        var projectPathToPackageId = publishableOnnxifyProjects.ToDictionary(
            static project => project.Path,
            static project => project.PackageId,
            StringComparer.OrdinalIgnoreCase
        );

        return publishableOnnxifyProjects
            .Select(project => new OnnxifyPackageReference(
                project.PackageId,
                project.Version!,
                project.ProjectReferences
                    .Select(reference => Path.GetFullPath(Path.Combine(project.Directory, reference.Include)))
                    .Where(projectPathToPackageId.ContainsKey)
                    .Select(path => projectPathToPackageId[path])
                    .OrderBy(static packageId => packageId, StringComparer.Ordinal)
                    .ToArray(),
                project.PackageReferences
                    .Where(static reference => !reference.Include.StartsWith("Onnxify.", StringComparison.Ordinal)
                        && !string.Equals(reference.Include, "Onnxify", StringComparison.Ordinal))
                    .OrderBy(static reference => reference.Include, StringComparer.Ordinal)
                    .ThenBy(static reference => reference.Condition, StringComparer.Ordinal)
                    .ToArray()
            ))
            .OrderBy(static package => package.PackageId, StringComparer.Ordinal)
            .ToArray();
    }

    private static ProjectMetadata LoadProjectMetadata(string projectPath)
    {
        var document = XDocument.Load(projectPath, LoadOptions.PreserveWhitespace);
        var root = document.Root ?? throw new InvalidDataException($"Project file has no root element: {projectPath}");
        var packageId = GetProperty(root, "PackageId") ?? Path.GetFileNameWithoutExtension(projectPath);
        var isPackableText = GetProperty(root, "IsPackable");
        var isPackable = isPackableText is null || !string.Equals(isPackableText, "false", StringComparison.OrdinalIgnoreCase);

        return new ProjectMetadata(
            Path.GetFullPath(projectPath),
            Path.GetDirectoryName(Path.GetFullPath(projectPath))!,
            packageId,
            GetProperty(root, "Version"),
            isPackable,
            root
                .Descendants("PackageReference")
                .Select(static element => new PackageReference(
                    GetRequiredAttribute(element, "Include"),
                    GetAttribute(element, "Version") ?? GetChildValue(element, "Version") ?? string.Empty,
                    GetAttribute(element, "Condition") ?? string.Empty
                ))
                .Where(static reference => !string.IsNullOrWhiteSpace(reference.Include))
                .ToArray(),
            root
                .Descendants("ProjectReference")
                .Select(static element => new ProjectReference(
                    GetRequiredAttribute(element, "Include")
                ))
                .Where(static reference => !string.IsNullOrWhiteSpace(reference.Include))
                .ToArray()
        );
    }

    private static string? GetProperty(XElement root, string name)
    {
        return root
            .Descendants(name)
            .Select(static element => element.Value.Trim())
            .FirstOrDefault(static value => !string.IsNullOrWhiteSpace(value));
    }

    private static string? GetChildValue(XElement element, string name)
    {
        return element
            .Elements(name)
            .Select(static child => child.Value.Trim())
            .FirstOrDefault(static value => !string.IsNullOrWhiteSpace(value));
    }

    private static string? GetAttribute(XElement element, string name)
    {
        return element.Attribute(name)?.Value.Trim();
    }

    private static string GetRequiredAttribute(XElement element, string name)
    {
        return GetAttribute(element, name) ?? string.Empty;
    }

    private static string BuildMarkdown(
        IEnumerable<ReportRow> rows,
        IReadOnlyList<OnnxifyPackageReference> packageReferences
    )
    {
        ReportRow[] rowArray = rows.ToArray();
        int total = rowArray.Length;
        int foundCount = rowArray.Count(static row => row.Found);
        int torchSharpCoveredCount = rowArray.Count(static row => row.TorchSharpCovered);
        int modelGeneratorCoveredCount = rowArray.Count(static row => row.ModelGeneratorCovered);
        int deepExportSupportedCount = rowArray.Count(static row => row.DeepExportSupported);
        int deepImportSupportedCount = rowArray.Count(static row => row.DeepImportSupported);

        var builder = new StringBuilder();
        builder.AppendLine("# TorchSharp operator coverage");
        builder.AppendLine();
        builder.AppendLine($"* Found: {FormatPercentage(foundCount, total)} ({foundCount}/{total})");
        builder.AppendLine($"* Onnxify.TorchSharp coverage: {FormatPercentage(torchSharpCoveredCount, total)} ({torchSharpCoveredCount}/{total})");
        builder.AppendLine($"* Onnxify.ModelGenerator coverage: {FormatPercentage(modelGeneratorCoveredCount, total)} ({modelGeneratorCoveredCount}/{total})");
        builder.AppendLine($"* Deep export support: {FormatPercentage(deepExportSupportedCount, total)} ({deepExportSupportedCount}/{total})");
        builder.AppendLine($"* Deep import support: {FormatPercentage(deepImportSupportedCount, total)} ({deepImportSupportedCount}/{total})");
        builder.AppendLine();
        builder.AppendLine("## Package Versions");
        builder.AppendLine();
        builder.AppendLine("Current versions and direct dependencies are read from the publishable `Onnxify.*` project files under `src/`.");
        builder.AppendLine();

        foreach (var package in packageReferences)
        {
            builder.AppendLine($"### `{EscapeMarkdown(package.PackageId)}`");
            builder.AppendLine();
            builder.AppendLine($"* Version: `{EscapeMarkdown(package.Version)}`");
            builder.AppendLine("* Onnxify project references:");
            AppendOnnxifyPackageDependencies(builder, package.OnnxifyPackageDependencies);
            builder.AppendLine("* Third-party NuGet PackageReferences:");
            AppendThirdPartyPackageReferences(builder, package.ThirdPartyPackageReferences);
            builder.AppendLine();
        }

        builder.AppendLine();
        builder.AppendLine("## Coverage Columns");
        builder.AppendLine();
        builder.AppendLine("* `Found` means the observer found a likely matching public TorchSharp API or module for the ONNXScript Torch operator name. This is a discovery signal, not an Onnxify implementation guarantee.");
        builder.AppendLine("* `Onnxify.TorchSharp coverage` means `Onnxify.TorchSharp` declares exporter support for that Torch operator through `[TorchOp(...)]`, so TorchSharp code can be exported to ONNX through that converter path.");
        builder.AppendLine("* `Onnxify.ModelGenerator coverage` means `Onnxify.ModelGenerator` declares reverse TorchModule reconstruction support through `[TorchSharpOp(...)]` or the shared canonical ONNX mapping resolves to actual deep-import registry support for that operator family.");
        builder.AppendLine("* `Deep export support` means the exact ONNXScript Torch operator is registered in the actual `Onnxify.TorchSharp` deep-export coverage set through `[TorchOp(...)]`.");
        builder.AppendLine("* `Deep import support` means the observer can map the ONNXScript Torch operator to expected ONNX `OpType` nodes and every mapped `OpType` is registered in the actual `Onnxify.ModelGenerator` TorchModule deep-import registries.");
        builder.AppendLine("* `Onnxify.Tests tests` is the number of `[Fact]` / `[Theory]` test methods in `src/Onnxify.Tests` whose name or body mentions the ONNXScript operator, normalized TorchSharp API name, or a known operator alias.");
        builder.AppendLine("* `✅` means the category is covered/found. `❌` means it is not covered/found.");
        builder.AppendLine();
        builder.AppendLine("| ONNXScript operator | TorchSharp module | Found | Onnxify.TorchSharp coverage | Onnxify.ModelGenerator coverage | Deep export support | Deep import support | Onnxify.Tests tests |");
        builder.AppendLine("| --- | --- | --- | --- | --- | --- | --- | --- |");

        foreach (ReportRow row in rowArray)
        {
            builder
                .Append("| ")
                .Append($"`{EscapeMarkdown(row.Operator)}`")
                .Append(" | ")
                .Append(!string.IsNullOrWhiteSpace(row.TorchSharpModule) ? $"`{EscapeMarkdown(row.TorchSharpModule)}`" : string.Empty)
                .Append(" | ")
                .Append(FormatMarker(row.Found))
                .Append(" | ")
                .Append(FormatMarker(row.TorchSharpCovered))
                .Append(" | ")
                .Append(FormatMarker(row.ModelGeneratorCovered))
                .Append(" | ")
                .Append(FormatMarker(row.DeepExportSupported))
                .Append(" | ")
                .Append(FormatMarker(row.DeepImportSupported))
                .Append(" | ")
                .Append(row.OnnxifyTestsCount.ToString(CultureInfo.InvariantCulture))
                .AppendLine(" |");
        }

        return builder.ToString();
    }

    private static void AppendOnnxifyPackageDependencies(StringBuilder builder, IReadOnlyList<string> packageIds)
    {
        if (packageIds.Count == 0)
        {
            builder.AppendLine("  * None");
            return;
        }

        foreach (var packageId in packageIds)
        {
            builder.AppendLine($"  * `{EscapeMarkdown(packageId)}`");
        }
    }

    private static void AppendThirdPartyPackageReferences(StringBuilder builder, IReadOnlyList<PackageReference> packageReferences)
    {
        if (packageReferences.Count == 0)
        {
            builder.AppendLine("  * None");
            return;
        }

        foreach (var reference in packageReferences)
        {
            var version = string.IsNullOrWhiteSpace(reference.Version) ? "no explicit version" : reference.Version;
            var condition = string.IsNullOrWhiteSpace(reference.Condition)
                ? string.Empty
                : $" ({reference.Condition})";

            builder.AppendLine($"  * `{EscapeMarkdown(reference.Include)}` `{EscapeMarkdown(version)}`{EscapeMarkdown(condition)}");
        }
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

    [GeneratedRegex(@"(?s)\[(?:Fact|Theory)(?:\([^\]]*\))?\]\s*(?:\[[^\]]+\]\s*)*(?:public|internal|private)\s+(?:async\s+)?(?:[\w<>,\.\?\[\]\s]+)\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{", RegexOptions.Compiled)]
    private static partial Regex TestMethodPattern();

    [GeneratedRegex(@"[A-Za-z0-9_]+", RegexOptions.Compiled)]
    private static partial Regex WordPattern();

    [GeneratedRegex(@"(?<=[a-z0-9])(?=[A-Z])|_", RegexOptions.Compiled)]
    private static partial Regex CamelCaseBoundaryPattern();

    private sealed record OperatorRecord(string Name, string SourceModule);

    private sealed record TorchSharpCandidate(string NormalizedName, string Path);

    private sealed record ProjectMetadata(
        string Path,
        string Directory,
        string PackageId,
        string? Version,
        bool IsPackable,
        IReadOnlyList<PackageReference> PackageReferences,
        IReadOnlyList<ProjectReference> ProjectReferences
    );

    private sealed record PackageReference(
        string Include,
        string Version,
        string Condition
    );

    private sealed record ProjectReference(string Include);

    private sealed record OnnxifyPackageReference(
        string PackageId,
        string Version,
        IReadOnlyList<string> OnnxifyPackageDependencies,
        IReadOnlyList<PackageReference> ThirdPartyPackageReferences
    );

    private sealed record OperatorMapping
    {
        public string[] OperatorNames { get; init; } = [];

        public string[] NormalizedNames { get; init; } = [];

        public string[] TorchSharpAliases { get; init; } = [];

        public string[] TestAliases { get; init; } = [];

        public string[] DeepImportOnnxOps { get; init; } = [];

        public string[] OperatorNameContains { get; init; } = [];

        public string[] NormalizedPrefixes { get; init; } = [];

        public string[] TorchSharpPathContains { get; init; } = [];
    }

    private sealed record TestMethodRecord(
        string Name,
        IReadOnlySet<string> NameTokens,
        IReadOnlySet<string> Tokens,
        string CompactNameTokenText,
        string CompactTokenText,
        string NameRawText,
        string RawText
    );

    private readonly record struct TestCoverageTerm(
        string Value,
        bool ExactToken,
        bool AllowBodySubstring
    );

    private sealed record ReportRow(
        string Operator,
        string TorchSharpModule,
        bool Found,
        bool TorchSharpCovered,
        bool DeepExportSupported,
        bool ModelGeneratorCovered,
        bool DeepImportSupported,
        int OnnxifyTestsCount
    );
}
