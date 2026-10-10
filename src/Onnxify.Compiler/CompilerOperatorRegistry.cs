using Onnxify.Compiler.Operators;

namespace Onnxify.Compiler;

internal static class CompilerOperatorRegistry
{
    private static readonly CompilerOperator[] _operators =
    [
        new AbsOperator(),
        new AddOperator(),
        new AcosOperator(),
        new AcoshOperator(),
        new AsinOperator(),
        new AsinhOperator(),
        new AtanOperator(),
        new AtanhOperator(),
        new CeilOperator(),
        new CastOperator(),
        new ClipOperator(),
        new ConvOperator(),
        new CosOperator(),
        new CoshOperator(),
        new ExpOperator(),
        new FloorOperator(),
        new FlattenOperator(),
        new GlobalAveragePoolOperator(),
        new LogOperator(),
        new ModOperator(),
        new MatMulOperator(),
        new GemmOperator(),
        new NegOperator(),
        new SubOperator(),
        new SinOperator(),
        new SinhOperator(),
        new SqrtOperator(),
        new TanOperator(),
        new MulOperator(),
        new DivOperator(),
        new EqualOperator(),
        new PowOperator(),
        new GreaterOperator(),
        new GreaterOrEqualOperator(),
        new LessOperator(),
        new LessOrEqualOperator(),
        new MaxOperator(),
        new MinOperator(),
        new AndOperator(),
        new OrOperator(),
        new XorOperator(),
        new NotOperator(),
        new WhereOperator(),
        new ReciprocalOperator(),
        new RoundOperator(),
        new SignOperator(),
        new TruncOperator(),
        new ErfOperator(),
        new IsNaNOperator(),
        new CeluOperator(),
        new EluOperator(),
        new GeluOperator(),
        new HardSigmoidOperator(),
        new HardSwishOperator(),
        new LeakyReluOperator(),
        new MishOperator(),
        new PReluOperator(),
        new ReluOperator(),
        new SeluOperator(),
        new SigmoidOperator(),
        new SoftplusOperator(),
        new SoftsignOperator(),
        new SwishOperator(),
        new TanhOperator(),
        new ThresholdedReluOperator(),
    ];

    private static readonly IReadOnlyDictionary<CompilerOperatorIdentity, CompilerOperator> _onnx = BuildOnnxIndex(_operators);
    private static readonly IReadOnlyDictionary<Type, CompilerOperator> _nodeTypes = BuildNodeTypeIndex(_operators);
    private static readonly IReadOnlyDictionary<string, CompilerOperator> _calls = BuildCallIndex();
    private static readonly IReadOnlyDictionary<string, CompilerOperator> _binary = BuildFormIndex(CompilerTorchSharpFormKind.Binary);
    private static readonly IReadOnlyDictionary<string, CompilerOperator> _unary = BuildFormIndex(CompilerTorchSharpFormKind.Unary);

    public static bool TryGetOnnx(string domain, string name, out CompilerOperator? compilerOperator)
    {
        return _onnx.TryGetValue(CompilerOperatorIdentity.Onnx(name, NormalizeDomain(domain)), out compilerOperator);
    }

    public static bool TryGetOnnx(OnnxNode node, out CompilerOperator? compilerOperator)
    {
        var identity = CompilerOperatorIdentity.Onnx(node.OpType, NormalizeDomain(node.Domain));
        if (_nodeTypes.TryGetValue(node.GetType(), out compilerOperator)
            && compilerOperator.Identity.Equals(identity)
            && compilerOperator.Accepts(node))
        {
            return true;
        }

        if (TryGetOnnx(node.Domain, node.OpType, out compilerOperator)
            && compilerOperator is not null
            && compilerOperator.Accepts(node))
        {
            return true;
        }

        compilerOperator = null;
        return false;
    }

    public static bool TryGetTorchSharp(string name, out CompilerOperator? compilerOperator)
    {
        return _calls.TryGetValue(name, out compilerOperator);
    }

    public static bool TryGetTorchSharpBinaryOperator(string token, out CompilerOperator? compilerOperator)
    {
        return _binary.TryGetValue(token, out compilerOperator);
    }

    public static bool TryGetTorchSharpUnaryOperator(string token, out CompilerOperator? compilerOperator)
    {
        return _unary.TryGetValue(token, out compilerOperator);
    }

    public static bool TryGetTorchSharpCall(CompilerExpression target, out CompilerOperator? compilerOperator, out CompilerExpression? receiver)
    {
        receiver = null;
        if (TryGetMemberPath(target, out var name) && TryGetTorchSharp(name, out compilerOperator))
        {
            return true;
        }

        compilerOperator = null;
        if (target is CompilerMemberAccessExpression member
            && member.Target is CompilerReferenceExpression
            && TryGetTorchSharp($"Tensor.{member.MemberName}", out compilerOperator))
        {
            receiver = member.Target;
            return true;
        }

        return false;
    }

    public static string GetTorchSharpCallName(CompilerExpression target, CompilerExpression? receiver)
    {
        return receiver is not null && target is CompilerMemberAccessExpression member
            ? $"Tensor.{member.MemberName}"
            : TryGetMemberPath(target, out var name) ? name : string.Empty;
    }

    internal static IReadOnlyDictionary<CompilerOperatorIdentity, CompilerOperator> BuildOnnxIndex(IEnumerable<CompilerOperator> operators)
    {
        return BuildIndex(operators, static compilerOperator => compilerOperator.Identity, "ONNX identity");
    }

    internal static IReadOnlyDictionary<Type, CompilerOperator> BuildNodeTypeIndex(IEnumerable<CompilerOperator> operators)
    {
        var index = new Dictionary<Type, CompilerOperator>();
        foreach (var compilerOperator in operators)
        {
            if (index.ContainsKey(compilerOperator.NodeType))
            {
                throw new InvalidOperationException($"Duplicate compiler node type '{compilerOperator.NodeType.FullName}'.");
            }

            index.Add(compilerOperator.NodeType, compilerOperator);
        }

        return index;
    }

    private static IReadOnlyDictionary<string, CompilerOperator> BuildCallIndex()
    {
        return BuildFormIndex(CompilerTorchSharpFormKind.Call);
    }

    internal static IReadOnlyDictionary<string, CompilerOperator> BuildFormIndex(CompilerTorchSharpFormKind kind)
    {
        return BuildFormIndex(_operators, kind);
    }

    internal static IReadOnlyDictionary<string, CompilerOperator> BuildFormIndex(IEnumerable<CompilerOperator> operators, CompilerTorchSharpFormKind kind)
    {
        var index = new Dictionary<string, CompilerOperator>(StringComparer.Ordinal);
        foreach (var compilerOperator in operators)
        {
            foreach (var form in compilerOperator.TorchSharpForms.Where(form => form.Kind == kind))
            {
                if (index.ContainsKey(form.Name))
                {
                    throw new InvalidOperationException($"Duplicate compiler operator {kind} '{form.Name}'.");
                }

                index.Add(form.Name, compilerOperator);
            }
        }

        return index;
    }

    private static IReadOnlyDictionary<CompilerOperatorIdentity, CompilerOperator> BuildIndex(
        IEnumerable<CompilerOperator> items,
        Func<CompilerOperator, CompilerOperatorIdentity> keySelector,
        string keyDescription
    )
    {
        var index = new Dictionary<CompilerOperatorIdentity, CompilerOperator>();
        foreach (var compilerOperator in items)
        {
            var key = keySelector(compilerOperator);
            if (index.ContainsKey(key))
            {
                throw new InvalidOperationException($"Duplicate compiler operator {keyDescription} '{key}'.");
            }

            index.Add(key, compilerOperator);
        }

        return index;
    }

    private static string NormalizeDomain(string domain)
    {
        return string.Equals(domain, "ai.onnx", StringComparison.Ordinal) ? string.Empty : domain ?? string.Empty;
    }

    internal static bool TryGetMemberPath(CompilerExpression expression, out string name)
    {
        if (expression is CompilerReferenceExpression reference)
        {
            name = reference.Name;
            return true;
        }

        if (expression is CompilerMemberAccessExpression member && TryGetMemberPath(member.Target, out var prefix))
        {
            name = $"{prefix}.{member.MemberName}";
            return true;
        }

        name = string.Empty;
        return false;
    }
}
