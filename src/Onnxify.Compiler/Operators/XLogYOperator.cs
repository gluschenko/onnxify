using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::xlogy.Tensor")]
[CompilerTorchOperator("aten::xlogy.Scalar_Self")]
[CompilerTorchOperator("aten::xlogy.Scalar_Other")]
internal sealed class XLogYOperator() : TorchSharpCompositeOperator("XLogY", "com.onnxify.compiler", "torch.xlogy")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;
    public override bool SupportsMultidirectionalBroadcast => true;

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return ValidateFloatingPointTensorInputs(node, knownTypes,
            CompilerElementType.Float16, CompilerElementType.Float32, CompilerElementType.Float64);
    }
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.xlogy"),
        CompilerTorchSharpForm.Call("Tensor.xlogy"),
    ];

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        return $"torch.xlogy({CompilerCSharpNaming.Identifier(operation.Node.Inputs[0].Name)}, {CompilerCSharpNaming.Identifier(operation.Node.Inputs[1].Name)})";
    }

    protected override bool AllowsScalarOperand(int index) => index is 0 or 1;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (TryScanCastLikePattern(context, out operation))
        {
            consumedNodeCount = 7;
            return true;
        }

        if (!context.TryGetNode<Onnxify.IsNaN>(0, out var isNaN)
            || !context.TryGetNode<Onnxify.Equal>(1, out var isZero)
            || !context.TryGetNode<Onnxify.Log>(2, out var logarithm)
            || !context.TryGetNode<Onnxify.Mul>(3, out var multiply)
            || !context.TryGetNode<Onnxify.Where>(4, out var zeroCase)
            || !context.TryGetNode<Onnxify.Where>(5, out var nanCase)
            || isNaN is null || isZero is null || logarithm is null || multiply is null || nanCase is null || zeroCase is null
            || !context.IsScalarConstant(isZero.B.Name, 0d)
            || !string.Equals(isNaN.X.Name, logarithm.Input.Name, StringComparison.Ordinal)
            || !string.Equals(isZero.C.Name, zeroCase.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(isZero.A.Name, multiply.A.Name, StringComparison.Ordinal)
            || !string.Equals(isNaN.Y.Name, nanCase.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(logarithm.Output.Name, multiply.B.Name, StringComparison.Ordinal)
            || !string.Equals(multiply.C.Name, zeroCase.Y.Name, StringComparison.Ordinal)
            || !string.Equals(zeroCase.Output.Name, nanCase.Y.Name, StringComparison.Ordinal)
            || !string.Equals(isZero.A.Name, zeroCase.X.Name, StringComparison.Ordinal)
            || !string.Equals(isNaN.X.Name, nanCase.X.Name, StringComparison.Ordinal)
            || !context.CanCollapse(6))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [isZero.A.Name, isNaN.X.Name], nanCase.Output.Name, 6);
        consumedNodeCount = 6;
        return true;
    }

    private bool TryScanCastLikePattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation)
    {
        operation = null;
        if (!context.TryGetNode<Onnxify.IsNaN>(1, out var isNaN)
            || !context.TryGetNode<Onnxify.Equal>(2, out var isZero)
            || !context.TryGetNode<Onnxify.Log>(3, out var logarithm)
            || !context.TryGetNode<Onnxify.Mul>(4, out var multiply)
            || !context.TryGetNode<Onnxify.Where>(5, out var zeroCase)
            || !context.TryGetNode<Onnxify.Where>(6, out var nanCase)
            || isNaN is null || isZero is null || logarithm is null || multiply is null || zeroCase is null || nanCase is null)
        {
            return false;
        }

        var castLike = context.Graph.Nodes[context.StartIndex];
        if (!IsDefaultCastLike(castLike)
            || !context.IsScalarConstant(castLike.Inputs[0].Name, 0d)
            || !string.Equals(castLike.Inputs[1].Name, isZero.A.Name, StringComparison.Ordinal)
            || !string.Equals(isZero.B.Name, castLike.Outputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(isNaN.X.Name, logarithm.Input.Name, StringComparison.Ordinal)
            || !string.Equals(isZero.C.Name, zeroCase.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(isZero.A.Name, multiply.A.Name, StringComparison.Ordinal)
            || !string.Equals(isNaN.Y.Name, nanCase.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(logarithm.Output.Name, multiply.B.Name, StringComparison.Ordinal)
            || !string.Equals(multiply.C.Name, zeroCase.Y.Name, StringComparison.Ordinal)
            || !string.Equals(zeroCase.Output.Name, nanCase.Y.Name, StringComparison.Ordinal)
            || !string.Equals(isZero.A.Name, zeroCase.X.Name, StringComparison.Ordinal)
            || !string.Equals(isNaN.X.Name, nanCase.X.Name, StringComparison.Ordinal)
            || !context.CanCollapse(7))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [isZero.A.Name, isNaN.X.Name], nanCase.Output.Name, 7);
        return true;
    }

    private static bool IsDefaultCastLike(OnnxNode node) =>
        string.Equals(node.Domain, string.Empty, StringComparison.Ordinal)
        && string.Equals(node.OpType, "CastLike", StringComparison.Ordinal)
        && node.Inputs.Count == 2
        && node.Outputs.Count == 1
        && node.Attributes.All(static attribute =>
            string.Equals(attribute.Name, "saturate", StringComparison.Ordinal)
            && attribute.GetValue() is long value
            && value == 1);

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var zeroSource = graph.AddTensor(GetUniqueValueName(graph, $"{operation.Name}__zero_source"), [], [0d]);
        var zero = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__zero"));
        graph.AddNode(new OnnxNode(
            GetUniqueValueName(graph, $"{operation.Name}__cast_zero"),
            "CastLike",
            string.Empty,
            string.Empty,
            [zeroSource, Input(operation)],
            [zero],
            []));
        var isNaN = new OnnxEdge($"{operation.Name}__isnan_result");
        var isZero = new OnnxEdge($"{operation.Name}__iszero_result");
        var logarithm = new OnnxEdge($"{operation.Name}__log_result");
        var product = new OnnxEdge($"{operation.Name}__product");
        var zeroCase = new OnnxEdge($"{operation.Name}__zero_case");
        graph.AddNode(new Onnxify.IsNaN($"{operation.Name}__isnan", new Onnxify.IsNaNInputOutputOptions
        {
            X = Input(operation, 1), Y = isNaN,
        }));
        graph.AddNode(new Onnxify.Equal($"{operation.Name}__equal_zero", new Onnxify.EqualInputOutputOptions
        {
            A = Input(operation), B = zero, C = isZero,
        }));
        graph.AddNode(new Onnxify.Log($"{operation.Name}__log", new Onnxify.LogInputOutputOptions
        {
            Input = Input(operation, 1), Output = logarithm,
        }));
        graph.AddNode(new Onnxify.Mul($"{operation.Name}__mul", new Onnxify.MulInputOutputOptions
        {
            A = Input(operation), B = logarithm, C = product,
        }));
        graph.AddNode(new Onnxify.Where($"{operation.Name}__zero_where", new WhereInputOutputOptions
        {
            Condition = isZero, X = Input(operation), Y = product, Output = zeroCase,
        }));
        graph.AddNode(new Onnxify.Where(operation.Name, new WhereInputOutputOptions
        {
            Condition = isNaN, X = Input(operation, 1), Y = zeroCase, Output = Output(operation),
        }));
    }

    private static string GetUniqueValueName(OnnxGraph graph, string baseName)
    {
        var candidate = baseName;
        for (var suffix = 1; graph.GetValue(candidate) is not null; suffix++)
        {
            candidate = $"{baseName}__{suffix}";
        }

        return candidate;
    }
}
