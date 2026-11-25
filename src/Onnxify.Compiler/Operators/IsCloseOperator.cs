using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::isclose")]
internal sealed class IsCloseOperator() : TorchSharpCompositeOperator("IsClose", "com.onnxify.compiler", "Tensor.isclose")
{
    private const double RelativeTolerance = 1e-5d;
    private const double AbsoluteTolerance = 1e-8d;

    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;
    public override bool SupportsMultidirectionalBroadcast => true;
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return ValidateFloatingPointTensorInputs(node, knownTypes,
            CompilerElementType.Float16, CompilerElementType.Float32, CompilerElementType.Float64);
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        return $"{CompilerCSharpNaming.Identifier(operation.Node.Inputs[0].Name)}.isclose({CompilerCSharpNaming.Identifier(operation.Node.Inputs[1].Name)}, {RelativeTolerance:R}, {AbsoluteTolerance:R}, false)";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        var invocation = context.Invocation;
        if (invocation is null
            || context.Receiver is null
            || invocation.Arguments.Count != 4
            || invocation.Arguments[0] is not CompilerReferenceExpression
            || invocation.Arguments[1] is not CompilerLiteralExpression { Literal: CompilerFloatingPointLiteral rtol } || rtol.Value != 1e-5d
            || invocation.Arguments[2] is not CompilerLiteralExpression { Literal: CompilerFloatingPointLiteral atol } || atol.Value != 1e-8d
            || invocation.Arguments[3] is not CompilerLiteralExpression { Literal: CompilerBooleanLiteral { Value: false } })
        {
            throw context.Unsupported(expression, "Pointwise isclose supports tensor operands with rtol=1e-5, atol=1e-8, and equal_nan=false.");
        }

        var left = context.RequireTensorReference(context.Receiver, OnnxName);
        var right = context.RequireTensorReference(invocation.Arguments[0], OnnxName);
        context.AddOperation(this, [left, right], span: expression.Span);
        return true;
    }

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (TryScanCastLikePattern(context, out operation))
        {
            consumedNodeCount = 10;
            return true;
        }

        if (!context.TryGetNode<Onnxify.Sub>(0, out var subtract)
            || !context.TryGetNode<Onnxify.Abs>(1, out var absDifference)
            || !context.TryGetNode<Onnxify.Abs>(2, out var absRight)
            || !context.TryGetNode<Onnxify.Mul>(3, out var scaledRight)
            || !context.TryGetNode<Onnxify.Add>(4, out var tolerance)
            || !context.TryGetNode<Onnxify.LessOrEqual>(5, out var withinTolerance)
            || !context.TryGetNode<Onnxify.Equal>(6, out var equal)
            || !context.TryGetNode<Onnxify.Or>(7, out var result)
            || subtract is null || absDifference is null || absRight is null || scaledRight is null
            || tolerance is null || withinTolerance is null || equal is null || result is null
            || !(context.IsScalarConstant(scaledRight.B.Name, RelativeTolerance)
                || context.IsScalarConstant(scaledRight.B.Name, (double)(float)RelativeTolerance))
            || !(context.IsScalarConstant(tolerance.A.Name, AbsoluteTolerance)
                || context.IsScalarConstant(tolerance.A.Name, (double)(float)AbsoluteTolerance))
            || !string.Equals(subtract.C.Name, absDifference.X.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.B.Name, absRight.X.Name, StringComparison.Ordinal)
            || !string.Equals(absRight.Y.Name, scaledRight.A.Name, StringComparison.Ordinal)
            || !string.Equals(scaledRight.C.Name, tolerance.B.Name, StringComparison.Ordinal)
            || !string.Equals(absDifference.Y.Name, withinTolerance.A.Name, StringComparison.Ordinal)
            || !string.Equals(tolerance.C.Name, withinTolerance.B.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.A.Name, equal.A.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.B.Name, equal.B.Name, StringComparison.Ordinal)
            || !string.Equals(equal.C.Name, result.A.Name, StringComparison.Ordinal)
            || !string.Equals(withinTolerance.C.Name, result.B.Name, StringComparison.Ordinal)
            || !context.CanCollapse(8))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [subtract.A.Name, subtract.B.Name], result.C.Name, 8);
        consumedNodeCount = 8;
        return true;
    }

    private bool TryScanCastLikePattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation)
    {
        operation = null;
        if (!context.TryGetNode<Onnxify.Sub>(2, out var subtract)
            || !context.TryGetNode<Onnxify.Abs>(3, out var absDifference)
            || !context.TryGetNode<Onnxify.Abs>(4, out var absRight)
            || !context.TryGetNode<Onnxify.Mul>(5, out var scaledRight)
            || !context.TryGetNode<Onnxify.Add>(6, out var tolerance)
            || !context.TryGetNode<Onnxify.LessOrEqual>(7, out var withinTolerance)
            || !context.TryGetNode<Onnxify.Equal>(8, out var equal)
            || !context.TryGetNode<Onnxify.Or>(9, out var result)
            || subtract is null || absDifference is null || absRight is null || scaledRight is null
            || tolerance is null || withinTolerance is null || equal is null || result is null)
        {
            return false;
        }

        var relativeCast = context.Graph.Nodes[context.StartIndex];
        var absoluteCast = context.Graph.Nodes[context.StartIndex + 1];
        if (!IsDefaultCastLike(relativeCast)
            || !IsDefaultCastLike(absoluteCast)
            || !(context.IsScalarConstant(relativeCast.Inputs[0].Name, RelativeTolerance)
                || context.IsScalarConstant(relativeCast.Inputs[0].Name, (double)(float)RelativeTolerance))
            || !(context.IsScalarConstant(absoluteCast.Inputs[0].Name, AbsoluteTolerance)
                || context.IsScalarConstant(absoluteCast.Inputs[0].Name, (double)(float)AbsoluteTolerance))
            || !string.Equals(relativeCast.Inputs[1].Name, subtract.A.Name, StringComparison.Ordinal)
            || !string.Equals(absoluteCast.Inputs[1].Name, subtract.A.Name, StringComparison.Ordinal)
            || !string.Equals(relativeCast.Outputs[0].Name, scaledRight.B.Name, StringComparison.Ordinal)
            || !string.Equals(absoluteCast.Outputs[0].Name, tolerance.A.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.C.Name, absDifference.X.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.B.Name, absRight.X.Name, StringComparison.Ordinal)
            || !string.Equals(absRight.Y.Name, scaledRight.A.Name, StringComparison.Ordinal)
            || !string.Equals(scaledRight.C.Name, tolerance.B.Name, StringComparison.Ordinal)
            || !string.Equals(absDifference.Y.Name, withinTolerance.A.Name, StringComparison.Ordinal)
            || !string.Equals(tolerance.C.Name, withinTolerance.B.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.A.Name, equal.A.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.B.Name, equal.B.Name, StringComparison.Ordinal)
            || !string.Equals(equal.C.Name, result.A.Name, StringComparison.Ordinal)
            || !string.Equals(withinTolerance.C.Name, result.B.Name, StringComparison.Ordinal)
            || !context.CanCollapse(10))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [subtract.A.Name, subtract.B.Name], result.C.Name, 10);
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
        var relativeSource = graph.AddTensor(GetUniqueValueName(graph, $"{operation.Name}__rtol_source"), [], [RelativeTolerance]);
        var absoluteSource = graph.AddTensor(GetUniqueValueName(graph, $"{operation.Name}__atol_source"), [], [AbsoluteTolerance]);
        var relativeTolerance = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__rtol"));
        var absoluteTolerance = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__atol"));
        graph.AddNode(new OnnxNode(GetUniqueValueName(graph, $"{operation.Name}__cast_rtol"),
            "CastLike", string.Empty, string.Empty, [relativeSource, Input(operation)], [relativeTolerance], []));
        graph.AddNode(new OnnxNode(GetUniqueValueName(graph, $"{operation.Name}__cast_atol"),
            "CastLike", string.Empty, string.Empty, [absoluteSource, Input(operation)], [absoluteTolerance], []));
        var difference = new OnnxEdge($"{operation.Name}__difference");
        var absoluteDifference = new OnnxEdge($"{operation.Name}__absolute_difference");
        var absoluteRight = new OnnxEdge($"{operation.Name}__absolute_right");
        var scaledRight = new OnnxEdge($"{operation.Name}__scaled_right");
        var tolerance = new OnnxEdge($"{operation.Name}__tolerance");
        var withinTolerance = new OnnxEdge($"{operation.Name}__within_tolerance");
        var equal = new OnnxEdge($"{operation.Name}__equal");
        graph.AddNode(new Onnxify.Sub($"{operation.Name}__sub", new Onnxify.SubInputOutputOptions
        {
            A = Input(operation), B = Input(operation, 1), C = difference,
        }));
        graph.AddNode(new Onnxify.Abs($"{operation.Name}__abs_difference", new Onnxify.AbsInputOutputOptions
        {
            X = difference, Y = absoluteDifference,
        }));
        graph.AddNode(new Onnxify.Abs($"{operation.Name}__abs_right", new Onnxify.AbsInputOutputOptions
        {
            X = Input(operation, 1), Y = absoluteRight,
        }));
        graph.AddNode(new Onnxify.Mul($"{operation.Name}__mul", new Onnxify.MulInputOutputOptions
        {
            A = absoluteRight, B = relativeTolerance, C = scaledRight,
        }));
        graph.AddNode(new Onnxify.Add($"{operation.Name}__add", new Onnxify.AddInputOutputOptions
        {
            A = absoluteTolerance, B = scaledRight, C = tolerance,
        }));
        graph.AddNode(new Onnxify.LessOrEqual($"{operation.Name}__less_or_equal", new Onnxify.LessOrEqualInputOutputOptions
        {
            A = absoluteDifference, B = tolerance, C = withinTolerance,
        }));
        graph.AddNode(new Onnxify.Equal($"{operation.Name}__equal", new Onnxify.EqualInputOutputOptions
        {
            A = Input(operation), B = Input(operation, 1), C = equal,
        }));
        graph.AddNode(new Onnxify.Or(operation.Name, new Onnxify.OrInputOutputOptions
        {
            A = equal, B = withinTolerance, C = Output(operation),
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
