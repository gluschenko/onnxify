using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::heaviside")]
internal sealed class HeavisideOperator() : TorchSharpCompositeOperator("Heaviside", "com.onnxify.compiler", "torch.heaviside")
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

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (TryScanCastLikePattern(context, out operation))
        {
            consumedNodeCount = 6;
            return true;
        }

        if (!context.TryGetNode<Onnxify.Less>(0, out var less)
            || !context.TryGetNode<Onnxify.Where>(1, out var negativeCase)
            || !context.TryGetNode<Onnxify.Equal>(2, out var equal)
            || !context.TryGetNode<Onnxify.Where>(3, out var zeroCase)
            || less is null || negativeCase is null || equal is null || zeroCase is null
            || !context.IsScalarConstant(less.B.Name, 0d)
            || !context.IsScalarConstant(negativeCase.Y.Name, 1d)
            || !string.Equals(less.B.Name, negativeCase.X.Name, StringComparison.Ordinal)
            || !string.Equals(less.C.Name, negativeCase.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(negativeCase.Output.Name, zeroCase.Y.Name, StringComparison.Ordinal)
            || !context.IsScalarConstant(equal.B.Name, 0d)
            || !string.Equals(less.A.Name, equal.A.Name, StringComparison.Ordinal)
            || !string.Equals(equal.C.Name, zeroCase.Condition.Name, StringComparison.Ordinal)
            || !context.CanCollapse(4))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [less.A.Name, zeroCase.X.Name], zeroCase.Output.Name, 4);
        consumedNodeCount = 4;
        return true;
    }

    private bool TryScanCastLikePattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation)
    {
        operation = null;
        if (!context.TryGetNode<Onnxify.Less>(2, out var less)
            || !context.TryGetNode<Onnxify.Where>(3, out var negativeCase)
            || !context.TryGetNode<Onnxify.Equal>(4, out var equal)
            || !context.TryGetNode<Onnxify.Where>(5, out var zeroCase)
            || less is null || negativeCase is null || equal is null || zeroCase is null)
        {
            return false;
        }

        var nodes = context.Graph.Nodes;
        var zeroCast = nodes[context.StartIndex];
        var oneCast = nodes[context.StartIndex + 1];
        if (!IsDefaultCastLike(zeroCast)
            || !IsDefaultCastLike(oneCast)
            || !context.IsScalarConstant(zeroCast.Inputs[0].Name, 0d)
            || !context.IsScalarConstant(oneCast.Inputs[0].Name, 1d)
            || !string.Equals(zeroCast.Inputs[1].Name, oneCast.Inputs[1].Name, StringComparison.Ordinal)
            || !string.Equals(less.A.Name, zeroCast.Inputs[1].Name, StringComparison.Ordinal)
            || !string.Equals(less.B.Name, zeroCast.Outputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(negativeCase.Condition.Name, less.C.Name, StringComparison.Ordinal)
            || !string.Equals(negativeCase.X.Name, zeroCast.Outputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(negativeCase.Y.Name, oneCast.Outputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(equal.A.Name, less.A.Name, StringComparison.Ordinal)
            || !string.Equals(equal.B.Name, zeroCast.Outputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(zeroCase.Condition.Name, equal.C.Name, StringComparison.Ordinal)
            || !string.Equals(zeroCase.Y.Name, negativeCase.Output.Name, StringComparison.Ordinal)
            || !context.CanCollapse(6))
        {
            return false;
        }

        operation = context.CreateCompositeStep(
            this,
            [less.A.Name, zeroCase.X.Name],
            zeroCase.Output.Name,
            6);
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
        string Name(string suffix) => GetUniqueValueName(graph, $"{operation.Name}__{suffix}");

        var zeroSource = graph.AddTensor(Name("zero_source"), [], [0d]);
        var oneSource = graph.AddTensor(Name("one_source"), [], [1d]);
        var zero = new OnnxEdge(Name("zero"));
        var one = new OnnxEdge(Name("one"));
        graph.AddNode(new OnnxNode(
            $"{operation.Name}__cast_zero",
            "CastLike",
            string.Empty,
            string.Empty,
            [zeroSource, Input(operation)],
            [zero],
            []));
        graph.AddNode(new OnnxNode(
            $"{operation.Name}__cast_one",
            "CastLike",
            string.Empty,
            string.Empty,
            [oneSource, Input(operation)],
            [one],
            []));
        var lessResult = new OnnxEdge(Name("less_result"));
        var negativeOrPositive = new OnnxEdge(Name("negative_or_positive"));
        var equalResult = new OnnxEdge(Name("equal_result"));
        graph.AddNode(new Onnxify.Less($"{operation.Name}__less", new Onnxify.LessInputOutputOptions
        {
            A = Input(operation), B = zero, C = lessResult,
        }));
        graph.AddNode(new Onnxify.Where($"{operation.Name}__negative_case", new WhereInputOutputOptions
        {
            Condition = lessResult, X = zero, Y = one, Output = negativeOrPositive,
        }));
        graph.AddNode(new Onnxify.Equal($"{operation.Name}__equal", new Onnxify.EqualInputOutputOptions
        {
            A = Input(operation), B = zero, C = equalResult,
        }));
        graph.AddNode(new Onnxify.Where($"{operation.Name}__zero_case", new WhereInputOutputOptions
        {
            Condition = equalResult, X = Input(operation, 1), Y = negativeOrPositive, Output = Output(operation),
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
