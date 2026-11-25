using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::lerp.Tensor")]
[CompilerTorchOperator("aten::lerp.Scalar")]
internal sealed class LerpOperator() : TorchSharpCompositeOperator("Lerp", "com.onnxify.compiler", "torch.lerp")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override int MinimumInputCount => 3;
    public override int MaximumInputCount => 3;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Sub>(0, out var difference)
            || !context.TryGetNode<Onnxify.Mul>(1, out var weightedDifference)
            || !context.TryGetNode<Onnxify.Add>(2, out var nearStart)
            || !context.TryGetNode<Onnxify.Sub>(3, out var oneMinusWeight)
            || !context.TryGetNode<Onnxify.Mul>(4, out var correction)
            || !context.TryGetNode<Onnxify.Sub>(5, out var nearEnd)
            || !context.TryGetNode<Onnxify.Less>(6, out var weightBelowHalf)
            || !context.TryGetNode<Onnxify.Where>(7, out var where)
            || difference is null || weightedDifference is null || nearStart is null
            || oneMinusWeight is null || correction is null || nearEnd is null
            || weightBelowHalf is null || where is null
            || !context.IsScalarConstant(oneMinusWeight.A.Name, 1d)
            || !context.IsScalarConstant(weightBelowHalf.B.Name, 0.5d)
            || !string.Equals(difference.C.Name, weightedDifference.B.Name, StringComparison.Ordinal)
            || !string.Equals(difference.C.Name, correction.A.Name, StringComparison.Ordinal)
            || !string.Equals(weightedDifference.C.Name, nearStart.B.Name, StringComparison.Ordinal)
            || !string.Equals(oneMinusWeight.C.Name, correction.B.Name, StringComparison.Ordinal)
            || !string.Equals(correction.C.Name, nearEnd.B.Name, StringComparison.Ordinal)
            || !string.Equals(weightBelowHalf.C.Name, where.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(nearStart.C.Name, where.X.Name, StringComparison.Ordinal)
            || !string.Equals(nearEnd.C.Name, where.Y.Name, StringComparison.Ordinal)
            || !context.CanCollapse(8))
        {
            return false;
        }

        var startName = difference.B.Name;
        var endName = difference.A.Name;
        var weightName = weightedDifference.A.Name;
        if (!string.Equals(nearStart.A.Name, startName, StringComparison.Ordinal)
            || !string.Equals(oneMinusWeight.B.Name, weightName, StringComparison.Ordinal)
            || !string.Equals(nearEnd.A.Name, endName, StringComparison.Ordinal)
            || !string.Equals(weightBelowHalf.A.Name, weightName, StringComparison.Ordinal))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [startName, endName, weightName], where.Output.Name, 8);
        consumedNodeCount = 8;
        return true;
    }

    protected override bool AllowsScalarOperand(int index)
    {
        return index == 2;
    }

    public override string PrintTorchSharp(
        CompilerOnnxStep operation,
        CompilerTorchSharpPrintContext context)
    {
        var start = TorchSharpInput(operation, 0, context);
        var end = TorchSharpInput(operation, 1, context);
        var weight = TorchSharpInput(operation, 2, context);
        return $"torch.lerp({start}, {end}, {weight})";
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var one = AddFloatInitializer(graph, operation, "one", 1f);
        var half = AddFloatInitializer(graph, operation, "half", 0.5f);
        var difference = new OnnxEdge($"{operation.Name}__difference");
        var weightedDifference = new OnnxEdge($"{operation.Name}__weighted_difference");
        var nearEnd = new OnnxEdge($"{operation.Name}__near_end");
        var oneMinusWeight = new OnnxEdge($"{operation.Name}__one_minus_weight");
        var correction = new OnnxEdge($"{operation.Name}__correction");
        var nearStart = new OnnxEdge($"{operation.Name}__near_start");
        var condition = new OnnxEdge($"{operation.Name}__condition");

        graph.AddNode(new Onnxify.Sub(
            $"{operation.Name}__difference_node",
            new Onnxify.SubInputOutputOptions
            {
                A = Input(operation, 1),
                B = Input(operation),
                C = difference,
            }));
        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__weighted_difference_node",
            new Onnxify.MulInputOutputOptions
            {
                A = Input(operation, 2),
                B = difference,
                C = weightedDifference,
            }));
        graph.AddNode(new Onnxify.Add(
            $"{operation.Name}__near_start_node",
            new Onnxify.AddInputOutputOptions
            {
                A = Input(operation),
                B = weightedDifference,
                C = nearStart,
            }));
        graph.AddNode(new Onnxify.Sub(
            $"{operation.Name}__one_minus_weight_node",
            new Onnxify.SubInputOutputOptions
            {
                A = one,
                B = Input(operation, 2),
                C = oneMinusWeight,
            }));
        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__correction_node",
            new Onnxify.MulInputOutputOptions
            {
                A = difference,
                B = oneMinusWeight,
                C = correction,
            }));
        graph.AddNode(new Onnxify.Sub(
            $"{operation.Name}__near_end_node",
            new Onnxify.SubInputOutputOptions
            {
                A = Input(operation, 1),
                B = correction,
                C = nearEnd,
            }));
        graph.AddNode(new Onnxify.Less(
            $"{operation.Name}__condition_node",
            new Onnxify.LessInputOutputOptions
            {
                A = Input(operation, 2),
                B = half,
                C = condition,
            }));
        graph.AddNode(new Onnxify.Where(
            operation.Name,
            new Onnxify.WhereInputOutputOptions
            {
                Condition = condition,
                X = nearStart,
                Y = nearEnd,
                Output = Output(operation),
            }));
    }
}
