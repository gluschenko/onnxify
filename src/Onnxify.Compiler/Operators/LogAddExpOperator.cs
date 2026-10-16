using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::logaddexp")]
internal sealed class LogAddExpOperator() : TorchSharpCompositeOperator("LogAddExp", "com.onnxify.compiler", "torch.logaddexp")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Sub>(0, out var subtract)
            || !context.TryGetNode<Onnxify.Abs>(1, out var absolute)
            || !context.TryGetNode<Onnxify.Neg>(2, out var negate)
            || !context.TryGetNode<Onnxify.Exp>(3, out var exponential)
            || !context.TryGetNode<Onnxify.Add>(4, out var addOne)
            || !context.TryGetNode<Onnxify.Log>(5, out var logarithm)
            || !context.TryGetNode<Onnxify.Max>(6, out var maximum)
            || !context.TryGetNode<Onnxify.Add>(7, out var addMaximum)
            || !context.TryGetNode<Onnxify.IsInf>(8, out var leftIsInfinite)
            || !context.TryGetNode<Onnxify.IsInf>(9, out var rightIsInfinite)
            || !context.TryGetNode<Onnxify.Equal>(10, out var equal)
            || !context.TryGetNode<Onnxify.And>(11, out var bothInfinite)
            || !context.TryGetNode<Onnxify.And>(12, out var sameInfinite)
            || !context.TryGetNode<Onnxify.Where>(13, out var where)
            || subtract is null || absolute is null || negate is null || exponential is null
            || addOne is null || logarithm is null || maximum is null || addMaximum is null
            || leftIsInfinite is null || rightIsInfinite is null || equal is null
            || bothInfinite is null || sameInfinite is null || where is null
            || !context.IsScalarConstant(addOne.A.Name, 1d)
            || leftIsInfinite.DetectNegative is not (null or 1)
            || leftIsInfinite.DetectPositive is not (null or 1)
            || rightIsInfinite.DetectNegative is not (null or 1)
            || rightIsInfinite.DetectPositive is not (null or 1)
            || !string.Equals(subtract.C.Name, absolute.X.Name, StringComparison.Ordinal)
            || !string.Equals(absolute.Y.Name, negate.X.Name, StringComparison.Ordinal)
            || !string.Equals(negate.Y.Name, exponential.Input.Name, StringComparison.Ordinal)
            || !string.Equals(exponential.Output.Name, addOne.B.Name, StringComparison.Ordinal)
            || !string.Equals(addOne.C.Name, logarithm.Input.Name, StringComparison.Ordinal)
            || !string.Equals(maximum.OutputMax.Name, addMaximum.A.Name, StringComparison.Ordinal)
            || !string.Equals(logarithm.Output.Name, addMaximum.B.Name, StringComparison.Ordinal)
            || !string.Equals(leftIsInfinite.Y.Name, bothInfinite.A.Name, StringComparison.Ordinal)
            || !string.Equals(rightIsInfinite.Y.Name, bothInfinite.B.Name, StringComparison.Ordinal)
            || !string.Equals(bothInfinite.C.Name, sameInfinite.A.Name, StringComparison.Ordinal)
            || !string.Equals(equal.C.Name, sameInfinite.B.Name, StringComparison.Ordinal)
            || !string.Equals(sameInfinite.C.Name, where.Condition.Name, StringComparison.Ordinal)
            || !context.CanCollapse(14))
        {
            return false;
        }

        var leftName = subtract.A.Name;
        var rightName = subtract.B.Name;
        if (!string.Equals(maximum.Data0.ElementAtOrDefault(0)?.Name, leftName, StringComparison.Ordinal)
            || !string.Equals(maximum.Data0.ElementAtOrDefault(1)?.Name, rightName, StringComparison.Ordinal)
            || !string.Equals(leftIsInfinite.X.Name, leftName, StringComparison.Ordinal)
            || !string.Equals(rightIsInfinite.X.Name, rightName, StringComparison.Ordinal)
            || !string.Equals(equal.A.Name, leftName, StringComparison.Ordinal)
            || !string.Equals(equal.B.Name, rightName, StringComparison.Ordinal)
            || !string.Equals(where.X.Name, leftName, StringComparison.Ordinal)
            || !string.Equals(where.Y.Name, addMaximum.C.Name, StringComparison.Ordinal))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [leftName, rightName], where.Output.Name, 14);
        consumedNodeCount = 14;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var one = AddFloatInitializer(graph, operation, "one", 1f);
        var difference = new OnnxEdge($"{operation.Name}__difference");
        var absoluteDifference = new OnnxEdge($"{operation.Name}__absolute_difference");
        var negativeDifference = new OnnxEdge($"{operation.Name}__negative_difference");
        var exponential = new OnnxEdge($"{operation.Name}__exponential");
        var sum = new OnnxEdge($"{operation.Name}__sum");
        var logarithm = new OnnxEdge($"{operation.Name}__logarithm");
        var maximum = new OnnxEdge($"{operation.Name}__maximum");
        var regularResult = new OnnxEdge($"{operation.Name}__regular_result");
        var leftIsInfinite = new OnnxEdge($"{operation.Name}__left_is_infinite");
        var rightIsInfinite = new OnnxEdge($"{operation.Name}__right_is_infinite");
        var valuesAreEqual = new OnnxEdge($"{operation.Name}__values_are_equal");
        var bothInfinite = new OnnxEdge($"{operation.Name}__both_infinite");
        var sameInfiniteValue = new OnnxEdge($"{operation.Name}__same_infinite_value");

        graph.AddNode(new Onnxify.Sub(
            $"{operation.Name}__sub",
            new Onnxify.SubInputOutputOptions
            {
                A = Input(operation),
                B = Input(operation, 1),
                C = difference,
            }));
        graph.AddNode(new Onnxify.Abs(
            $"{operation.Name}__abs",
            new Onnxify.AbsInputOutputOptions
            {
                X = difference,
                Y = absoluteDifference,
            }));
        graph.AddNode(new Onnxify.Neg(
            $"{operation.Name}__neg",
            new Onnxify.NegInputOutputOptions
            {
                X = absoluteDifference,
                Y = negativeDifference,
            }));
        graph.AddNode(new Onnxify.Exp(
            $"{operation.Name}__exp",
            new Onnxify.ExpInputOutputOptions
            {
                Input = negativeDifference,
                Output = exponential,
            }));
        graph.AddNode(new Onnxify.Add(
            $"{operation.Name}__add_one",
            new Onnxify.AddInputOutputOptions
            {
                A = one,
                B = exponential,
                C = sum,
            }));
        graph.AddNode(new Onnxify.Log(
            $"{operation.Name}__log",
            new Onnxify.LogInputOutputOptions
            {
                Input = sum,
                Output = logarithm,
            }));
        graph.AddNode(new Onnxify.Max(
            $"{operation.Name}__max",
            new Onnxify.MaxInputOutputOptions
            {
                Data0 = [Input(operation), Input(operation, 1)],
                OutputMax = maximum,
            }));
        graph.AddNode(new Onnxify.Add(
            $"{operation.Name}__add_maximum",
            new Onnxify.AddInputOutputOptions
            {
                A = maximum,
                B = logarithm,
                C = regularResult,
            }));
        graph.AddNode(new Onnxify.IsInf(
            $"{operation.Name}__isinf_left",
            new Onnxify.IsInfInputOutputOptions
            {
                X = Input(operation),
                Y = leftIsInfinite,
            }));
        graph.AddNode(new Onnxify.IsInf(
            $"{operation.Name}__isinf_right",
            new Onnxify.IsInfInputOutputOptions
            {
                X = Input(operation, 1),
                Y = rightIsInfinite,
            }));
        graph.AddNode(new Onnxify.Equal(
            $"{operation.Name}__equal",
            new Onnxify.EqualInputOutputOptions
            {
                A = Input(operation),
                B = Input(operation, 1),
                C = valuesAreEqual,
            }));
        graph.AddNode(new Onnxify.And(
            $"{operation.Name}__and",
            new Onnxify.AndInputOutputOptions
            {
                A = leftIsInfinite,
                B = rightIsInfinite,
                C = bothInfinite,
            }));
        graph.AddNode(new Onnxify.And(
            $"{operation.Name}__and_equal",
            new Onnxify.AndInputOutputOptions
            {
                A = bothInfinite,
                B = valuesAreEqual,
                C = sameInfiniteValue,
            }));
        graph.AddNode(new Onnxify.Where(
            operation.Name,
            new Onnxify.WhereInputOutputOptions
            {
                Condition = sameInfiniteValue,
                X = Input(operation),
                Y = regularResult,
                Output = Output(operation),
            }));
    }
}
