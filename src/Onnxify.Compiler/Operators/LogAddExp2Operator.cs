using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::logaddexp2")]
internal sealed class LogAddExp2Operator() : TorchSharpCompositeOperator("LogAddExp2", "com.onnxify.compiler", "torch.logaddexp2")
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
            || !context.TryGetNode<Onnxify.Pow>(3, out var power)
            || !context.TryGetNode<Onnxify.Add>(4, out var addOne)
            || !context.TryGetNode<Onnxify.Log>(5, out var logSum)
            || !context.TryGetNode<Onnxify.Log>(6, out var logTwo)
            || !context.TryGetNode<Onnxify.Div>(7, out var divide)
            || !context.TryGetNode<Onnxify.Max>(8, out var maximum)
            || !context.TryGetNode<Onnxify.Add>(9, out var addMaximum)
            || !context.TryGetNode<Onnxify.IsInf>(10, out var leftIsInfinite)
            || !context.TryGetNode<Onnxify.IsInf>(11, out var rightIsInfinite)
            || !context.TryGetNode<Onnxify.Equal>(12, out var equal)
            || !context.TryGetNode<Onnxify.And>(13, out var bothInfinite)
            || !context.TryGetNode<Onnxify.And>(14, out var sameInfinite)
            || !context.TryGetNode<Onnxify.Where>(15, out var where)
            || subtract is null || absolute is null || negate is null || power is null
            || addOne is null || logSum is null || logTwo is null || divide is null
            || maximum is null || addMaximum is null || leftIsInfinite is null
            || rightIsInfinite is null || equal is null || bothInfinite is null
            || sameInfinite is null || where is null
            || !context.IsScalarConstant(power.X.Name, 2d)
            || !context.IsScalarConstant(addOne.A.Name, 1d)
            || leftIsInfinite.DetectNegative is not (null or 1)
            || leftIsInfinite.DetectPositive is not (null or 1)
            || rightIsInfinite.DetectNegative is not (null or 1)
            || rightIsInfinite.DetectPositive is not (null or 1)
            || !string.Equals(subtract.C.Name, absolute.X.Name, StringComparison.Ordinal)
            || !string.Equals(absolute.Y.Name, negate.X.Name, StringComparison.Ordinal)
            || !string.Equals(negate.Y.Name, power.Y.Name, StringComparison.Ordinal)
            || !string.Equals(power.Z.Name, addOne.B.Name, StringComparison.Ordinal)
            || !string.Equals(addOne.C.Name, logSum.Input.Name, StringComparison.Ordinal)
            || !string.Equals(power.X.Name, logTwo.Input.Name, StringComparison.Ordinal)
            || !string.Equals(logSum.Output.Name, divide.A.Name, StringComparison.Ordinal)
            || !string.Equals(logTwo.Output.Name, divide.B.Name, StringComparison.Ordinal)
            || !string.Equals(maximum.OutputMax.Name, addMaximum.A.Name, StringComparison.Ordinal)
            || !string.Equals(divide.C.Name, addMaximum.B.Name, StringComparison.Ordinal)
            || !string.Equals(leftIsInfinite.Y.Name, bothInfinite.A.Name, StringComparison.Ordinal)
            || !string.Equals(rightIsInfinite.Y.Name, bothInfinite.B.Name, StringComparison.Ordinal)
            || !string.Equals(bothInfinite.C.Name, sameInfinite.A.Name, StringComparison.Ordinal)
            || !string.Equals(equal.C.Name, sameInfinite.B.Name, StringComparison.Ordinal)
            || !string.Equals(sameInfinite.C.Name, where.Condition.Name, StringComparison.Ordinal)
            || !context.CanCollapse(16))
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

        operation = context.CreateCompositeStep(this, [leftName, rightName], where.Output.Name, 16);
        consumedNodeCount = 16;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var two = AddFloatInitializer(graph, operation, "two", 2f);
        var one = AddFloatInitializer(graph, operation, "one", 1f);
        var difference = new OnnxEdge($"{operation.Name}__difference");
        var absoluteDifference = new OnnxEdge($"{operation.Name}__absolute_difference");
        var negativeDifference = new OnnxEdge($"{operation.Name}__negative_difference");
        var relativePower = new OnnxEdge($"{operation.Name}__relative_power");
        var sum = new OnnxEdge($"{operation.Name}__sum");
        var logSum = new OnnxEdge($"{operation.Name}__log_sum");
        var logTwo = new OnnxEdge($"{operation.Name}__log_two");
        var maximum = new OnnxEdge($"{operation.Name}__maximum");
        var logarithm = new OnnxEdge($"{operation.Name}__logarithm");
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
        graph.AddNode(new Onnxify.Pow(
            $"{operation.Name}__pow",
            new Onnxify.PowInputOutputOptions
            {
                X = two,
                Y = negativeDifference,
                Z = relativePower,
            }));
        graph.AddNode(new Onnxify.Add(
            $"{operation.Name}__add_one",
            new Onnxify.AddInputOutputOptions
            {
                A = one,
                B = relativePower,
                C = sum,
            }));
        graph.AddNode(new Onnxify.Log(
            $"{operation.Name}__log_sum",
            new Onnxify.LogInputOutputOptions
            {
                Input = sum,
                Output = logSum,
            }));
        graph.AddNode(new Onnxify.Log(
            $"{operation.Name}__log_two_node",
            new Onnxify.LogInputOutputOptions
            {
                Input = two,
                Output = logTwo,
            }));
        graph.AddNode(new Onnxify.Div(
            $"{operation.Name}__log2",
            new Onnxify.DivInputOutputOptions
            {
                A = logSum,
                B = logTwo,
                C = logarithm,
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
