using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::logaddexp2")]
internal sealed class LogAddExp2Operator() : TorchSharpCompositeOperator("LogAddExp2", "com.onnxify.compiler", "torch.logaddexp2")
{
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;

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
