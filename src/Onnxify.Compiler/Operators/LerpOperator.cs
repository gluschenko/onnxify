using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::lerp.Tensor")]
[CompilerTorchOperator("aten::lerp.Scalar")]
internal sealed class LerpOperator() : TorchSharpCompositeOperator("Lerp", "com.onnxify.compiler", "torch.lerp")
{
    public override int MinimumInputCount => 3;
    public override int MaximumInputCount => 3;

    protected override bool AllowsScalarOperand(int index)
    {
        return index == 2;
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
