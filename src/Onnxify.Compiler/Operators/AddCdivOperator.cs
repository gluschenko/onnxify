using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::addcdiv")]
internal sealed class AddCdivOperator() : TorchSharpCompositeOperator("AddCdiv", "com.onnxify.compiler", "torch.addcdiv")
{
    public override int MinimumInputCount => 4;
    public override int MaximumInputCount => 4;

    protected override bool AllowsScalarOperand(int index)
    {
        return index == 3;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var quotient = new OnnxEdge($"{operation.Name}__quotient");
        var scaled = new OnnxEdge($"{operation.Name}__scaled");
        var value = Input(operation, 3);

        graph.AddNode(new Onnxify.Div(
            $"{operation.Name}__div",
            new Onnxify.DivInputOutputOptions
            {
                A = Input(operation, 1),
                B = Input(operation, 2),
                C = quotient,
            }));
        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__mul",
            new Onnxify.MulInputOutputOptions
            {
                A = quotient,
                B = value,
                C = scaled,
            }));
        graph.AddNode(new Onnxify.Add(
            operation.Name,
            new Onnxify.AddInputOutputOptions
            {
                A = Input(operation),
                B = scaled,
                C = Output(operation),
            }));
    }
}
