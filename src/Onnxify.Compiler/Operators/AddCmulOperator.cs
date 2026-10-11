using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::addcmul")]
internal sealed class AddCmulOperator() : TorchSharpCompositeOperator("AddCmul", "com.onnxify.compiler", "torch.addcmul")
{
    public override int MinimumInputCount => 4;
    public override int MaximumInputCount => 4;

    protected override bool AllowsScalarOperand(int index)
    {
        return index == 3;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var firstProduct = new OnnxEdge($"{operation.Name}__first_product");
        var scaled = new OnnxEdge($"{operation.Name}__scaled");
        var value = Input(operation, 3);

        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__scale",
            new Onnxify.MulInputOutputOptions
            {
                A = value,
                B = Input(operation, 1),
                C = firstProduct,
            }));
        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__mul",
            new Onnxify.MulInputOutputOptions
            {
                A = firstProduct,
                B = Input(operation, 2),
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
