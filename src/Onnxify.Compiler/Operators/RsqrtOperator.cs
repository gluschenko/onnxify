using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::rsqrt")]
internal sealed class RsqrtOperator() : TorchSharpCompositeOperator("Rsqrt", "com.onnxify.compiler", "torch.rsqrt")
{
    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var squareRoot = new OnnxEdge($"{operation.Name}__sqrt");
        graph.AddNode(new Onnxify.Sqrt(
            $"{operation.Name}__sqrt_node",
            new Onnxify.SqrtInputOutputOptions
            {
                X = Input(operation),
                Y = squareRoot,
            }));
        graph.AddNode(new Onnxify.Reciprocal(
            operation.Name,
            new Onnxify.ReciprocalInputOutputOptions
            {
                X = squareRoot,
                Y = Output(operation),
            }));
    }
}
