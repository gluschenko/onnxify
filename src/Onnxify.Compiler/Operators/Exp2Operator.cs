using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::exp2")]
internal sealed class Exp2Operator() : TorchSharpCompositeOperator("Exp2", "com.onnxify.compiler", "torch.exp2")
{
    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var basis = AddFloatInitializer(graph, operation, "basis", 2f);
        graph.AddNode(new Onnxify.Pow(
            operation.Name,
            new Onnxify.PowInputOutputOptions
            {
                X = basis,
                Y = Input(operation),
                Z = Output(operation),
            }));
    }
}
