using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::deg2rad")]
internal sealed class Deg2RadOperator() : TorchSharpCompositeOperator("Deg2Rad", "com.onnxify.compiler", "torch.deg2rad")
{
    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var scale = AddFloatInitializer(graph, operation, "scale", (float)(Math.PI / 180d));
        graph.AddNode(new Onnxify.Mul(
            operation.Name,
            new Onnxify.MulInputOutputOptions
            {
                A = Input(operation),
                B = scale,
                C = Output(operation),
            }));
    }
}
