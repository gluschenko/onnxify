using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::rad2deg")]
internal sealed class Rad2DegOperator() : TorchSharpCompositeOperator("Rad2Deg", "com.onnxify.compiler", "torch.rad2deg")
{
    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var scale = AddFloatInitializer(graph, operation, "scale", (float)(180d / Math.PI));
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
