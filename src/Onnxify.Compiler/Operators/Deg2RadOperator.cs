using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::deg2rad")]
internal sealed class Deg2RadOperator() : TorchSharpCompositeOperator("Deg2Rad", "com.onnxify.compiler", "torch.deg2rad")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Mul>(0, out var multiply)
            || multiply is null
            || !context.CanCollapse(1))
        {
            return false;
        }

        var scale = (double)(float)(Math.PI / 180d);
        var inputName = context.IsScalarConstant(multiply.A.Name, scale) ? multiply.B.Name
            : context.IsScalarConstant(multiply.B.Name, scale) ? multiply.A.Name
            : null;
        if (inputName is null)
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [inputName], multiply.C.Name, 1);
        consumedNodeCount = 1;
        return true;
    }

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
