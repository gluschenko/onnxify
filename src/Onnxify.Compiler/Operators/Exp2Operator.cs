using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::exp2")]
internal sealed class Exp2Operator() : TorchSharpCompositeOperator("Exp2", "com.onnxify.compiler", "torch.exp2")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Pow>(0, out var power)
            || power is null
            || !context.IsScalarConstant(power.X.Name, 2d)
            || !context.CanCollapse(1))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [power.Y.Name], power.Z.Name, 1);
        consumedNodeCount = 1;
        return true;
    }

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
