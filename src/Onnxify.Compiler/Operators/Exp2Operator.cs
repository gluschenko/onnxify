using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::exp2")]
internal sealed class Exp2Operator() : TorchSharpCompositeOperator("Exp2", "com.onnxify.compiler", "torch.exp2")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    protected override IReadOnlyCollection<CompilerElementType> SupportedTensorInputTypes =>
    [CompilerElementType.Float16, CompilerElementType.Float32, CompilerElementType.Float64];

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (context.TryGetNode<Onnxify.Pow>(0, out var power)
            && power is not null
            && context.IsScalarConstant(power.X.Name, 2d)
            && context.CanCollapse(1))
        {
            operation = context.CreateCompositeStep(this, [power.Y.Name], power.Z.Name, 1);
            consumedNodeCount = 1;
            return true;
        }

        if (!context.TryGetNode<OnnxNode>(0, out var castLike)
            || castLike is null
            || !string.Equals(castLike.OpType, "CastLike", StringComparison.Ordinal)
            || castLike.Inputs.Count != 2
            || castLike.Outputs.Count != 1
            || !context.IsScalarConstant(castLike.Inputs[0].Name, 2d)
            || !context.TryGetNode<Onnxify.Pow>(1, out power)
            || power is null
            || !string.Equals(castLike.Outputs[0].Name, power.X.Name, StringComparison.Ordinal)
            || !string.Equals(castLike.Inputs[1].Name, power.Y.Name, StringComparison.Ordinal)
            || !context.CanCollapse(2))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [power.Y.Name], power.Z.Name, 2);
        consumedNodeCount = 2;
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
