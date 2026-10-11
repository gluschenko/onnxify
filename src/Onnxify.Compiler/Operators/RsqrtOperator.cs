using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::rsqrt")]
internal sealed class RsqrtOperator() : TorchSharpCompositeOperator("Rsqrt", "com.onnxify.compiler", "torch.rsqrt")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Sqrt>(0, out var squareRoot)
            || !context.TryGetNode<Onnxify.Reciprocal>(1, out var reciprocal)
            || squareRoot is null
            || reciprocal is null
            || !string.Equals(squareRoot.Y.Name, reciprocal.X.Name, StringComparison.Ordinal)
            || !context.CanCollapse(2))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [squareRoot.X.Name], reciprocal.Y.Name, 2);
        consumedNodeCount = 2;
        return true;
    }

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
