using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::frac")]
internal sealed class FracOperator() : TorchSharpCompositeOperator("Frac", "com.onnxify.compiler", "torch.frac")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Abs>(0, out var absolute)
            || !context.TryGetNode<Onnxify.Floor>(1, out var floor)
            || !context.TryGetNode<Onnxify.Sign>(2, out var sign)
            || !context.TryGetNode<Onnxify.Mul>(3, out var multiply)
            || !context.TryGetNode<Onnxify.Sub>(4, out var subtract)
            || absolute is null || floor is null || sign is null || multiply is null || subtract is null
            || !string.Equals(floor.X.Name, absolute.Y.Name, StringComparison.Ordinal)
            || !string.Equals(sign.Input.Name, absolute.X.Name, StringComparison.Ordinal)
            || !string.Equals(multiply.A.Name, floor.Y.Name, StringComparison.Ordinal)
            || !string.Equals(multiply.B.Name, sign.Output.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.A.Name, absolute.X.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.B.Name, multiply.C.Name, StringComparison.Ordinal)
            || !context.CanCollapse(5))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [absolute.X.Name], subtract.C.Name, 5);
        consumedNodeCount = 5;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var input = Input(operation);
        var absolute = new OnnxEdge($"{operation.Name}__abs");
        var floor = new OnnxEdge($"{operation.Name}__floor");
        var sign = new OnnxEdge($"{operation.Name}__sign");
        var product = new OnnxEdge($"{operation.Name}__product");
        graph.AddNode(new Onnxify.Abs($"{operation.Name}__abs_node", new Onnxify.AbsInputOutputOptions { X = input, Y = absolute }));
        graph.AddNode(new Onnxify.Floor($"{operation.Name}__floor_node", new Onnxify.FloorInputOutputOptions { X = absolute, Y = floor }));
        graph.AddNode(new Onnxify.Sign($"{operation.Name}__sign_node", new Onnxify.SignInputOutputOptions { Input = input, Output = sign }));
        graph.AddNode(new Onnxify.Mul($"{operation.Name}__mul_node", new Onnxify.MulInputOutputOptions { A = floor, B = sign, C = product }));
        graph.AddNode(new Onnxify.Sub(operation.Name, new Onnxify.SubInputOutputOptions { A = input, B = product, C = Output(operation) }));
    }
}
