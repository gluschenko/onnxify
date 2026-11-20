using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::expm1")]
[CompilerTorchOperator("aten::special_expm1")]
internal sealed class Expm1Operator() : TorchSharpCompositeOperator("Expm1", "com.onnxify.compiler", "torch.expm1")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Exp>(0, out var exp)
            || !context.TryGetNode<Onnxify.Sub>(1, out var subtract)
            || exp is null
            || subtract is null
            || !string.Equals(exp.Output.Name, subtract.A.Name, StringComparison.Ordinal)
            || !context.IsScalarConstant(subtract.B.Name, 1d)
            || !context.CanCollapse(2))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [exp.Input.Name], subtract.C.Name, 2);
        consumedNodeCount = 2;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var exp = new OnnxEdge($"{operation.Name}__exp");
        graph.AddNode(new Onnxify.Exp(
            $"{operation.Name}__exp_node",
            new Onnxify.ExpInputOutputOptions { Input = Input(operation), Output = exp }));
        graph.AddNode(new Onnxify.Sub(
            operation.Name,
            new Onnxify.SubInputOutputOptions
            {
                A = exp,
                B = AddFloatInitializer(graph, operation, "one", 1d),
                C = Output(operation),
            }));
    }
}
