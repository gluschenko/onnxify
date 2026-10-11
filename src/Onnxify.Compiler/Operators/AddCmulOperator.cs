using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::addcmul")]
internal sealed class AddCmulOperator() : TorchSharpCompositeOperator("AddCmul", "com.onnxify.compiler", "torch.addcmul")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override int MinimumInputCount => 4;
    public override int MaximumInputCount => 4;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Mul>(0, out var product)
            || !context.TryGetNode<Onnxify.Mul>(1, out var scale)
            || !context.TryGetNode<Onnxify.Add>(2, out var add)
            || product is null || scale is null || add is null
            || !context.IsScalarInitializer(product.A.Name)
            || !string.Equals(product.C.Name, scale.A.Name, StringComparison.Ordinal)
            || !string.Equals(scale.C.Name, add.B.Name, StringComparison.Ordinal)
            || !context.CanCollapse(3))
        {
            return false;
        }

        operation = context.CreateCompositeStep(
            this,
            [add.A.Name, product.B.Name, scale.B.Name, product.A.Name],
            add.C.Name,
            3);
        consumedNodeCount = 3;
        return true;
    }

    protected override bool AllowsScalarOperand(int index)
    {
        return index == 3;
    }

    public override string PrintTorchSharp(
        CompilerOnnxStep operation,
        CompilerTorchSharpPrintContext context)
    {
        var input = TorchSharpInput(operation, 0, context);
        var first = TorchSharpInput(operation, 1, context);
        var second = TorchSharpInput(operation, 2, context);
        var value = TorchSharpInput(operation, 3, context);
        return $"torch.addcmul({input}, {first}, {second}, value: {value})";
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var firstProduct = new OnnxEdge($"{operation.Name}__first_product");
        var scaled = new OnnxEdge($"{operation.Name}__scaled");
        var value = Input(operation, 3);

        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__scale",
            new Onnxify.MulInputOutputOptions
            {
                A = value,
                B = Input(operation, 1),
                C = firstProduct,
            }));
        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__mul",
            new Onnxify.MulInputOutputOptions
            {
                A = firstProduct,
                B = Input(operation, 2),
                C = scaled,
            }));
        graph.AddNode(new Onnxify.Add(
            operation.Name,
            new Onnxify.AddInputOutputOptions
            {
                A = Input(operation),
                B = scaled,
                C = Output(operation),
            }));
    }
}
