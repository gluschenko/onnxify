using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::addcdiv")]
internal sealed class AddCdivOperator() : TorchSharpCompositeOperator("AddCdiv", "com.onnxify.compiler", "torch.addcdiv")
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
        if (!context.TryGetNode<Onnxify.Div>(0, out var divide)
            || !context.TryGetNode<Onnxify.Mul>(1, out var scale)
            || !context.TryGetNode<Onnxify.Add>(2, out var add)
            || divide is null || scale is null || add is null
            || !context.IsScalarInitializer(scale.B.Name)
            || !string.Equals(divide.C.Name, scale.A.Name, StringComparison.Ordinal)
            || !string.Equals(scale.C.Name, add.B.Name, StringComparison.Ordinal)
            || !context.CanCollapse(3))
        {
            return false;
        }

        operation = context.CreateCompositeStep(
            this,
            [add.A.Name, divide.A.Name, divide.B.Name, scale.B.Name],
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
        return $"torch.addcdiv({input}, {first}, {second}, value: {value})";
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var quotient = new OnnxEdge($"{operation.Name}__quotient");
        var scaled = new OnnxEdge($"{operation.Name}__scaled");
        var value = Input(operation, 3);

        graph.AddNode(new Onnxify.Div(
            $"{operation.Name}__div",
            new Onnxify.DivInputOutputOptions
            {
                A = Input(operation, 1),
                B = Input(operation, 2),
                C = quotient,
            }));
        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__mul",
            new Onnxify.MulInputOutputOptions
            {
                A = quotient,
                B = value,
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
