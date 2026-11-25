using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::log2")]
internal sealed class Log2Operator() : TorchSharpCompositeOperator("Log2", "com.onnxify.compiler", "torch.log2")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        return TryScanLogBase(context, 2d, out operation, out consumedNodeCount);
    }

    private bool TryScanLogBase(
        CompilerOnnxSubgraphScanContext context,
        double logBase,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Log>(0, out var baseLog)
            || !context.TryGetNode<Onnxify.Log>(1, out var inputLog)
            || !context.TryGetNode<Onnxify.Div>(2, out var divide)
            || baseLog is null
            || inputLog is null
            || divide is null
            || !context.IsScalarConstant(baseLog.Input.Name, logBase)
            || !string.Equals(divide.A.Name, inputLog.Output.Name, StringComparison.Ordinal)
            || !string.Equals(divide.B.Name, baseLog.Output.Name, StringComparison.Ordinal)
            || !context.CanCollapse(3))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [inputLog.Input.Name], divide.C.Name, 3);
        consumedNodeCount = 3;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var two = AddFloatInitializer(graph, operation, "two", 2f);
        var denominator = new OnnxEdge($"{operation.Name}__log_two");
        var numerator = new OnnxEdge($"{operation.Name}__log_input");
        graph.AddNode(new Onnxify.Log(
            $"{operation.Name}__log_two_node",
            new Onnxify.LogInputOutputOptions
            {
                Input = two,
                Output = denominator,
            }));
        graph.AddNode(new Onnxify.Log(
            $"{operation.Name}__log_input_node",
            new Onnxify.LogInputOutputOptions
            {
                Input = Input(operation),
                Output = numerator,
            }));
        graph.AddNode(new Onnxify.Div(
            operation.Name,
            new Onnxify.DivInputOutputOptions
            {
                A = numerator,
                B = denominator,
                C = Output(operation),
            }));
    }
}
