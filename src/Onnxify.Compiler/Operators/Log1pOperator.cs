using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::log1p")]
internal sealed class Log1pOperator() : TorchSharpCompositeOperator("Log1p", "com.onnxify.compiler", "torch.log1p")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Add>(0, out var add)
            || !context.TryGetNode<Onnxify.Log>(1, out var log)
            || add is null
            || log is null
            || !string.Equals(add.C.Name, log.Input.Name, StringComparison.Ordinal)
            || !context.CanCollapse(2))
        {
            return false;
        }

        var inputName = context.IsScalarConstant(add.A.Name, 1d) ? add.B.Name
            : context.IsScalarConstant(add.B.Name, 1d) ? add.A.Name
            : null;
        if (inputName is null)
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [inputName], log.Output.Name, 2);
        consumedNodeCount = 2;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var one = AddFloatInitializer(graph, operation, "one", 1f);
        var sum = new OnnxEdge($"{operation.Name}__sum");
        graph.AddNode(new Onnxify.Add(
            $"{operation.Name}__add",
            new Onnxify.AddInputOutputOptions
            {
                A = Input(operation),
                B = one,
                C = sum,
            }));
        graph.AddNode(new Onnxify.Log(
            operation.Name,
            new Onnxify.LogInputOutputOptions
            {
                Input = sum,
                Output = Output(operation),
            }));
    }
}
