using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::erfc")]
[CompilerTorchOperator("aten::special_erfc")]
internal sealed class ErfcOperator() : TorchSharpCompositeOperator("Erfc", "com.onnxify.compiler", "torch.erfc")
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
        if (context.TryGetNode<Onnxify.Erf>(0, out var erf)
            && context.TryGetNode<Onnxify.Sub>(1, out var subtract)
            && erf is not null
            && subtract is not null
            && context.IsScalarConstant(subtract.A.Name, 1d)
            && string.Equals(erf.Output.Name, subtract.B.Name, StringComparison.Ordinal)
            && context.CanCollapse(2))
        {
            operation = context.CreateCompositeStep(this, [erf.Input.Name], subtract.C.Name, 2);
            consumedNodeCount = 2;
            return true;
        }

        if (context.TryGetNode<Onnxify.Erf>(0, out erf)
            && erf is not null
            && context.TryGetNode<OnnxNode>(1, out var castLikeAfterErf)
            && castLikeAfterErf is not null
            && IsCastLike(castLikeAfterErf)
            && context.IsScalarConstant(castLikeAfterErf.Inputs[0].Name, 1d)
            && context.TryGetNode<Onnxify.Sub>(2, out var subtractAfterCast)
            && subtractAfterCast is not null
            && string.Equals(castLikeAfterErf.Inputs[1].Name, erf.Input.Name, StringComparison.Ordinal)
            && string.Equals(castLikeAfterErf.Outputs[0].Name, subtractAfterCast.A.Name, StringComparison.Ordinal)
            && string.Equals(erf.Output.Name, subtractAfterCast.B.Name, StringComparison.Ordinal)
            && context.CanCollapse(3))
        {
            operation = context.CreateCompositeStep(this, [erf.Input.Name], subtractAfterCast.C.Name, 3);
            consumedNodeCount = 3;
            return true;
        }

        if (!context.TryGetNode<OnnxNode>(0, out var castLike)
            || castLike is null
            || !IsCastLike(castLike)
            || !context.IsScalarConstant(castLike.Inputs[0].Name, 1d)
            || !context.TryGetNode<Onnxify.Erf>(1, out var typedErf)
            || typedErf is null
            || !context.TryGetNode<Onnxify.Sub>(2, out var typedSubtract)
            || typedSubtract is null
            || !string.Equals(castLike.Inputs[1].Name, typedErf.Input.Name, StringComparison.Ordinal)
            || !string.Equals(castLike.Outputs[0].Name, typedSubtract.A.Name, StringComparison.Ordinal)
            || !string.Equals(typedErf.Output.Name, typedSubtract.B.Name, StringComparison.Ordinal)
            || !context.CanCollapse(3))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [typedErf.Input.Name], typedSubtract.C.Name, 3);
        consumedNodeCount = 3;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var erf = new OnnxEdge($"{operation.Name}__erf");
        graph.AddNode(new Onnxify.Erf(
            $"{operation.Name}__erf_node",
            new Onnxify.ErfInputOutputOptions { Input = Input(operation), Output = erf }));
        graph.AddNode(new Onnxify.Sub(
            operation.Name,
            new Onnxify.SubInputOutputOptions
            {
                A = AddFloatInitializer(graph, operation, "one", 1d),
                B = erf,
                C = Output(operation),
            }));
    }

    private static bool IsCastLike(OnnxNode node) =>
        string.Equals(node.Domain, string.Empty, StringComparison.Ordinal)
        && string.Equals(node.OpType, "CastLike", StringComparison.Ordinal)
        && node.Inputs.Count == 2
        && node.Outputs.Count == 1
        && node.Attributes.All(static attribute =>
            string.Equals(attribute.Name, "saturate", StringComparison.Ordinal)
            && attribute.GetValue() is long value
            && value == 1);
}
