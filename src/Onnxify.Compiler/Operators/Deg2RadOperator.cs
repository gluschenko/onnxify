using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::deg2rad")]
internal sealed class Deg2RadOperator() : TorchSharpCompositeOperator("Deg2Rad", "com.onnxify.compiler", "torch.deg2rad")
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
        const double scale = Math.PI / 180d;
        if (context.TryGetNode<Onnxify.Mul>(0, out var multiply)
            && multiply is not null
            && context.CanCollapse(1))
        {
            var inputName = IsScale(context, multiply.A.Name, scale) ? multiply.B.Name
                : IsScale(context, multiply.B.Name, scale) ? multiply.A.Name
                : null;
            if (inputName is not null)
            {
                operation = context.CreateCompositeStep(this, [inputName], multiply.C.Name, 1);
                consumedNodeCount = 1;
                return true;
            }
        }

        if (!context.TryGetNode<OnnxNode>(0, out var castLike)
            || castLike is null
            || !IsCastLike(castLike)
            || !IsScale(context, castLike.Inputs[0].Name, scale)
            || !context.TryGetNode<Onnxify.Mul>(1, out var typedMultiply)
            || typedMultiply is null
            || !string.Equals(castLike.Outputs[0].Name, typedMultiply.B.Name, StringComparison.Ordinal)
            || !string.Equals(castLike.Inputs[1].Name, typedMultiply.A.Name, StringComparison.Ordinal)
            || !context.CanCollapse(2))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [typedMultiply.A.Name], typedMultiply.C.Name, 2);
        consumedNodeCount = 2;
        return true;
    }

    private static bool IsScale(CompilerOnnxSubgraphScanContext context, string name, double scale)
    {
        return context.IsScalarConstant(name, scale)
            || context.IsScalarConstant(name, (double)(float)scale);
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

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var scale = AddFloatInitializer(graph, operation, "scale", Math.PI / 180d);
        graph.AddNode(new Onnxify.Mul(
            operation.Name,
            new Onnxify.MulInputOutputOptions
            {
                A = Input(operation),
                B = scale,
                C = Output(operation),
            }));
    }
}
