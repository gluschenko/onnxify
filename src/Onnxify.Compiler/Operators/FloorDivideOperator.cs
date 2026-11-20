using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::floor_divide")]
internal sealed class FloorDivideOperator() : TorchSharpCompositeOperator("FloorDivide", "com.onnxify.compiler", "torch.floor_divide")
{
    protected override IReadOnlyCollection<CompilerElementType> SupportedTensorInputTypes =>
    [
        CompilerElementType.Float32, CompilerElementType.Float64,
        CompilerElementType.Int8, CompilerElementType.Int16, CompilerElementType.Int32, CompilerElementType.Int64,
        CompilerElementType.UInt8, CompilerElementType.UInt16, CompilerElementType.UInt32, CompilerElementType.UInt64,
    ];

    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;
    public override bool SupportsMultidirectionalBroadcast => true;

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        return $"torch.floor_divide({CompilerCSharpNaming.Identifier(operation.Node.Inputs[0].Name)}, {CompilerCSharpNaming.Identifier(operation.Node.Inputs[1].Name)})";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        throw context.Unsupported(expression, "TorchSharp floor_divide export is not enabled because the source scanner cannot specialize the ONNX lowering by runtime tensor dtype.");
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        throw Unsupported(operation.Span, "FloorDivide is import-only; TorchSharp export cannot specialize its lowering by runtime tensor dtype.");
    }

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;

        if (TryScanSignedIntegerPattern(context, out operation, out consumedNodeCount))
        {
            return true;
        }

        if (context.TryGetNode<Onnxify.Div>(0, out var divide)
            && divide is not null
            && TryGetElementType(context.Graph, divide.A.Name, out var elementType))
        {
            if (elementType == typeof(float) || elementType == typeof(double))
            {
                if (context.TryGetNode<Onnxify.Floor>(1, out var floor)
                    && floor is not null
                    && string.Equals(divide.C.Name, floor.X.Name, StringComparison.Ordinal)
                    && context.CanCollapse(2))
                {
                    operation = context.CreateCompositeStep(this, [divide.A.Name, divide.B.Name], floor.Y.Name, 2);
                    consumedNodeCount = 2;
                    return true;
                }
            }
            else if (elementType == typeof(byte)
                || elementType == typeof(ushort)
                || elementType == typeof(uint)
                || elementType == typeof(ulong))
            {
                operation = context.CreateCompositeStep(this, [divide.A.Name, divide.B.Name], divide.C.Name, 1);
                consumedNodeCount = 1;
                return true;
            }
        }

        return false;
    }

    private bool TryScanSignedIntegerPattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Sign>(0, out var signA)
            || !context.TryGetNode<Onnxify.Sign>(1, out var signB)
            || !context.TryGetNode<Onnxify.Equal>(2, out var equalSigns)
            || !context.TryGetNode<Onnxify.Not>(3, out var differentSigns)
            || !context.TryGetNode<Onnxify.Mod>(4, out var modulo)
            || !context.TryGetNode<Onnxify.Cast>(5, out var remainderIsNonZero)
            || !context.TryGetNode<Onnxify.And>(6, out var correctionCondition)
            || !context.TryGetNode<Onnxify.Cast>(7, out var correction)
            || !context.TryGetNode<Onnxify.Div>(8, out var divide)
            || !context.TryGetNode<Onnxify.Sub>(9, out var subtract)
            || signA is null || signB is null || equalSigns is null || differentSigns is null
            || modulo is null || remainderIsNonZero is null || correctionCondition is null || correction is null
            || divide is null || subtract is null
            || modulo.Fmod is not (null or 0)
            || remainderIsNonZero.To != 9
            || correction.To != GetTensorElementTypeCode(context.Graph, modulo.A.Name)
            || !string.Equals(signB.Input.Name, modulo.B.Name, StringComparison.Ordinal)
            || !string.Equals(signA.Input.Name, modulo.A.Name, StringComparison.Ordinal)
            || !string.Equals(signA.Output.Name, equalSigns.A.Name, StringComparison.Ordinal)
            || !string.Equals(signB.Output.Name, equalSigns.B.Name, StringComparison.Ordinal)
            || !string.Equals(equalSigns.C.Name, differentSigns.Inputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(differentSigns.Outputs[0].Name, correctionCondition.A.Name, StringComparison.Ordinal)
            || !string.Equals(modulo.C.Name, remainderIsNonZero.Input.Name, StringComparison.Ordinal)
            || !string.Equals(remainderIsNonZero.Output.Name, correctionCondition.B.Name, StringComparison.Ordinal)
            || !string.Equals(correctionCondition.C.Name, correction.Input.Name, StringComparison.Ordinal)
            || !string.Equals(signA.Input.Name, divide.A.Name, StringComparison.Ordinal)
            || !string.Equals(signB.Input.Name, divide.B.Name, StringComparison.Ordinal)
            || !string.Equals(divide.C.Name, subtract.A.Name, StringComparison.Ordinal)
            || !string.Equals(correction.Output.Name, subtract.B.Name, StringComparison.Ordinal)
            || !context.CanCollapse(10))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [divide.A.Name, divide.B.Name], subtract.C.Name, 10);
        consumedNodeCount = 10;
        return true;
    }

    private static bool TryGetElementType(OnnxGraph graph, string name, out Type? elementType)
    {
        elementType = graph.GetValue(name) is OnnxValue { Type: OnnxTensorType tensorType }
            ? tensorType.Type
            : null;
        return elementType is not null;
    }

    private static long GetTensorElementTypeCode(OnnxGraph graph, string name)
    {
        if (!TryGetElementType(graph, name, out var type))
        {
            return -1;
        }

        return type == typeof(float) ? 1
            : type == typeof(byte) ? 2
            : type == typeof(sbyte) ? 3
            : type == typeof(ushort) ? 4
            : type == typeof(short) ? 5
            : type == typeof(int) ? 6
            : type == typeof(long) ? 7
            : type == typeof(bool) ? 9
            : type == typeof(double) ? 11
            : type == typeof(uint) ? 12
            : type == typeof(ulong) ? 13
            : -1;
    }
}
