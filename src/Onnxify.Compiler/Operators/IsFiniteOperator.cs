using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::isfinite")]
internal sealed class IsFiniteOperator() : TorchSharpCompositeOperator("IsFinite", "com.onnxify.compiler", "torch.isfinite")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return ValidateFloatingPointTensorInputs(node, knownTypes,
            CompilerElementType.Float16, CompilerElementType.Float32, CompilerElementType.Float64);
    }

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.IsInf>(0, out var isInf)
            || !context.TryGetNode<Onnxify.Not>(1, out var notInf)
            || !context.TryGetNode<Onnxify.IsNaN>(2, out var isNaN)
            || !context.TryGetNode<Onnxify.Not>(3, out var notNaN)
            || !context.TryGetNode<Onnxify.And>(4, out var and)
            || isInf is null || notInf is null || isNaN is null || notNaN is null || and is null
            || isInf.DetectNegative is not (null or 1) || isInf.DetectPositive is not (null or 1)
            || !string.Equals(isInf.Y.Name, notInf.Inputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(isNaN.Y.Name, notNaN.Inputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(notInf.Outputs[0].Name, and.A.Name, StringComparison.Ordinal)
            || !string.Equals(notNaN.Outputs[0].Name, and.B.Name, StringComparison.Ordinal)
            || !string.Equals(isInf.X.Name, isNaN.X.Name, StringComparison.Ordinal)
            || !context.CanCollapse(5))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [isInf.X.Name], and.C.Name, 5);
        consumedNodeCount = 5;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var isInf = new Onnxify.IsInf($"{operation.Name}__isinf", new Onnxify.IsInfInputOutputOptions
        {
            X = Input(operation),
            Y = new OnnxEdge($"{operation.Name}__isinf_result"),
        });
        var notInf = new Onnxify.Not($"{operation.Name}__notinf", new Onnxify.NotInputOutputOptions
        {
            X = isInf.Y,
            Y = new OnnxEdge($"{operation.Name}__notinf_result"),
        });
        var isNaN = new Onnxify.IsNaN($"{operation.Name}__isnan", new Onnxify.IsNaNInputOutputOptions
        {
            X = Input(operation),
            Y = new OnnxEdge($"{operation.Name}__isnan_result"),
        });
        var notNaN = new Onnxify.Not($"{operation.Name}__notnan", new Onnxify.NotInputOutputOptions
        {
            X = isNaN.Y,
            Y = new OnnxEdge($"{operation.Name}__notnan_result"),
        });
        graph.AddNode(isInf);
        graph.AddNode(notInf);
        graph.AddNode(isNaN);
        graph.AddNode(notNaN);
        graph.AddNode(new Onnxify.And($"{operation.Name}__and", new Onnxify.AndInputOutputOptions
        {
            A = notInf.Y,
            B = notNaN.Y,
            C = Output(operation),
        }));
    }
}
