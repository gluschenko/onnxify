using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::logical_not")]
internal sealed class NotOperator() : UnaryOperator<Onnxify.Not>("Not", "logical_not", ["torch.logical_not", "Tensor.logical_not"])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
    public override IReadOnlyList<CompilerElementType?> InputElementTypes => [CompilerElementType.Boolean];
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.logical_not"),
        CompilerTorchSharpForm.Call("Tensor.logical_not"),
        CompilerTorchSharpForm.Unary("!"),
    ];
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return ValidateKnownInputElementTypes(
            node,
            knownTypes,
            PointwiseSupportedElementTypes.Boolean,
            "the ONNX Runtime CPU logical kernel accepts Boolean inputs only.");
    }

    protected override string PrintTorchSharp(Onnxify.Not node, CompilerSourceSpan? span)
    {
        return $"torch.logical_not({node.Inputs[0].Name})";
    }
}

