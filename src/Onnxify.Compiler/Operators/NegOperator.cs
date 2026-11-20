using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::neg")]
internal sealed class NegOperator() : UnaryOperator<Onnxify.Neg>("Neg", "neg", ["torch.neg", "Tensor.neg"])
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return ValidateKnownInputElementTypes(
            node,
            knownTypes,
            PointwiseSupportedElementTypes.Neg,
            "the ONNXScript aten_neg converter and ONNX Runtime CPU kernel intersection does not include this type.");
    }

    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.neg"),
        CompilerTorchSharpForm.Call("Tensor.neg"),
        CompilerTorchSharpForm.Unary("-"),
    ];
    protected override string PrintTorchSharp(Onnxify.Neg node, CompilerSourceSpan? span)
    {
        return $"(-{node.Inputs[0].Name})";
    }
}

