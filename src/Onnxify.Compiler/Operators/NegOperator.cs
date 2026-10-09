using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class NegOperator() : UnaryOperator<Onnxify.Neg>("Neg", "neg", ["torch.neg", "Tensor.neg"])
{
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

