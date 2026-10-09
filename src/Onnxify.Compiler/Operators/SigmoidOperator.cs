using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class SigmoidOperator() : ActivationOperator<Onnxify.Sigmoid>("Sigmoid", "torch.nn.functional.sigmoid", includeInstance: true)
{
    protected override string PrintTorchSharp(Onnxify.Sigmoid node, CompilerSourceSpan? span)
    {
        return $"{Input(node)}.sigmoid()";
    }
}
