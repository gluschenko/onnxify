using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::tanh")]
internal sealed class TanhOperator() : ActivationOperator<Onnxify.Tanh>("Tanh", "torch.nn.functional.tanh", includeInstance: true)
{
    protected override string PrintTorchSharp(Onnxify.Tanh node, CompilerSourceSpan? span)
    {
        return $"{Input(node)}.tanh()";
    }
}
