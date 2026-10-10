using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::softplus")]
internal sealed class SoftplusOperator() : ActivationOperator<Onnxify.Softplus>("Softplus", "torch.nn.functional.softplus", includeInstance: true)
{
    protected override string PrintTorchSharp(Onnxify.Softplus node, CompilerSourceSpan? span)
    {
        return $"{Input(node)}.softplus()";
    }
}
