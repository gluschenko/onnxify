using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class SoftsignOperator() : ActivationOperator<Onnxify.Softsign>("Softsign", "torch.nn.functional.softsign")
{
    protected override string PrintTorchSharp(Onnxify.Softsign node, CompilerSourceSpan? span)
    {
        return $"({Input(node)} / (1.0f + {Input(node)}.abs()))";
    }
}
