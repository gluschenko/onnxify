using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class GeluOperator() : ActivationOperator<Onnxify.Gelu>("Gelu", "torch.nn.functional.gelu", ["approximate"])
{
    protected override string PrintTorchSharp(Onnxify.Gelu node, CompilerSourceSpan? span)
    {
        var mode = node.Approximate ?? "none";
        if (mode is not ("none" or "tanh"))
        {
            throw Unsupported(span, $"Gelu approximate mode '{mode}' is unsupported.");
        }

        return $"torch.nn.functional.gelu({Input(node)}, approximate: global::TorchSharp.Modules.GELU.Approximate.{mode})";
    }
}
