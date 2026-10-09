using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class PReluOperator() : ActivationOperator<Onnxify.PRelu>("PRelu", "torch.nn.functional.prelu", inputCount: 2)
{
    protected override string PrintTorchSharp(Onnxify.PRelu node, CompilerSourceSpan? span)
    {
        return $"torch.nn.functional.prelu({Input(node)}, {node.Inputs[1].Name})";
    }
}
