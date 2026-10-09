using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class LeakyReluOperator() : ActivationOperator<Onnxify.LeakyRelu>("LeakyRelu", "torch.nn.functional.leaky_relu", ["alpha"])
{
    protected override string PrintTorchSharp(Onnxify.LeakyRelu node, CompilerSourceSpan? span)
    {
        return $"torch.nn.functional.leaky_relu({Input(node)}, negative_slope: {Float(node.Alpha ?? 0.01f)})";
    }
}
