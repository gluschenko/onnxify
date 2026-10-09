using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class ThresholdedReluOperator() : ActivationOperator<Onnxify.ThresholdedRelu>("ThresholdedRelu", "torch.nn.functional.threshold", ["alpha"], fixedArguments: [0f])
{
    protected override string PrintTorchSharp(Onnxify.ThresholdedRelu node, CompilerSourceSpan? span)
    {
        return $"torch.nn.functional.threshold({Input(node)}, {Float(node.Alpha ?? 1f)}, 0.0f)";
    }
}

