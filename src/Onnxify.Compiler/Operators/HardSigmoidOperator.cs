using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::hardsigmoid")]
internal sealed class HardSigmoidOperator() : ActivationOperator<Onnxify.HardSigmoid>("HardSigmoid", "torch.nn.functional.hardsigmoid", ["alpha", "beta"])
{
    protected override string PrintTorchSharp(Onnxify.HardSigmoid node, CompilerSourceSpan? span)
    {
        return $"({Input(node)} * {Float(node.Alpha ?? 0.2f)} " +
        $"+ {Float(node.Beta ?? 0.5f)}).clamp(0.0f, 1.0f)";
    }
}
