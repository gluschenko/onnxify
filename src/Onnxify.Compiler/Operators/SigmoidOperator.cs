using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::sigmoid")]
internal sealed class SigmoidOperator() : ActivationOperator<Onnxify.Sigmoid>(
    "Sigmoid",
    "torch.nn.functional.sigmoid",
    includeInstance: true,
    supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu)
{
    protected override string PrintTorchSharp(Onnxify.Sigmoid node, CompilerSourceSpan? span)
    {
        return $"{Input(node)}.sigmoid()";
    }
}
