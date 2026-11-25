using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::mish")]
internal sealed class MishOperator() : ActivationOperator<Onnxify.Mish>("Mish", "torch.nn.functional.mish", supportedInputTypes: PointwiseSupportedElementTypes.Float32ActivationCpu)
{
    protected override string PrintTorchSharp(Onnxify.Mish node, CompilerSourceSpan? span)
    {
        return $"({Input(node)} * {Input(node)}.softplus().tanh())";
    }
}
