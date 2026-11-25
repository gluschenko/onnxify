using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::elu")]
internal sealed class EluOperator() : ActivationOperator<Onnxify.Elu>("Elu", "torch.nn.functional.elu", ["alpha"], supportedInputTypes: PointwiseSupportedElementTypes.Float32ActivationCpu)
{
    protected override string PrintTorchSharp(Onnxify.Elu node, CompilerSourceSpan? span)
    {
        return $"torch.nn.functional.elu({Input(node)}, alpha: {Float(node.Alpha ?? 1f)})";
    }
}
