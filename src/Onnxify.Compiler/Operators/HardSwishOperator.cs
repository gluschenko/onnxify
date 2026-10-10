using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::hardswish")]
internal sealed class HardSwishOperator() : ActivationOperator<Onnxify.HardSwish>("HardSwish", "torch.nn.functional.hardswish")
{
    protected override string PrintTorchSharp(Onnxify.HardSwish node, CompilerSourceSpan? span)
    {
        return $"({Input(node)} * ({Input(node)} + 3.0f).clamp(0.0f, 6.0f) / 6.0f)";
    }
}
