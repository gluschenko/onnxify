using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::pow.Tensor_Tensor")]
[CompilerTorchOperator("aten::pow.Tensor_Scalar")]
[CompilerTorchOperator("aten::pow.Scalar")]
internal sealed class PowOperator() : BinaryOperator<Onnxify.Pow>("Pow", "pow", true)
{
    protected override string PrintTorchSharp(Onnxify.Pow node, CompilerSourceSpan? span)
    {
        return $"{node.Inputs[0].Name}.pow({node.Inputs[1].Name})";
    }
}
