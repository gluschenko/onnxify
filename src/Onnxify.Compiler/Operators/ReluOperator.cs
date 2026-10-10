using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::relu")]
internal sealed class ReluOperator() : ActivationOperator<Onnxify.Relu>("Relu", "torch.nn.functional.relu")
{
    protected override string PrintTorchSharp(Onnxify.Relu node, CompilerSourceSpan? span)
    {
        return $"torch.nn.functional.relu({Input(node)})";
    }
}
