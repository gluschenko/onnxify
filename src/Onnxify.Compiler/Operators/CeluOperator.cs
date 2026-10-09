using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class CeluOperator() : ActivationOperator<Onnxify.Celu>("Celu", "torch.nn.functional.celu", ["alpha"])
{
    protected override string PrintTorchSharp(Onnxify.Celu node, CompilerSourceSpan? span)
    {
        return $"torch.nn.functional.celu({Input(node)}, alpha: {Float(node.Alpha ?? 1f)})";
    }
}
