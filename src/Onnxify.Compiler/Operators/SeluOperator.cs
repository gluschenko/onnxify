using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class SeluOperator() : ActivationOperator<Onnxify.Selu>("Selu", "torch.nn.functional.selu", ["alpha", "gamma"])
{
    private const float DEFAULT_ALPHA = 1.6732631921768188f;
    private const float DEFAULT_GAMMA = 1.0507010221481323f;

    protected override string PrintTorchSharp(Onnxify.Selu node, CompilerSourceSpan? span)
    {
        var alpha = node.Alpha ?? DEFAULT_ALPHA;
        var gamma = node.Gamma ?? DEFAULT_GAMMA;
        var input = Input(node);
        if (alpha == DEFAULT_ALPHA && gamma == DEFAULT_GAMMA)
        {
            return $"torch.nn.functional.selu({input})";
        }

        return $"torch.where({input} > 0.0f, ({Float(gamma)} * {input}), ({Float(alpha)} * {Float(gamma)} * ({input}.exp() - 1.0f)))";
    }
}
