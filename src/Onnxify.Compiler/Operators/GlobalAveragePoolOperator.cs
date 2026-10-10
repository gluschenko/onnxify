using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class GlobalAveragePoolOperator() : CompilerOperator<Onnxify.GlobalAveragePool>(CompilerOperatorIdentity.Onnx("GlobalAveragePool"))
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];

    protected override string PrintTorchSharp(Onnxify.GlobalAveragePool node, CompilerSourceSpan? span)
    {
        return $"torch.nn.functional.adaptive_avg_pool2d({CompilerCSharpNaming.Identifier(node.X.Name)}, new long[] {{ 1L, 1L }})";
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.GlobalAveragePool pool)
        {
            return $"GlobalAveragePool requires the generated node type '{nameof(Onnxify.GlobalAveragePool)}'.";
        }

        if (knownTypes.TryGetValue(pool.X.Name, out var inputType)
            && inputType is CompilerTensorType { Dimensions: { Count: not 4 } })
        {
            return "Compiler GlobalAveragePool currently supports rank-4 NCHW tensors only.";
        }

        return null;
    }
}
