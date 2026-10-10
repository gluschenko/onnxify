using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::clamp")]
[CompilerTorchOperator("aten::clamp.Tensor")]
internal sealed class ClipOperator() : CompilerOperator<Onnxify.Clip>(CompilerOperatorIdentity.Onnx("Clip"))
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override int MinimumInputCount => 1;
    public override int MaximumInputCount => 3;
    protected override bool AllowsEmptyOptionalInputs => true;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];

    protected override string PrintTorchSharp(Onnxify.Clip node, CompilerSourceSpan? span)
    {
        var min = node.Min is null
            ? "null"
            : $"{CompilerCSharpNaming.Identifier(node.Min.Name)}.item<float>()";
        var max = node.Max is null
            ? "null"
            : $"{CompilerCSharpNaming.Identifier(node.Max.Name)}.item<float>()";
        return $"{CompilerCSharpNaming.Identifier(node.Input.Name)}.clamp({min}, {max})";
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Clip clip)
        {
            return $"Clip requires the generated node type '{nameof(Onnxify.Clip)}'.";
        }

        if (clip.Input is null)
        {
            return "ONNX Clip requires its input tensor.";
        }

        foreach (var bound in new[] { clip.Min, clip.Max })
        {
            if (bound is null || !knownTypes.TryGetValue(bound.Name, out var type))
            {
                continue;
            }

            if (type is not CompilerTensorType tensorType
                || tensorType.ElementType != CompilerElementType.Float32
                || tensorType.Dimensions is { Count: > 0 })
            {
                return "Compiler Clip supports Float32 scalar min and max inputs only.";
            }
        }

        return null;
    }
}
