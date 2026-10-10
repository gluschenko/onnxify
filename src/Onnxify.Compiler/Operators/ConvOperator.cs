using System.Globalization;
using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class ConvOperator() : CompilerOperator<Onnxify.Conv>(CompilerOperatorIdentity.Onnx("Conv"))
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 3;
    protected override bool AllowsEmptyOptionalInputs => true;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];
    public override IReadOnlyList<string> AttributeNames { get; } =
    [
        "auto_pad",
        "dilations",
        "group",
        "kernel_shape",
        "pads",
        "strides",
    ];

    protected override string PrintTorchSharp(Onnxify.Conv node, CompilerSourceSpan? span)
    {
        var x = CompilerCSharpNaming.Identifier(node.X.Name);
        var weight = CompilerCSharpNaming.Identifier(node.W.Name);
        var bias = node.B is null ? "null" : CompilerCSharpNaming.Identifier(node.B.Name);
        var strides = node.Strides ?? [1L, 1L];
        var dilations = node.Dilations ?? [1L, 1L];
        var pads = NormalizePads(node.Pads ?? [0L, 0L, 0L, 0L], span);

        if (pads[0] != pads[2] || pads[1] != pads[3])
        {
            x = $"torch.nn.functional.pad({x}, {LongArray([pads[1], pads[3], pads[0], pads[2]])})";
            pads = [0L, 0L, 0L, 0L];
        }

        var padding = new[] { pads[0], pads[1] };
        return $"torch.nn.functional.conv2d({x}, {weight}, {bias}, {LongArray(strides)}, {LongArray(padding)}, {LongArray(dilations)}, {node.Group ?? 1L}L)";
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Conv conv)
        {
            return $"Conv requires the generated node type '{nameof(Onnxify.Conv)}'.";
        }

        if (conv.AutoPad is not (null or "NOTSET" or "VALID"))
        {
            return $"Compiler Conv does not support auto_pad '{conv.AutoPad}'.";
        }

        if (conv.Group is < 1)
        {
            return "ONNX Conv attribute 'group' must be positive.";
        }

        if (conv.Strides is { Length: not 2 } || conv.Dilations is { Length: not 2 })
        {
            return "Compiler Conv supports two-dimensional strides and dilations only.";
        }

        if (conv.KernelShape is { Length: not 2 })
        {
            return "Compiler Conv supports two-dimensional kernel_shape only.";
        }

        if (conv.Pads is { Length: not 0 and not 2 and not 4 })
        {
            return "Compiler Conv supports two or four explicit padding values.";
        }

        var input = GetTensorType(conv.X.Name, knownTypes);
        var weight = GetTensorType(conv.W.Name, knownTypes);
        var bias = conv.B is null ? null : GetTensorType(conv.B.Name, knownTypes);
        foreach (var tensor in new[] { input, weight, bias })
        {
            if (tensor is not null && tensor.ElementType != CompilerElementType.Float32)
            {
                return "Compiler Conv currently supports Float32 tensors only.";
            }
        }

        if (input?.Dimensions is { Count: not 4 } || weight?.Dimensions is { Count: not 4 })
        {
            return "Compiler Conv supports rank-4 NCHW input and weights only.";
        }

        return null;
    }

    private static CompilerTensorType? GetTensorType(
        string name,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return knownTypes.TryGetValue(name, out var type) ? type as CompilerTensorType : null;
    }

    private static long[] NormalizePads(long[] pads, CompilerSourceSpan? span)
    {
        return pads.Length switch
        {
            0 => [0L, 0L, 0L, 0L],
            2 => [pads[0], pads[1], pads[0], pads[1]],
            4 => pads,
            _ => throw Unsupported(span, "Compiler Conv expects two or four explicit padding values."),
        };
    }

    private static string LongArray(IEnumerable<long> values)
    {
        return $"new long[] {{ {string.Join(", ", values.Select(static value => value.ToString(CultureInfo.InvariantCulture) + "L"))} }}";
    }
}
