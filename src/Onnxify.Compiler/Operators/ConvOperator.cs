using System.Globalization;
using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::conv2d")]
internal sealed class ConvOperator() : CompilerOperator<Onnxify.Conv>(CompilerOperatorIdentity.Onnx("Conv"))
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 3;
    protected override bool AllowsEmptyOptionalInputs => true;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.nn.functional.conv2d"),
    ];
    public override IReadOnlyList<string> AttributeNames { get; } =
    [
        "auto_pad",
        "dilations",
        "group",
        "kernel_shape",
        "pads",
        "strides",
    ];

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        if (context.Receiver is not null || invocation.Arguments.Count is < 2 or > 7)
        {
            throw context.Unsupported(expression, "conv2d supports input, weight, optional bias, stride, padding, dilation, and groups.");
        }

        var arguments = invocation.Arguments;
        var inputExpression = arguments[0];
        long[]? externalPads = null;
        if (inputExpression is CompilerInvocationExpression padInvocation
            && CompilerOperatorRegistry.TryGetMemberPath(padInvocation.Target, out var padName)
            && string.Equals(padName, "torch.nn.functional.pad", StringComparison.Ordinal))
        {
            if (padInvocation.Arguments.Count != 2)
            {
                throw context.Unsupported(inputExpression, "Compiler Conv supports only constant-zero functional.pad with an input and a static padding vector.");
            }

            var torchPads = StaticInt64Vector.ReadVector(padInvocation.Arguments[1], context, "functional.pad padding");
            if (torchPads.Length != 4)
            {
                throw context.Unsupported(padInvocation.Arguments[1], "Compiler Conv expects two-dimensional functional.pad padding in left, right, top, bottom order.");
            }

            externalPads = [torchPads[2], torchPads[0], torchPads[3], torchPads[1]];
            inputExpression = padInvocation.Arguments[0];
        }

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var weight = context.RequireTensorReference(arguments[1], OnnxName);
        var bias = arguments.Count > 2 && arguments[2] is not (CompilerLiteralExpression { Literal: CompilerNullLiteral })
            ? context.RequireTensorReference(arguments[2], OnnxName)
            : null;
        var strides = ReadPair(3, 1, "conv2d stride");
        var pads = ReadPadding(4);
        if (externalPads is not null)
        {
            if (pads.Any(static value => value != 0))
            {
                throw context.Unsupported(arguments[4], "Compiler Conv cannot combine functional.pad with non-zero conv2d padding.");
            }

            pads = externalPads;
        }
        var dilations = ReadPair(5, 1, "conv2d dilation");
        var groups = arguments.Count > 6
            ? StaticInt64Vector.ReadScalar(arguments[6], context, "conv2d groups")
            : 1L;
        if (groups < 1)
        {
            throw context.Unsupported(expression, "conv2d groups must be a positive compile-time integer.");
        }

        var node = new Onnxify.Conv(
            OnnxName.ToLowerInvariant(),
            new Onnxify.ConvInputOutputOptions
            {
                X = new OnnxEdge(input.Name),
                W = new OnnxEdge(weight.Name),
                B = bias is null ? null : new OnnxEdge(bias.Name),
                Strides = strides,
                Dilations = dilations,
                Group = groups,
                Pads = pads,
                Y = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;

        long[] ReadPair(int index, long defaultValue, string description)
        {
            if (arguments.Count <= index)
            {
                return [defaultValue, defaultValue];
            }

            var value = arguments[index];
            if (value is CompilerLiteralExpression { Literal: CompilerNullLiteral })
            {
                return [defaultValue, defaultValue];
            }

            if (value is CompilerArrayExpression)
            {
                var vector = StaticInt64Vector.ReadVector(value, context, description);
                if (vector.Length != 2)
                {
                    throw context.Unsupported(value, $"{description} must contain exactly two values.");
                }

                return vector;
            }

            var scalar = StaticInt64Vector.ReadScalar(value, context, description);
            return [scalar, scalar];
        }

        long[] ReadPadding(int index)
        {
            if (arguments.Count <= index)
            {
                return [0L, 0L, 0L, 0L];
            }

            var value = arguments[index];
            if (value is CompilerLiteralExpression { Literal: CompilerNullLiteral })
            {
                return [0L, 0L, 0L, 0L];
            }

            var padding = value is CompilerArrayExpression
                ? StaticInt64Vector.ReadVector(value, context, "conv2d padding")
                : [StaticInt64Vector.ReadScalar(value, context, "conv2d padding")];
            return padding.Length switch
            {
                1 => [padding[0], padding[0], padding[0], padding[0]],
                2 => [padding[0], padding[1], padding[0], padding[1]],
                4 => padding,
                _ => throw context.Unsupported(value, "conv2d padding must be a scalar or contain two or four values."),
            };
        }
    }

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
