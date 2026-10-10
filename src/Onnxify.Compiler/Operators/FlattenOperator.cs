using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::flatten.using_ints")]
internal sealed class FlattenOperator() : CompilerOperator<Onnxify.Flatten>(CompilerOperatorIdentity.Onnx("Flatten"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.flatten"),
        CompilerTorchSharpForm.Call("Tensor.flatten"),
    ];

    public override IReadOnlyList<string> AttributeNames { get; } = ["axis"];

    protected override string PrintTorchSharp(Onnxify.Flatten node, CompilerSourceSpan? span)
    {
        var axis = node.Axis ?? 1;
        return $"torch.flatten({CompilerCSharpNaming.Identifier(node.Input.Name)}, start_dim: {axis})";
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Flatten flatten)
        {
            return $"Flatten requires the generated node type '{nameof(Onnxify.Flatten)}'.";
        }

        if (flatten.Axis is not { } axis
            || !knownTypes.TryGetValue(flatten.Input.Name, out var inputType)
            || inputType is not CompilerTensorType { Dimensions: not null } tensorType)
        {
            return null;
        }

        return axis >= -tensorType.Dimensions.Count && axis <= tensorType.Dimensions.Count
            ? null
            : "Flatten axis must be within the input tensor rank.";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        var receiver = context.Receiver;
        var argumentOffset = receiver is null ? 1 : 0;
        var inputExpression = receiver ?? invocation.Arguments.FirstOrDefault();
        if (inputExpression is null)
        {
            throw context.Unsupported(expression, "TorchSharp flatten requires a tensor input.");
        }

        var startIndex = argumentOffset;
        var endIndex = argumentOffset + 1;
        var startDimension = invocation.Arguments.Count > startIndex
            ? RequireInteger(invocation.Arguments[startIndex], context, "start_dim")
            : 0;
        var endDimension = invocation.Arguments.Count > endIndex
            ? RequireInteger(invocation.Arguments[endIndex], context, "end_dim")
            : -1;
        if (endDimension != -1)
        {
            throw context.Unsupported(
                expression,
                "TorchSharp flatten maps to ONNX Flatten only when end_dim is -1.");
        }

        if (invocation.Arguments.Count > endIndex + 1)
        {
            throw context.Unsupported(expression, "TorchSharp flatten accepts only input, start_dim, and end_dim.");
        }

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var inputType = context.Inputs
            .FirstOrDefault(value => string.Equals(value.Name, input.Name, StringComparison.Ordinal))
            ?.Type as CompilerTensorType;
        if (inputType?.Dimensions is { } dimensions
            && (startDimension < -dimensions.Count || startDimension > dimensions.Count))
        {
            throw context.Unsupported(expression, "TorchSharp flatten start_dim must be within the input tensor rank.");
        }

        var node = new Onnxify.Flatten(
            OnnxName.ToLowerInvariant(),
            new Onnxify.FlattenInputOutputOptions
            {
                Input = new OnnxEdge(input.Name),
                Axis = startDimension,
                Output = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }

    private static long RequireInteger(
        CompilerExpression expression,
        TorchSharpOperatorScanContext context,
        string parameterName)
    {
        if (expression is CompilerUnaryExpression { Operator: "-", Expression: CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signedMagnitude } }
            && signedMagnitude.Value != long.MinValue)
        {
            return -signedMagnitude.Value;
        }

        if (expression is CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signed })
        {
            return signed.Value;
        }

        if (expression is CompilerLiteralExpression { Literal: CompilerUnsignedIntegerLiteral unsigned }
            && unsigned.Value <= long.MaxValue)
        {
            return (long)unsigned.Value;
        }

        throw context.Unsupported(expression, $"TorchSharp flatten '{parameterName}' must be a compile-time integer.");
    }
}
