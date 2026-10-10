using Onnxify;

namespace Onnxify.Compiler.Operators;

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
        if (node.Axis is not (null or 1))
        {
            throw Unsupported(span, "Compiler Flatten currently supports ONNX axis 1 only.");
        }

        return $"torch.flatten({CompilerCSharpNaming.Identifier(node.Input.Name)}, start_dim: 1)";
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Flatten flatten)
        {
            return $"Flatten requires the generated node type '{nameof(Onnxify.Flatten)}'.";
        }

        return flatten.Axis is null or 1
            ? null
            : "Compiler Flatten currently supports ONNX axis 1 only.";
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
        if (startDimension != 1 || endDimension != -1)
        {
            throw context.Unsupported(
                expression,
                "TorchSharp flatten maps to ONNX Flatten only when start_dim is 1 and end_dim is -1.");
        }

        if (invocation.Arguments.Count > endIndex + 1)
        {
            throw context.Unsupported(expression, "TorchSharp flatten accepts only input, start_dim, and end_dim.");
        }

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var node = new Onnxify.Flatten(
            OnnxName.ToLowerInvariant(),
            new Onnxify.FlattenInputOutputOptions
            {
                Input = new OnnxEdge(input.Name),
                Axis = 1,
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
