using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::unsqueeze")]
internal sealed class UnsqueezeOperator() : CompilerOperator<Onnxify.Unsqueeze>(CompilerOperatorIdentity.Onnx("Unsqueeze"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.unsqueeze"),
        CompilerTorchSharpForm.Call("Tensor.unsqueeze"),
    ];

    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;

    protected override string PrintTorchSharp(Onnxify.Unsqueeze node, CompilerSourceSpan? span)
    {
        throw Unsupported(span, "Unsqueeze printing requires its static axes initializer.");
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation, CompilerTorchSharpPrintContext context)
    {
        if (operation.Node is not Onnxify.Unsqueeze unsqueeze)
        {
            throw Unsupported(operation.Span, "Unsqueeze requires the generated ONNX Unsqueeze node type.");
        }

        if (!context.TryGetTensorInitializer(unsqueeze.Axes.Name, out var axes)
            || axes!.ElementType != CompilerElementType.Int64
            || axes.Values.Count != 1
            || axes.Values[0] is not CompilerSignedIntegerLiteral axis)
        {
            throw Unsupported(operation.Span, "Unsqueeze requires a static int64 axes initializer with exactly one axis.");
        }

        return $"{CompilerCSharpNaming.Identifier(unsqueeze.Data.Name)}.unsqueeze({axis.Value})";
    }

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Unsqueeze unsqueeze)
        {
            return $"Unsqueeze requires the generated node type '{nameof(Onnxify.Unsqueeze)}'.";
        }

        return knownTypes.TryGetValue(unsqueeze.Axes.Name, out var axesType)
            && axesType is CompilerTensorType { ElementType: not CompilerElementType.Int64 }
                ? "Unsqueeze axes input must have int64 element type."
                : null;
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        var inputExpression = context.Receiver ?? invocation.Arguments.FirstOrDefault();
        var axisExpression = context.Receiver is null
            ? invocation.Arguments.Skip(1).FirstOrDefault()
            : invocation.Arguments.FirstOrDefault();
        if (inputExpression is null || axisExpression is null)
        {
            throw context.Unsupported(expression, "TorchSharp unsqueeze requires a tensor input and one static axis.");
        }

        if ((context.Receiver is null && invocation.Arguments.Count != 2)
            || (context.Receiver is not null && invocation.Arguments.Count != 1))
        {
            throw context.Unsupported(expression, "TorchSharp unsqueeze accepts exactly one axis.");
        }

        var axis = axisExpression switch
        {
            CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signed } => signed.Value,
            CompilerLiteralExpression { Literal: CompilerUnsignedIntegerLiteral unsigned } when unsigned.Value <= long.MaxValue => (long)unsigned.Value,
            CompilerUnaryExpression { Operator: "-", Expression: CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signed } }
                when signed.Value != long.MinValue => -signed.Value,
            _ => throw context.Unsupported(axisExpression, "TorchSharp unsqueeze axis must be a compile-time integer."),
        };

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var inputType = context.Inputs.FirstOrDefault(value => string.Equals(value.Name, input.Name, StringComparison.Ordinal))?.Type as CompilerTensorType;
        if (inputType?.Dimensions is { } dimensions && (axis < -dimensions.Count - 1 || axis > dimensions.Count))
        {
            throw context.Unsupported(axisExpression, "TorchSharp unsqueeze axis is outside the input rank.");
        }

        var output = context.RequireSingleOutputEdge(this);
        var axesName = CompilerCSharpNaming.Identifier($"__unsqueeze_axes_{output.Name}");
        context.Builder.AddStateMember(
            new CompilerStateMember(
                axesName,
                CompilerStateMemberKind.Initializer,
                new CompilerTensorType(CompilerElementType.Int64, [new CompilerFixedDimension(1)]),
                new CompilerTensorLiteral(
                    CompilerElementType.Int64,
                    [new CompilerFixedDimension(1)],
                    [(CompilerScalarLiteral)new CompilerSignedIntegerLiteral(CompilerElementType.Int64, axis)])));

        var node = new Onnxify.Unsqueeze(
            "unsqueeze",
            new Onnxify.UnsqueezeInputOutputOptions
            {
                Data = new OnnxEdge(input.Name),
                Axes = new OnnxEdge(axesName),
                Expanded = output,
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }
}
