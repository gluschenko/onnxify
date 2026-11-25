using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::squeeze.dim")]
internal sealed class SqueezeOperator() : CompilerOperator<Onnxify.Squeeze>(CompilerOperatorIdentity.Onnx("Squeeze"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.squeeze"),
        CompilerTorchSharpForm.Call("Tensor.squeeze"),
    ];

    public override int MinimumInputCount => 1;
    public override int MaximumInputCount => 2;

    protected override string PrintTorchSharp(Onnxify.Squeeze node, CompilerSourceSpan? span)
    {
        throw Unsupported(span, "Squeeze requires its static axes initializer.");
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation, CompilerTorchSharpPrintContext context)
    {
        if (operation.Node is not Onnxify.Squeeze squeeze
            || squeeze.Axes is null
            || !context.TryGetTensorInitializer(squeeze.Axes.Name, out var axes)
            || axes!.ElementType != CompilerElementType.Int64
            || axes.Values.Count != 1
            || axes.Values[0] is not CompilerSignedIntegerLiteral axis)
        {
            throw Unsupported(operation.Span, "Squeeze requires a static int64 axes initializer with exactly one axis.");
        }

        return $"{CompilerCSharpNaming.Identifier(squeeze.Data.Name)}.squeeze({axis.Value})";
    }

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Squeeze squeeze)
        {
            return $"Squeeze requires the generated node type '{nameof(Onnxify.Squeeze)}'.";
        }

        if (squeeze.Axes is null)
        {
            return "Compiler Squeeze requires an explicit axes input.";
        }

        if (knownTypes.TryGetValue(squeeze.Axes.Name, out var axesType)
            && axesType is CompilerTensorType { ElementType: not CompilerElementType.Int64 })
        {
            return "Squeeze axes input must have int64 element type.";
        }

        return null;
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
            throw context.Unsupported(expression, "TorchSharp squeeze requires a tensor input and one static axis.");
        }

        if ((context.Receiver is null && invocation.Arguments.Count != 2)
            || (context.Receiver is not null && invocation.Arguments.Count != 1))
        {
            throw context.Unsupported(expression, "TorchSharp squeeze accepts exactly one axis.");
        }

        var axis = axisExpression switch
        {
            CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signed } => signed.Value,
            CompilerLiteralExpression { Literal: CompilerUnsignedIntegerLiteral unsigned } when unsigned.Value <= long.MaxValue => (long)unsigned.Value,
            CompilerUnaryExpression { Operator: "-", Expression: CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signed } }
                when signed.Value != long.MinValue => -signed.Value,
            _ => throw context.Unsupported(axisExpression, "TorchSharp squeeze axis must be a compile-time integer."),
        };

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var inputType = context.Inputs.FirstOrDefault(value => string.Equals(value.Name, input.Name, StringComparison.Ordinal))?.Type as CompilerTensorType;
        if (inputType?.Dimensions is not { } dimensions || dimensions.Count == 0)
        {
            throw context.Unsupported(expression, "TorchSharp squeeze requires a known input rank to resolve its axis.");
        }

        var normalizedAxis = axis < 0 ? axis + dimensions.Count : axis;
        if (normalizedAxis < 0 || normalizedAxis >= dimensions.Count)
        {
            throw context.Unsupported(axisExpression, "TorchSharp squeeze axis is outside the input rank.");
        }

        if (dimensions[(int)normalizedAxis] is not CompilerFixedDimension { Value: 1 })
        {
            throw context.Unsupported(axisExpression, "TorchSharp squeeze can only map to ONNX when the selected dimension is statically known to be one.");
        }

        var output = context.RequireSingleOutputEdge(this);
        var axesName = CompilerCSharpNaming.Identifier($"__squeeze_axes_{output.Name}");
        context.Builder.AddStateMember(
            new CompilerStateMember(
                axesName,
                CompilerStateMemberKind.Initializer,
                new CompilerTensorType(CompilerElementType.Int64, [new CompilerFixedDimension(1)]),
                new CompilerTensorLiteral(
                    CompilerElementType.Int64,
                    [new CompilerFixedDimension(1)],
                    [(CompilerScalarLiteral)new CompilerSignedIntegerLiteral(CompilerElementType.Int64, normalizedAxis)])));

        var node = new Onnxify.Squeeze(
            "squeeze",
            new Onnxify.SqueezeInputOutputOptions
            {
                Data = new OnnxEdge(input.Name),
                Axes = new OnnxEdge(axesName),
                Squeezed = output,
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }
}
