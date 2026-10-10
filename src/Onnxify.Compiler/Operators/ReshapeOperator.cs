using System.Globalization;
using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class ReshapeOperator() : CompilerOperator<Onnxify.Reshape>(CompilerOperatorIdentity.Onnx("Reshape"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.reshape"),
        CompilerTorchSharpForm.Call("Tensor.reshape"),
        CompilerTorchSharpForm.Call("Tensor.view"),
    ];

    public override IReadOnlyList<string> AttributeNames { get; } = ["allowzero"];

    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;

    protected override string PrintTorchSharp(Onnxify.Reshape node, CompilerSourceSpan? span)
    {
        return PrintTorchSharp(node, span, context: null);
    }

    public override string PrintTorchSharp(
        CompilerOnnxStep operation,
        CompilerTorchSharpPrintContext context)
    {
        if (operation.Node is not Onnxify.Reshape reshape)
        {
            throw Unsupported(operation.Span, "Reshape requires the generated ONNX Reshape node type.");
        }

        return PrintTorchSharp(reshape, operation.Span, context);
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Reshape reshape)
        {
            return $"Reshape requires the generated node type '{nameof(Onnxify.Reshape)}'.";
        }

        if (reshape.Allowzero is not (null or 0 or 1))
        {
            return "Reshape allowzero must be 0 or 1.";
        }

        return knownTypes.TryGetValue(reshape.Shape.Name, out var shapeType)
            && shapeType is CompilerTensorType { ElementType: not CompilerElementType.Int64 }
                ? "Reshape shape input must have int64 element type."
                : null;
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        var inputExpression = context.Receiver ?? invocation.Arguments.FirstOrDefault();
        var shapeExpression = context.Receiver is null
            ? invocation.Arguments.Skip(1).FirstOrDefault()
            : invocation.Arguments.FirstOrDefault();
        if (inputExpression is null || shapeExpression is null)
        {
            throw context.Unsupported(expression, "TorchSharp reshape requires a tensor input and a static shape array.");
        }

        if ((context.Receiver is null && invocation.Arguments.Count != 2)
            || (context.Receiver is not null && invocation.Arguments.Count != 1))
        {
            throw context.Unsupported(expression, "TorchSharp reshape accepts only the input tensor and one shape array.");
        }

        if (shapeExpression is not CompilerArrayExpression shapeArray)
        {
            throw context.Unsupported(expression, "TorchSharp reshape requires a compile-time integer shape array.");
        }

        var shape = shapeArray.Items.Select(item => RequireShapeDimension(item, context)).ToArray();
        if (shape.Count(static dimension => dimension == -1) > 1
            || shape.Any(static dimension => dimension < -1))
        {
            throw context.Unsupported(expression, "TorchSharp reshape supports at most one inferred dimension (-1), and no dimensions below -1.");
        }

        if (shape.Contains(0) && shape.Contains(-1))
        {
            throw context.Unsupported(expression, "TorchSharp reshape cannot combine a zero dimension with an inferred dimension.");
        }

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var output = context.RequireSingleOutputEdge(this);
        var shapeName = CompilerCSharpNaming.Identifier($"__reshape_shape_{output.Name}");
        var shapeValues = shape.Select(static dimension => (CompilerScalarLiteral)new CompilerSignedIntegerLiteral(
            CompilerElementType.Int64,
            dimension));
        context.Builder.AddStateMember(
            new CompilerStateMember(
                shapeName,
                CompilerStateMemberKind.Initializer,
                new CompilerTensorType(
                    CompilerElementType.Int64,
                    [new CompilerFixedDimension(shape.Length)]),
                new CompilerTensorLiteral(
                    CompilerElementType.Int64,
                    [new CompilerFixedDimension(shape.Length)],
                    shapeValues)));

        var node = new Onnxify.Reshape(
            OnnxName.ToLowerInvariant(),
            new Onnxify.ReshapeInputOutputOptions
            {
                Data = new OnnxEdge(input.Name),
                Shape = new OnnxEdge(shapeName),
                Allowzero = 1,
                Reshaped = output,
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }

    private static string PrintTorchSharp(
        Onnxify.Reshape node,
        CompilerSourceSpan? span,
        CompilerTorchSharpPrintContext? context)
    {
        var inputName = CompilerCSharpNaming.Identifier(node.Data.Name);
        var shapeValues = context is not null
            && context.TryGetTensorInitializer(node.Shape.Name, out var initializer)
                ? GetShapeValues(initializer!, node, span)
                : null;
        if (shapeValues is null)
        {
            if (node.Allowzero is not (1))
            {
                throw Unsupported(
                    span,
                    "Reshape with dynamic shape input requires allowzero=1 so its shape semantics remain exact.");
            }

            return $"{inputName}.reshape({CompilerCSharpNaming.Identifier(node.Shape.Name)}.data<long>())";
        }

        var dimensions = shapeValues.Select((dimension, index) =>
            dimension == 0 && node.Allowzero is not 1
                ? $"{inputName}.shape[{index.ToString(CultureInfo.InvariantCulture)}]"
                : dimension.ToString(CultureInfo.InvariantCulture) + "L");
        return $"{inputName}.reshape(new long[] {{ {string.Join(", ", dimensions)} }})";
    }

    private static long[]? GetShapeValues(
        CompilerTensorLiteral shape,
        Onnxify.Reshape node,
        CompilerSourceSpan? span)
    {
        if (shape.ElementType != CompilerElementType.Int64
            || shape.Values.Any(static value => value is not CompilerSignedIntegerLiteral))
        {
            throw Unsupported(span, "Reshape shape initializer must be a statically known int64 tensor.");
        }

        var values = shape.Values.Cast<CompilerSignedIntegerLiteral>().Select(static value => value.Value).ToArray();
        if (values.Count(static dimension => dimension == -1) > 1 || values.Any(static dimension => dimension < -1))
        {
            throw Unsupported(span, "Reshape shape must contain at most one inferred dimension (-1) and no dimensions below -1.");
        }

        if (node.Allowzero == 1 && values.Contains(0) && values.Contains(-1))
        {
            throw Unsupported(span, "ONNX Reshape allowzero=1 cannot combine a zero dimension with an inferred dimension.");
        }

        return values;
    }

    private static long RequireShapeDimension(
        CompilerExpression expression,
        TorchSharpOperatorScanContext context)
    {
        if (expression is CompilerUnaryExpression { Operator: "-", Expression: CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signedMagnitude } })
        {
            return -signedMagnitude.Value;
        }

        return expression switch
        {
            CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signed } => signed.Value,
            CompilerLiteralExpression { Literal: CompilerUnsignedIntegerLiteral unsigned } when unsigned.Value <= long.MaxValue => (long)unsigned.Value,
            _ => throw context.Unsupported(expression, "TorchSharp reshape dimensions must be compile-time integers."),
        };
    }
}
