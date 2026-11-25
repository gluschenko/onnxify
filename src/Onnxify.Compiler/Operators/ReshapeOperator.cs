using System.Globalization;
using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::reshape")]
[CompilerTorchOperator("aten::view")]
[CompilerTorchOperator("aten::unflatten.int")]
internal sealed class ReshapeOperator() : CompilerOperator<Onnxify.Reshape>(CompilerOperatorIdentity.Onnx("Reshape"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.reshape"),
        CompilerTorchSharpForm.Call("Tensor.reshape"),
        CompilerTorchSharpForm.Call("Tensor.view"),
        CompilerTorchSharpForm.Call("Tensor.unflatten"),
        CompilerTorchSharpForm.Call("torch.unflatten"),
    ];

    public override IReadOnlyList<string> AttributeNames { get; } = ["allowzero"];

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        if (operation.Node is Onnxify.Reshape { Allowzero: not null })
        {
            graph.AddNode(operation.Node);
            return;
        }

        // Reshape-14 introduced allowzero. Preserve legacy Reshape nodes as generic ONNX
        // nodes so opset-12 models are not accidentally emitted with the newer schema.
        var node = operation.Node;
        graph.AddNode(new OnnxNode(node.Name, node.OpType, node.Domain, node.DocString, node.Inputs, node.Outputs, node.Attributes));
    }

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

        if (string.Equals(context.CallName, "Tensor.unflatten", StringComparison.Ordinal)
            || string.Equals(context.CallName, "torch.unflatten", StringComparison.Ordinal))
        {
            return TryScanUnflatten(context, expression, invocation);
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

    private bool TryScanUnflatten(
        TorchSharpOperatorScanContext context,
        CompilerExpression expression,
        CompilerInvocationExpression invocation)
    {
        var inputExpression = context.Receiver ?? invocation.Arguments.FirstOrDefault();
        var arguments = context.Receiver is null
            ? invocation.Arguments.Skip(1).ToArray()
            : invocation.Arguments;
        if (inputExpression is null || arguments.Count < 2)
        {
            throw context.Unsupported(expression, "TorchSharp unflatten requires a tensor, one static axis, and compile-time sizes.");
        }

        var axis = StaticInt64Vector.ReadScalar(arguments[0], context, "TorchSharp unflatten axis");
        var sizes = arguments.Count == 2 && arguments[1] is CompilerArrayExpression sizesArray
            ? sizesArray.Items.Select(item => StaticInt64Vector.ReadScalar(item, context, "TorchSharp unflatten sizes")).ToArray()
            : arguments.Skip(1).Select(argument => StaticInt64Vector.ReadScalar(argument, context, "TorchSharp unflatten sizes")).ToArray();
        if (sizes.Length == 0 || sizes.Any(static dimension => dimension < 0))
        {
            throw context.Unsupported(expression, "TorchSharp unflatten sizes must be a non-empty array of non-negative integers.");
        }

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var inputType = context.Inputs.FirstOrDefault(value => string.Equals(value.Name, input.Name, StringComparison.Ordinal))?.Type as CompilerTensorType;
        if (inputType?.Dimensions is not { } dimensions
            || dimensions.Any(static dimension => dimension is not CompilerFixedDimension))
        {
            throw context.Unsupported(expression, "TorchSharp unflatten requires statically known input dimensions until runtime shape lowering is available.");
        }

        var normalizedAxis = axis < 0 ? axis + dimensions.Count : axis;
        if (normalizedAxis < 0 || normalizedAxis >= dimensions.Count)
        {
            throw context.Unsupported(arguments[0], "TorchSharp unflatten axis is outside the input rank.");
        }

        var inputDimensions = dimensions
            .Cast<CompilerFixedDimension>()
            .Select(static dimension => dimension.Value)
            .ToArray();
        long requestedDimension;
        try
        {
            requestedDimension = sizes.Aggregate(1L, static (product, size) => checked(product * size));
        }
        catch (OverflowException)
        {
            throw context.Unsupported(expression, "TorchSharp unflatten sizes overflow the supported ONNX dimension range.");
        }

        if (inputDimensions[(int)normalizedAxis] != requestedDimension)
        {
            throw context.Unsupported(expression, "TorchSharp unflatten sizes must multiply to the selected input dimension.");
        }

        var shape = inputDimensions
            .Take((int)normalizedAxis)
            .Concat(sizes)
            .Concat(inputDimensions.Skip((int)normalizedAxis + 1))
            .ToArray();
        var output = context.RequireSingleOutputEdge(this);
        var shapeName = CompilerCSharpNaming.Identifier($"__unflatten_shape_{output.Name}");
        StaticInt64Vector.AddInitializer(context, shapeName, shape);
        var node = new Onnxify.Reshape(
            "unflatten",
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
            return $"ReshapeWithRuntimeShape({inputName}, {CompilerCSharpNaming.Identifier(node.Shape.Name)}, {node.Allowzero ?? 0}L)";
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
