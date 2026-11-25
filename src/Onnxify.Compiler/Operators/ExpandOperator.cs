using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::expand")]
[CompilerTorchOperator("aten::broadcast_to")]
internal sealed class ExpandOperator() : CompilerOperator<Onnxify.Expand>(CompilerOperatorIdentity.Onnx("Expand"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.broadcast_to"),
        CompilerTorchSharpForm.Call("Tensor.expand"),
    ];

    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;

    protected override string PrintTorchSharp(Onnxify.Expand node, CompilerSourceSpan? span)
    {
        throw Unsupported(span, "Expand requires a static shape initializer.");
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation, CompilerTorchSharpPrintContext context)
    {
        if (operation.Node is not Onnxify.Expand expand
            || !context.TryGetTensorInitializer(expand.Shape.Name, out var shape)
            || shape!.ElementType != CompilerElementType.Int64
            || shape.Values.Any(static value => value is not CompilerSignedIntegerLiteral))
        {
            throw Unsupported(operation.Span, "Expand requires a static int64 shape initializer.");
        }

        var dimensions = shape.Values.Cast<CompilerSignedIntegerLiteral>().Select(static value => value.Value).ToArray();
        if (dimensions.Any(static value => value < 0))
        {
            throw Unsupported(operation.Span, "ONNX Expand cannot represent TorchSharp expand dimensions below zero.");
        }

        return $"ExpandWithRuntimeShape({CompilerCSharpNaming.Identifier(expand.Input.Name)}, new long[] {{ {StaticInt64Vector.Format(dimensions)} }})";
    }

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Expand expand)
        {
            return $"Expand requires the generated node type '{nameof(Onnxify.Expand)}'.";
        }

        return knownTypes.TryGetValue(expand.Shape.Name, out var shapeType)
            && shapeType is CompilerTensorType { ElementType: not CompilerElementType.Int64 }
                ? "Expand shape input must have int64 element type."
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
            throw context.Unsupported(expression, "TorchSharp expand requires a tensor input and a static target shape.");
        }

        if ((context.Receiver is null && invocation.Arguments.Count != 2)
            || (context.Receiver is not null && invocation.Arguments.Count != 1))
        {
            throw context.Unsupported(expression, "TorchSharp expand accepts exactly one target shape.");
        }

        var dimensions = StaticInt64Vector.ReadVector(shapeExpression, context, "TorchSharp expand shape");
        if (dimensions.Any(static value => value < 0))
        {
            throw context.Unsupported(shapeExpression, "TorchSharp expand dimensions must be non-negative compile-time integers.");
        }

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var inputType = context.Inputs.FirstOrDefault(value => string.Equals(value.Name, input.Name, StringComparison.Ordinal))?.Type as CompilerTensorType;
        if (inputType?.Dimensions is { } inputDimensions)
        {
            if (dimensions.Length < inputDimensions.Count)
            {
                throw context.Unsupported(shapeExpression, "TorchSharp expand target rank cannot be lower than the input rank.");
            }

            var rankOffset = dimensions.Length - inputDimensions.Count;
            for (var index = 0; index < inputDimensions.Count; index++)
            {
                if (inputDimensions[index] is CompilerFixedDimension fixedDimension
                    && fixedDimension.Value != 1
                    && fixedDimension.Value != dimensions[index + rankOffset])
                {
                    throw context.Unsupported(shapeExpression, "TorchSharp expand target shape is incompatible with a known input dimension.");
                }
            }
        }

        var output = context.RequireSingleOutputEdge(this);
        var shapeName = CompilerCSharpNaming.Identifier($"__expand_shape_{output.Name}");
        StaticInt64Vector.AddInitializer(context, shapeName, dimensions);
        var node = new Onnxify.Expand(
            "expand",
            new Onnxify.ExpandInputOutputOptions
            {
                Input = new OnnxEdge(input.Name),
                Shape = new OnnxEdge(shapeName),
                Output = output,
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }
}
