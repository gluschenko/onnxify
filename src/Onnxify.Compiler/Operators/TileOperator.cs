using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::tile")]
[CompilerTorchOperator("aten::repeat")]
internal sealed class TileOperator() : CompilerOperator<Onnxify.Tile>(CompilerOperatorIdentity.Onnx("Tile"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.tile"),
        CompilerTorchSharpForm.Call("Tensor.repeat"),
    ];

    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;

    protected override string PrintTorchSharp(Onnxify.Tile node, CompilerSourceSpan? span)
    {
        throw Unsupported(span, "Tile requires a static repeats initializer.");
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation, CompilerTorchSharpPrintContext context)
    {
        if (operation.Node is not Onnxify.Tile tile
            || !context.TryGetTensorInitializer(tile.Repeats.Name, out var repeats)
            || repeats!.ElementType != CompilerElementType.Int64
            || repeats.Values.Any(static value => value is not CompilerSignedIntegerLiteral))
        {
            throw Unsupported(operation.Span, "Tile requires a static int64 repeats initializer.");
        }

        var dimensions = repeats.Values.Cast<CompilerSignedIntegerLiteral>().Select(static value => value.Value).ToArray();
        if (dimensions.Any(static value => value < 0))
        {
            throw Unsupported(operation.Span, "Tile repeats must be non-negative.");
        }

        return $"{CompilerCSharpNaming.Identifier(tile.Input.Name)}.repeat(new long[] {{ {StaticInt64Vector.Format(dimensions)} }})";
    }

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Tile tile)
        {
            return $"Tile requires the generated node type '{nameof(Onnxify.Tile)}'.";
        }

        return knownTypes.TryGetValue(tile.Repeats.Name, out var repeatsType)
            && repeatsType is CompilerTensorType { ElementType: not CompilerElementType.Int64 }
                ? "Tile repeats input must have int64 element type."
                : null;
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        var inputExpression = context.Receiver ?? invocation.Arguments.FirstOrDefault();
        var repeatsExpression = context.Receiver is null
            ? invocation.Arguments.Skip(1).FirstOrDefault()
            : invocation.Arguments.FirstOrDefault();
        if (inputExpression is null || repeatsExpression is null)
        {
            throw context.Unsupported(expression, "TorchSharp repeat/tile requires a tensor input and static repeat counts.");
        }

        if ((context.Receiver is null && invocation.Arguments.Count != 2)
            || (context.Receiver is not null && invocation.Arguments.Count != 1))
        {
            throw context.Unsupported(expression, "TorchSharp repeat/tile accepts exactly one repeat-count array.");
        }

        var repeats = StaticInt64Vector.ReadVector(repeatsExpression, context, "TorchSharp tile repeats");
        if (repeats.Any(static value => value < 0))
        {
            throw context.Unsupported(repeatsExpression, "TorchSharp tile repeats must be non-negative compile-time integers.");
        }

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var inputType = context.Inputs.FirstOrDefault(value => string.Equals(value.Name, input.Name, StringComparison.Ordinal))?.Type as CompilerTensorType;
        if (inputType?.Dimensions is { } inputDimensions && repeats.Length != inputDimensions.Count)
        {
            throw context.Unsupported(repeatsExpression, "TorchSharp repeat/tile requires one repeat count per input dimension.");
        }

        var output = context.RequireSingleOutputEdge(this);
        var repeatsName = CompilerCSharpNaming.Identifier($"__tile_repeats_{output.Name}");
        StaticInt64Vector.AddInitializer(context, repeatsName, repeats);
        var node = new Onnxify.Tile(
            "tile",
            new Onnxify.TileInputOutputOptions
            {
                Input = new OnnxEdge(input.Name),
                Repeats = new OnnxEdge(repeatsName),
                Output = output,
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }
}
