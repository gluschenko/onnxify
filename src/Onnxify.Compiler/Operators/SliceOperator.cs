using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::slice.Tensor")]
internal sealed class SliceOperator() : CompilerOperator<Onnxify.Slice>(CompilerOperatorIdentity.Onnx("Slice"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("Tensor.slice"),
    ];

    public override int MinimumInputCount => 3;
    public override int MaximumInputCount => 5;
    protected override bool AllowsEmptyOptionalInputs => true;

    protected override string PrintTorchSharp(Onnxify.Slice node, CompilerSourceSpan? span)
    {
        throw Unsupported(span, "Slice printing requires static starts, ends, axes, and optional steps initializers.");
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation, CompilerTorchSharpPrintContext context)
    {
        if (operation.Node is not Onnxify.Slice slice
            || slice.Axes is null
            || !TryGetSingleValue(context, slice.Starts.Name, out var start)
            || !TryGetSingleValue(context, slice.Ends.Name, out var end)
            || !TryGetSingleValue(context, slice.Axes.Name, out var axis))
        {
            throw Unsupported(operation.Span, "Slice requires static int64 starts, ends, and one explicit axis.");
        }

        var step = slice.Steps is null
            ? 1
            : TryGetSingleValue(context, slice.Steps.Name, out var staticStep)
                ? staticStep
                : throw Unsupported(operation.Span, "Slice steps must be a static int64 initializer.");
        if (step <= 0)
        {
            throw Unsupported(operation.Span, "TorchSharp Tensor.slice requires a positive step for an exact ONNX mapping.");
        }

        return $"{CompilerCSharpNaming.Identifier(slice.Data.Name)}.slice({axis}, {start}, {end}, {step})";
    }

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Slice slice)
        {
            return $"Slice requires the generated node type '{nameof(Onnxify.Slice)}'.";
        }

        foreach (var input in new[] { slice.Starts, slice.Ends }.Concat(slice.Axes is null ? [] : [slice.Axes]).Concat(slice.Steps is null ? [] : [slice.Steps]))
        {
            if (knownTypes.TryGetValue(input.Name, out var inputType)
                && inputType is CompilerTensorType { ElementType: not (CompilerElementType.Int64 or CompilerElementType.Int32) })
            {
                return "Slice bounds, axes, and steps must use int32 or int64 tensors.";
            }
        }

        return null;
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation || context.Receiver is null)
        {
            return false;
        }

        if (invocation.Arguments.Count is < 3 or > 4)
        {
            throw context.Unsupported(expression, "TorchSharp Tensor.slice supports dim, start, end, and an optional positive step.");
        }

        var axis = StaticInt64Vector.ReadScalar(invocation.Arguments[0], context, "TorchSharp slice axis");
        var start = StaticInt64Vector.ReadScalar(invocation.Arguments[1], context, "TorchSharp slice start");
        var end = StaticInt64Vector.ReadScalar(invocation.Arguments[2], context, "TorchSharp slice end");
        var step = invocation.Arguments.Count == 4
            ? StaticInt64Vector.ReadScalar(invocation.Arguments[3], context, "TorchSharp slice step")
            : 1;
        if (step <= 0)
        {
            throw context.Unsupported(expression, "TorchSharp Tensor.slice step must be positive for this mapping.");
        }

        var input = context.RequireTensorReference(context.Receiver, OnnxName);
        var inputType = context.Inputs.FirstOrDefault(value => string.Equals(value.Name, input.Name, StringComparison.Ordinal))?.Type as CompilerTensorType;
        if (inputType?.Dimensions is { } dimensions && (axis < -dimensions.Count || axis >= dimensions.Count))
        {
            throw context.Unsupported(invocation.Arguments[0], "TorchSharp slice axis is outside the input rank.");
        }

        var output = context.RequireSingleOutputEdge(this);
        var startsName = AddVector(context, output.Name, "starts", [start]);
        var endsName = AddVector(context, output.Name, "ends", [end]);
        var axesName = AddVector(context, output.Name, "axes", [axis]);
        var stepsName = AddVector(context, output.Name, "steps", [step]);
        var node = new Onnxify.Slice(
            "slice",
            new Onnxify.SliceInputOutputOptions
            {
                Data = new OnnxEdge(input.Name),
                Starts = new OnnxEdge(startsName),
                Ends = new OnnxEdge(endsName),
                Axes = new OnnxEdge(axesName),
                Steps = new OnnxEdge(stepsName),
                Output = output,
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }

    private static string AddVector(TorchSharpOperatorScanContext context, string outputName, string suffix, long[] values)
    {
        var name = CompilerCSharpNaming.Identifier($"__slice_{suffix}_{outputName}");
        StaticInt64Vector.AddInitializer(context, name, values);
        return name;
    }

    private static bool TryGetSingleValue(CompilerTorchSharpPrintContext context, string name, out long value)
    {
        value = 0;
        return context.TryGetTensorInitializer(name, out var initializer)
            && initializer!.ElementType is CompilerElementType.Int32 or CompilerElementType.Int64
            && initializer.Values.Count == 1
            && initializer.Values[0] switch
            {
                CompilerSignedIntegerLiteral signed => Assign(signed.Value, out value),
                CompilerUnsignedIntegerLiteral unsigned when unsigned.Value <= long.MaxValue => Assign((long)unsigned.Value, out value),
                _ => false,
            };
    }

    private static bool Assign(long candidate, out long value)
    {
        value = candidate;
        return true;
    }
}
