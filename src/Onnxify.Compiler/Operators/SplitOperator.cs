using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::split")]
[CompilerTorchOperator("aten::split_with_sizes")]
[CompilerTorchOperator("aten::chunk")]
[CompilerTorchOperator("aten::unbind.int")]
internal sealed class SplitOperator() : CompilerOperator<Onnxify.Split>(CompilerOperatorIdentity.Onnx("Split"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.split"),
        CompilerTorchSharpForm.Call("Tensor.split"),
        CompilerTorchSharpForm.Call("torch.chunk"),
        CompilerTorchSharpForm.Call("Tensor.chunk"),
        CompilerTorchSharpForm.Call("torch.unbind"),
        CompilerTorchSharpForm.Call("Tensor.unbind"),
    ];

    public override int MinimumInputCount => 1;
    public override int MaximumInputCount => 2;
    public override int MinimumOutputCount => 2;
    public override int MaximumOutputCount => int.MaxValue;
    public override IReadOnlyList<string> AttributeNames { get; } = ["axis", "num_outputs"];

    protected override bool AllowsEmptyOptionalInputs => true;

    protected override string PrintTorchSharp(Onnxify.Split node, CompilerSourceSpan? span)
    {
        throw Unsupported(span, "Split printing requires statically known split sizes or a fixed output count.");
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation, CompilerTorchSharpPrintContext context)
    {
        if (operation.Node is not Onnxify.Split split)
        {
            throw Unsupported(operation.Span, "Split requires the generated ONNX Split node type.");
        }

        var inputName = CompilerCSharpNaming.Identifier(split.Input.Name);
        if (split.InputSplit is not null)
        {
            if (!context.TryGetTensorInitializer(split.InputSplit.Name, out var initializer)
                || initializer!.ElementType != CompilerElementType.Int64
                || initializer.Values.Any(static value => value is not CompilerSignedIntegerLiteral))
            {
                throw Unsupported(operation.Span, "Split sizes must be a static int64 initializer.");
            }

            var sizes = initializer.Values.Cast<CompilerSignedIntegerLiteral>()
                .Select(static value => value.Value)
                .ToArray();
            if (sizes.Length != split.Out.Count || sizes.Any(static size => size < 0))
            {
                throw Unsupported(operation.Span, "Split size count must match the node output count and sizes must be non-negative.");
            }

            return $"torch.split({inputName}, new long[] {{ {StaticInt64Vector.Format(sizes)} }}, dim: {split.Axis ?? 0})";
        }

        if (split.NumOutputs != split.Out.Count)
        {
            throw Unsupported(operation.Span, "Split num_outputs must be statically known and match the node output count.");
        }

        return $"torch.chunk({inputName}, {split.NumOutputs}, dim: {split.Axis ?? 0})";
    }

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Split split)
        {
            return $"Split requires the generated node type '{nameof(Onnxify.Split)}'.";
        }

        if (split.Out.Count < 2 || split.InputSplit is not null && split.NumOutputs is not null)
        {
            return "Split requires at least two outputs and cannot specify both split sizes and num_outputs.";
        }

        if (split.InputSplit is not null
            && knownTypes.TryGetValue(split.InputSplit.Name, out var splitType)
            && splitType is CompilerTensorType { ElementType: not CompilerElementType.Int64 })
        {
            return "Split sizes must use int64 element type.";
        }

        return null;
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        if (context.CallName is "torch.chunk" or "Tensor.chunk")
        {
            return TryScanChunk(context, expression, invocation);
        }

        if (context.CallName is "torch.unbind" or "Tensor.unbind")
        {
            return TryScanUnbind(context, expression, invocation);
        }

        var isReceiverCall = context.Receiver is not null;
        var inputExpression = context.Receiver ?? invocation.Arguments.FirstOrDefault();
        var parameters = isReceiverCall ? invocation.Arguments : invocation.Arguments.Skip(1).ToArray();
        if (inputExpression is null || parameters.Count is < 1 or > 2)
        {
            throw context.Unsupported(expression, "TorchSharp split requires a tensor, static split sizes, and an optional axis.");
        }

        var sizes = StaticInt64Vector.ReadVector(parameters[0], context, "TorchSharp split sizes");
        if (sizes.Length != context.Outputs.Count || sizes.Length < 2 || sizes.Any(static size => size < 0))
        {
            throw context.Unsupported(expression, "TorchSharp split sizes must be non-negative and match the statically declared output count.");
        }

        var axis = parameters.Count == 2
            ? StaticInt64Vector.ReadScalar(parameters[1], context, "TorchSharp split axis")
            : 0;
        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var splitName = CompilerCSharpNaming.Identifier($"__split_sizes_{context.Outputs[0].Name}");
        StaticInt64Vector.AddInitializer(context, splitName, sizes);
        var node = new Onnxify.Split(
            "split",
            new Onnxify.SplitInputOutputOptions
            {
                Input = new OnnxEdge(input.Name),
                InputSplit = new OnnxEdge(splitName),
                Axis = axis,
                Out = context.Outputs.Select(output => (IOnnxGraphEdge)new OnnxEdge(output.Name)).ToArray(),
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }

    private bool TryScanChunk(
        TorchSharpOperatorScanContext context,
        CompilerExpression expression,
        CompilerInvocationExpression invocation)
    {
        var inputExpression = context.Receiver ?? invocation.Arguments.FirstOrDefault();
        var arguments = context.Receiver is null
            ? invocation.Arguments.Skip(1).ToArray()
            : invocation.Arguments;
        if (inputExpression is null || arguments.Count is < 1 or > 2)
        {
            throw context.Unsupported(expression, "TorchSharp chunk requires a tensor, a static chunk count, and an optional static axis.");
        }

        var chunkCount = StaticInt64Vector.ReadScalar(arguments[0], context, "TorchSharp chunk count");
        var axis = arguments.Count == 2
            ? StaticInt64Vector.ReadScalar(arguments[1], context, "TorchSharp chunk axis")
            : 0;
        if (chunkCount <= 0)
        {
            throw context.Unsupported(arguments[0], "TorchSharp chunk count must be positive.");
        }

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var dimension = RequireStaticAxisDimension(context, input.Name, axis, expression);
        var chunkSize = dimension / chunkCount + (dimension % chunkCount == 0 ? 0 : 1);
        var sizes = new List<long>();
        for (var remaining = dimension; remaining > 0; remaining -= Math.Min(chunkSize, remaining))
        {
            sizes.Add(Math.Min(chunkSize, remaining));
        }

        if (sizes.Count != context.Outputs.Count)
        {
            throw context.Unsupported(expression, "TorchSharp chunk output count must match the statically known number of chunks.");
        }

        AddSplit(context, input.Name, axis, sizes, expression.Span);
        return true;
    }

    private bool TryScanUnbind(
        TorchSharpOperatorScanContext context,
        CompilerExpression expression,
        CompilerInvocationExpression invocation)
    {
        var inputExpression = context.Receiver ?? invocation.Arguments.FirstOrDefault();
        var arguments = context.Receiver is null
            ? invocation.Arguments.Skip(1).ToArray()
            : invocation.Arguments;
        if (inputExpression is null || arguments.Count > 1)
        {
            throw context.Unsupported(expression, "TorchSharp unbind requires a tensor and an optional static axis.");
        }

        var axis = arguments.Count == 1
            ? StaticInt64Vector.ReadScalar(arguments[0], context, "TorchSharp unbind axis")
            : 0;
        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var dimension = RequireStaticAxisDimension(context, input.Name, axis, expression);
        if (dimension != context.Outputs.Count || dimension < 2)
        {
            throw context.Unsupported(expression, "TorchSharp unbind output count must equal the selected statically known dimension and be at least two.");
        }

        AddSplit(context, input.Name, axis, Enumerable.Repeat(1L, (int)dimension).ToArray(), expression.Span);
        return true;
    }

    private void AddSplit(
        TorchSharpOperatorScanContext context,
        string inputName,
        long axis,
        IReadOnlyList<long> sizes,
        CompilerSourceSpan? span)
    {
        var splitName = CompilerCSharpNaming.Identifier($"__split_sizes_{context.Outputs[0].Name}");
        StaticInt64Vector.AddInitializer(context, splitName, sizes);
        var node = new Onnxify.Split(
            "split",
            new Onnxify.SplitInputOutputOptions
            {
                Input = new OnnxEdge(inputName),
                InputSplit = new OnnxEdge(splitName),
                Axis = axis,
                Out = context.Outputs.Select(output => (IOnnxGraphEdge)new OnnxEdge(output.Name)).ToArray(),
            });
        context.AddOperation(this, node, span);
    }

    private static long RequireStaticAxisDimension(
        TorchSharpOperatorScanContext context,
        string inputName,
        long axis,
        CompilerExpression source)
    {
        var inputType = context.Inputs.FirstOrDefault(value => string.Equals(value.Name, inputName, StringComparison.Ordinal))?.Type as CompilerTensorType;
        if (inputType?.Dimensions is not { } dimensions)
        {
            throw context.Unsupported(source, "TorchSharp split output count requires a statically known input rank and axis dimension.");
        }

        var normalizedAxis = axis < 0 ? axis + dimensions.Count : axis;
        if (normalizedAxis < 0 || normalizedAxis >= dimensions.Count
            || dimensions[(int)normalizedAxis] is not CompilerFixedDimension fixedDimension)
        {
            throw context.Unsupported(source, "TorchSharp split output count requires a fixed dimension at the selected axis.");
        }

        return fixedDimension.Value;
    }
}
