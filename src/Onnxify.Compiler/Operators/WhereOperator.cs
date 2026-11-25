using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::where.self")]
[CompilerTorchOperator("aten::where.Scalar")]
[CompilerTorchOperator("aten::where.ScalarSelf")]
[CompilerTorchOperator("aten::where.ScalarOther")]
internal sealed class WhereOperator() : BroadcastOperator<Onnxify.Where>(CompilerOperatorIdentity.Onnx("Where"))
{
    private static readonly IReadOnlyCollection<CompilerElementType> SUPPORTED_VALUE_TYPES =
    [CompilerElementType.UInt8, CompilerElementType.Int32, CompilerElementType.Int64, CompilerElementType.Float32, CompilerElementType.Float64];

    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [CompilerTorchSharpForm.Call("torch.where")];
    public override int MinimumInputCount => 3;
    public override int MaximumInputCount => 3;
    public override bool SupportsMultidirectionalBroadcast => true;
    public override IReadOnlyList<CompilerElementType?> InputElementTypes { get; } = [CompilerElementType.Boolean, null, null];

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        if (broadcastError is not null)
        {
            return broadcastError;
        }

        if (node is not Onnxify.Where where)
        {
            return "Where requires the generated ONNX node type.";
        }

        if (knownTypes.TryGetValue(where.Condition.Name, out var conditionType)
            && conditionType is CompilerTensorType conditionTensor
            && conditionTensor.ElementType != CompilerElementType.Boolean)
        {
            return $"ONNX Where condition must be Boolean; found '{conditionTensor.ElementType}'.";
        }

        CompilerElementType? branchType = null;
        foreach (var branch in new[] { where.X, where.Y })
        {
            if (branch is null || !knownTypes.TryGetValue(branch.Name, out var branchCompilerType)
                || branchCompilerType is not CompilerTensorType tensorType)
            {
                continue;
            }

            if (!SUPPORTED_VALUE_TYPES.Contains(tensorType.ElementType))
            {
                return $"ONNX Where value type '{tensorType.ElementType}' is unsupported by the ONNX Runtime CPU kernel and TorchLib converter intersection.";
            }

            if (branchType is { } knownBranchType && knownBranchType != tensorType.ElementType)
            {
                return $"ONNX Where requires matching value element types; found '{knownBranchType}' and '{tensorType.ElementType}'.";
            }

            branchType ??= tensorType.ElementType;
        }

        if (branchType is { } selectedType && knownTypes.TryGetValue(where.Output.Name, out var outputType)
            && outputType is CompilerTensorType outputTensor && outputTensor.ElementType != selectedType)
        {
            return $"ONNX Where output type '{outputTensor.ElementType}' must match its value inputs '{selectedType}'.";
        }

        return null;
    }

    protected override string PrintTorchSharp(Onnxify.Where node, CompilerSourceSpan? span)
    {
        return $"torch.where({string.Join(", ", node.Inputs.Select(static input => input.Name))})";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation || invocation.Arguments.Count != 3)
        {
            throw context.Unsupported(expression, "torch.where requires condition, true-value, and false-value tensors.");
        }

        var node = new Onnxify.Where(
            name: OnnxName.ToLowerInvariant(),
            options: new WhereInputOutputOptions
            {
                Condition = new OnnxEdge(context.RequireTensorReference(invocation.Arguments[0], "torch.where").Name),
                X = new OnnxEdge(context.RequireTensorReference(invocation.Arguments[1], "torch.where").Name),
                Y = new OnnxEdge(context.RequireTensorReference(invocation.Arguments[2], "torch.where").Name),
                Output = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }
}

