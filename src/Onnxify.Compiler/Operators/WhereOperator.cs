using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class WhereOperator() : BroadcastOperator<Onnxify.Where>(CompilerOperatorIdentity.Onnx("Where"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [CompilerTorchSharpForm.Call("torch.where")];
    public override int MinimumInputCount => 3;
    public override int MaximumInputCount => 3;
    public override bool SupportsMultidirectionalBroadcast => true;
    public override IReadOnlyList<CompilerElementType?> InputElementTypes { get; } = [CompilerElementType.Boolean, null, null];
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

