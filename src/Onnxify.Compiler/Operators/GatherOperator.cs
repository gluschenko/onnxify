using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::embedding")]
internal sealed class GatherOperator() : CompilerOperator<Onnxify.Gather>(CompilerOperatorIdentity.Onnx("Gather"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [CompilerTorchSharpForm.Call("Tensor.index_select")];
    public override IReadOnlyList<string> AttributeNames { get; } = ["axis"];
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;
    public override IReadOnlyList<CompilerElementType?> InputElementTypes { get; } = [null, CompilerElementType.Int64];
    protected override string PrintTorchSharp(Onnxify.Gather node, CompilerSourceSpan? span)
    {
        return $"Gather({CompilerCSharpNaming.Identifier(node.Data.Name)}, {CompilerCSharpNaming.Identifier(node.Indices.Name)}, {node.Axis ?? 0}L)";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (context.CallName != "Tensor.index_select" || context.Receiver is null || context.Invocation is not { Arguments.Count: 2 } invocation)
            return false;
        var axis = StaticInt64Vector.ReadScalar(invocation.Arguments[0], context, "Gather axis");
        var data = context.RequireTensorReference(context.Receiver, OnnxName);
        var indices = context.RequireTensorReference(invocation.Arguments[1], OnnxName);
        var output = context.RequireSingleOutputEdge(this);
        context.AddOperation(this, new Onnxify.Gather("gather", new Onnxify.GatherInputOutputOptions
        {
            Data = new OnnxEdge(data.Name),
            Indices = new OnnxEdge(indices.Name),
            Axis = axis,
            Output = output,
        }), expression.Span);
        return true;
    }
}
