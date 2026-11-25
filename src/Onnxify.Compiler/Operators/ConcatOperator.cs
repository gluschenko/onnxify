using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::cat")]
[CompilerTorchOperator("aten::concat")]
[CompilerTorchOperator("aten::concatenate")]
internal sealed class ConcatOperator() : CompilerOperator<Onnxify.Concat>(CompilerOperatorIdentity.Onnx("Concat"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.cat"),
        CompilerTorchSharpForm.Call("torch.concat"),
        CompilerTorchSharpForm.Call("torch.concatenate"),
    ];

    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => int.MaxValue;
    public override IReadOnlyList<string> AttributeNames { get; } = ["axis"];

    protected override string PrintTorchSharp(Onnxify.Concat node, CompilerSourceSpan? span)
    {
        var inputs = string.Join(", ", node.In.Select(static input => CompilerCSharpNaming.Identifier(input.Name)));
        return $"torch.cat(new global::TorchSharp.torch.Tensor[] {{ {inputs} }}, dim: {node.Axis})";
    }

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Concat concat)
        {
            return $"Concat requires the generated node type '{nameof(Onnxify.Concat)}'.";
        }

        if (concat.In.Count < 2)
        {
            return "Compiler Concat requires at least two tensor inputs.";
        }

        return knownTypes.TryGetValue(concat.In[0].Name, out var inputType)
            && inputType is CompilerTensorType { Dimensions: not null } tensorType
            && (concat.Axis < -tensorType.Dimensions.Count || concat.Axis >= tensorType.Dimensions.Count)
                ? "Concat axis is outside the input rank."
                : null;
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation || invocation.Arguments.Count != 2)
        {
            return false;
        }

        if (invocation.Arguments[0] is not CompilerArrayExpression inputArray || inputArray.Items.Count < 2)
        {
            throw context.Unsupported(expression, "TorchSharp cat requires an array of at least two tensor references.");
        }

        var inputs = inputArray.Items
            .Select(item => context.RequireTensorReference(item, OnnxName))
            .ToArray();
        var axis = StaticInt64Vector.ReadScalar(invocation.Arguments[1], context, "TorchSharp cat axis");
        var output = context.RequireSingleOutputEdge(this);
        var node = new Onnxify.Concat(
            "concat",
            new Onnxify.ConcatInputOutputOptions
            {
                In = inputs.Select(static input => (IOnnxGraphEdge)new OnnxEdge(input.Name)).ToArray(),
                Axis = axis,
                ConcatResult = output,
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }
}
