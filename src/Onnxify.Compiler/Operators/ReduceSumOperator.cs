using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

internal sealed class ReduceSumOperator : CompilerOperator
{
    private static readonly string[] _attributes = ["axes", "keepdims", "noop_with_empty_axes"];

    public ReduceSumOperator() : base(CompilerOperatorIdentity.Onnx("ReduceSum")) { }

    public override Type NodeType => typeof(OnnxNode);
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
        [CompilerTorchSharpForm.Call("torch.sum")];
    public override int MinimumInputCount => 1;
    public override int MaximumInputCount => 2;
    public override IReadOnlyList<string> AttributeNames => _attributes;
    protected override bool AllowsEmptyOptionalInputs => true;

    public override CompilerOnnxStep CreateNode(
        string name,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        IEnumerable<CompilerAttribute>? attributes = null,
        CompilerSourceSpan? span = null)
    {
        var node = new OnnxNode(
            name,
            OnnxName,
            OnnxDomain,
            string.Empty,
            inputs.Select(static value => (IOnnxGraphEdge)new OnnxEdge(value.IsEmptyOptional ? string.Empty : value.Name)),
            outputs.Select(static value => (IOnnxGraphEdge)new OnnxEdge(value.IsEmptyOptional ? string.Empty : value.Name)),
            (attributes ?? []).Select(attribute => OnnxCompilerBackend.CreateOnnxAttribute(attribute, [], null)));
        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    public override CompilerOnnxStep ScanOnnx(OnnxNode node, string name, CompilerSourceSpan? span) =>
        new(node, Descriptor, span, name);

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        var node = operation.Node;
        var input = CompilerCSharpNaming.Identifier(node.Inputs[0].Name);
        var keepDims = AttributeValue<long>(node, "keepdims", 1L) != 0;
        if (node.Inputs.Count > 1 && !string.IsNullOrEmpty(node.Inputs[1].Name))
        {
            var axesInput = CompilerCSharpNaming.Identifier(node.Inputs[1].Name);
            var noop = AttributeValue<long>(node, "noop_with_empty_axes", 0L) != 0;
            return $"ReduceSum({input}, {axesInput}.data<long>().ToArray(), {keepDims.ToString().ToLowerInvariant()}, {noop.ToString().ToLowerInvariant()})";
        }

        var axes = AttributeValue<long[]>(node, "axes", []);
        if (axes.Length == 0 && AttributeValue<long>(node, "noop_with_empty_axes", 0L) == 0)
        {
            return $"torch.sum({input}, keepdim: {keepDims.ToString().ToLowerInvariant()})";
        }

        return $"ReduceSum({input}, new long[] {{ {StaticInt64Vector.Format(axes)} }}, {keepDims.ToString().ToLowerInvariant()}, {(AttributeValue<long>(node, "noop_with_empty_axes", 0L) != 0).ToString().ToLowerInvariant()})";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (context.CallName != "torch.sum" || context.Receiver is not null
            || context.Invocation is not { Arguments.Count: >= 1 and <= 3 } invocation)
        {
            return false;
        }

        var input = context.RequireTensorReference(invocation.Arguments[0], OnnxName);
        long[] axes = [];
        if (invocation.Arguments.Count > 1)
        {
            axes = invocation.Arguments[1] is CompilerArrayExpression
                ? StaticInt64Vector.ReadVector(invocation.Arguments[1], context, "ReduceSum axes")
                : [StaticInt64Vector.ReadScalar(invocation.Arguments[1], context, "ReduceSum axis")];
        }

        var keepDims = invocation.Arguments.Count > 2
            && invocation.Arguments[2] is CompilerLiteralExpression { Literal: CompilerBooleanLiteral { Value: true } };
        var axesName = $"__{context.Outputs[0].Name}_axes";
        StaticInt64Vector.AddInitializer(context, axesName, axes);
        context.AddOperation(
            this,
            [input, new CompilerValueReference(axesName)],
            [new CompilerAttribute("keepdims", new CompilerSignedIntegerLiteral(CompilerElementType.Int64, keepDims ? 1 : 0)),
             new CompilerAttribute("noop_with_empty_axes", new CompilerSignedIntegerLiteral(CompilerElementType.Int64, invocation.Arguments.Count > 1 && axes.Length == 0 ? 1 : 0))],
            expression.Span);
        return true;
    }

    private static T AttributeValue<T>(OnnxNode node, string name, T defaultValue)
    {
        var attribute = node.Attributes.FirstOrDefault(attribute => string.Equals(attribute.Name, name, StringComparison.Ordinal));
        var value = attribute?.GetValue();
        return value is T typed ? typed : defaultValue;
    }
}
