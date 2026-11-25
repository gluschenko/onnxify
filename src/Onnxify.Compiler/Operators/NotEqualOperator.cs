using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::ne")]
[CompilerTorchOperator("aten::ne.Tensor")]
[CompilerTorchOperator("aten::ne.Scalar")]
[CompilerTorchOperator("_operator::ne")]
internal sealed class NotEqualOperator : TorchSharpCompositeOperator
{
    public NotEqualOperator()
        : base("NotEqual", "com.onnxify.compiler", "torch.ne", "Tensor.ne")
    {
    }

    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;
    public override bool SupportsMultidirectionalBroadcast => true;
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.ne"),
        CompilerTorchSharpForm.Call("Tensor.ne"),
        CompilerTorchSharpForm.Binary("!="),
    ];

    protected override bool AllowsScalarOperand(int index) => index is 0 or 1;

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        return $"torch.ne({CompilerCSharpNaming.Identifier(operation.Node.Inputs[0].Name)}, {CompilerCSharpNaming.Identifier(operation.Node.Inputs[1].Name)})";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        IReadOnlyList<CompilerExpression> operands = expression switch
        {
            CompilerBinaryExpression { Operator: "!=" } binary => [binary.Left, binary.Right],
            CompilerInvocationExpression invocation => (context.Receiver is null
                    ? Array.Empty<CompilerExpression>()
                    : [context.Receiver])
                .Concat(invocation.Arguments)
                .ToArray(),
            _ => [],
        };
        if (operands.Count != 2)
        {
            throw context.Unsupported(expression, "Pointwise NotEqual requires exactly two operands.");
        }

        var inputs = new List<CompilerValueReference>(2);
        for (var index = 0; index < operands.Count; index++)
        {
            if (operands[index] is CompilerReferenceExpression)
            {
                inputs.Add(context.RequireTensorReference(operands[index], OnnxName));
            }
            else if (context.TryAddScalarInitializer(this, index, operands[index], out var scalar))
            {
                inputs.Add(scalar);
            }
            else
            {
                throw context.Unsupported(operands[index], "Pointwise NotEqual requires tensor inputs or a supported scalar literal.");
            }
        }

        context.AddOperation(this, inputs, span: expression.Span);
        return true;
    }

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Equal>(0, out var equal)
            || !context.TryGetNode<Onnxify.Not>(1, out var not)
            || equal is null
            || not is null
            || !string.Equals(equal.C.Name, not.Inputs[0].Name, StringComparison.Ordinal)
            || !context.CanCollapse(2))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [equal.A.Name, equal.B.Name], not.Outputs[0].Name, 2);
        consumedNodeCount = 2;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var equal = new Onnxify.Equal(
            $"{operation.Name}__equal",
            new Onnxify.EqualInputOutputOptions
            {
                A = Input(operation, 0),
                B = Input(operation, 1),
                C = new OnnxEdge($"{operation.Name}__equal_result"),
            });
        graph.AddNode(equal);
        graph.AddNode(new Onnxify.Not(
            operation.Name,
            new Onnxify.NotInputOutputOptions
            {
                X = equal.C,
                Y = Output(operation),
            }));
    }
}
