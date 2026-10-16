using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class UnaryOperator<TNode> : CompilerOperator<TNode>
    where TNode : OnnxNode
{
    protected UnaryOperator(string name, string methodName, IEnumerable<string>? forms = null)
        : base(CompilerOperatorIdentity.Onnx(name))
    {
        MethodName = methodName;
        TorchSharpForms = (forms ?? [$"torch.{methodName}", $"Tensor.{methodName}"])
            .Select(CompilerTorchSharpForm.Call)
            .ToArray();
    }

    protected string MethodName { get; }
    public override string? TorchSharpMethod => MethodName;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; }
    protected override string PrintTorchSharp(TNode node, CompilerSourceSpan? span)
    {
        return $"{CompilerCSharpNaming.Identifier(node.Inputs[0].Name)}.{MethodName}()";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        var operand = expression switch
        {
            CompilerUnaryExpression unary => unary.Expression,
            CompilerInvocationExpression invocation when invocation.Arguments.Count == 1 => invocation.Arguments[0],
            CompilerInvocationExpression invocation when context.Receiver is not null && invocation.Arguments.Count == 0 => context.Receiver,
            _ => throw context.Unsupported(expression, $"TorchSharp operator '{OnnxName}' requires one tensor input."),
        };
        context.AddOperation(this, [context.RequireTensorReference(operand, OnnxName)], span: expression.Span);
        return true;
    }
}

