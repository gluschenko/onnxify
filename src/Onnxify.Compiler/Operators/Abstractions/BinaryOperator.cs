using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class BinaryOperator<TNode> : BroadcastOperator<TNode>
    where TNode : OnnxNode
{
    protected BinaryOperator(
        string name,
        string methodName,
        bool supportsBroadcast = false,
        CompilerElementType? outputType = null,
        IReadOnlyList<CompilerElementType?>? inputTypes = null,
        string? binaryToken = null,
        IEnumerable<string>? forms = null)
        : base(CompilerOperatorIdentity.Onnx(name))
    {
        MethodName = methodName;
        OperatorTokenValue = binaryToken ?? string.Empty;
        SupportsMultidirectionalBroadcast = supportsBroadcast;
        OutputElementType = outputType;
        InputElementTypes = inputTypes ?? Array.Empty<CompilerElementType?>();
        var aliases = forms ?? [$"torch.{methodName}", $"Tensor.{methodName}"];
        TorchSharpForms = aliases.Select(CompilerTorchSharpForm.Call)
            .Concat(binaryToken is null ? Array.Empty<CompilerTorchSharpForm>() : [CompilerTorchSharpForm.Binary(binaryToken)])
            .ToArray();
    }

    protected string MethodName { get; }
    public override string? TorchSharpMethod => MethodName;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; }
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;
    public override bool SupportsMultidirectionalBroadcast { get; }
    public override CompilerElementType? OutputElementType { get; }
    public override IReadOnlyList<CompilerElementType?> InputElementTypes { get; }
    protected override string PrintTorchSharp(TNode node, CompilerSourceSpan? span)
    {
        return $"({CompilerCSharpNaming.Identifier(node.Inputs[0].Name)} {OperatorToken} {CompilerCSharpNaming.Identifier(node.Inputs[1].Name)})";
    }
    private string OperatorTokenValue { get; }
    protected virtual string OperatorToken => OperatorTokenValue;

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        IReadOnlyList<CompilerExpression> operands = expression switch
        {
            CompilerBinaryExpression binary => [binary.Left, binary.Right],
            CompilerInvocationExpression invocation => (context.Receiver is null
                    ? Array.Empty<CompilerExpression>()
                    : [context.Receiver])
                .Concat(invocation.Arguments)
                .ToArray(),
            _ => Array.Empty<CompilerExpression>(),
        };
        if (operands.Count != 2)
        {
            throw context.Unsupported(expression, $"TorchSharp operator '{OnnxName}' requires exactly two operands.");
        }

        var references = new List<CompilerValueReference>(2);
        for (var index = 0; index < operands.Count; index++)
        {
            if (operands[index] is CompilerReferenceExpression reference
                && context.Inputs.Any(input => string.Equals(input.Name, reference.Name, StringComparison.Ordinal)
                    && input.Type is CompilerTensorType))
            {
                references.Add(new CompilerValueReference(reference.Name));
                continue;
            }
            if (context.TryAddScalarInitializer(this, index, operands[index], out var scalar))
            {
                references.Add(scalar);
                continue;
            }
            throw context.Unsupported(
                operands[index],
                $"TorchSharp operator '{OnnxName}' requires tensor inputs or a scalar literal paired with a tensor input.");
        }
        context.AddOperation(this, references, span: expression.Span);
        return true;
    }
}

