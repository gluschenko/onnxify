using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class MatMulOperator() : MatrixMultiplicationOperator<Onnxify.MatMul>("MatMul")
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.matmul"),
        CompilerTorchSharpForm.Call("torch.mm"),
        CompilerTorchSharpForm.Call("torch.bmm"),
        CompilerTorchSharpForm.Call("Tensor.matmul"),
        CompilerTorchSharpForm.Call("Tensor.mm"),
        CompilerTorchSharpForm.Call("Tensor.bmm"),
    ];
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;

    protected override bool IsSupportedElementType(CompilerElementType elementType)
    {
        return elementType is CompilerElementType.Float32
            or CompilerElementType.Float64
            or CompilerElementType.Int32
            or CompilerElementType.Int64;
    }

    protected override string? ValidateTypedNode(
        Onnxify.MatMul node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var left = GetTensorType(node.A.Name, knownTypes);
        var right = GetTensorType(node.B.Name, knownTypes);
        if (left is null || right is null)
        {
            return null;
        }

        if (ValidateElementTypes(left, right) is { } elementTypeError)
        {
            return elementTypeError;
        }

        var aDimensions = GetFixedDimensions(left);
        var bDimensions = GetFixedDimensions(right);
        if (aDimensions is null || bDimensions is null)
        {
            return null;
        }

        if (aDimensions.Length == 0 || bDimensions.Length == 0)
        {
            return "ONNX MatMul requires both inputs to have rank at least 1.";
        }

        var leftContract = aDimensions[aDimensions.Length - 1];
        var rightContract = bDimensions.Length == 1
            ? bDimensions[0]
            : bDimensions[bDimensions.Length - 2];
        if (leftContract != rightContract)
        {
            return $"ONNX MatMul inner dimensions are incompatible ({leftContract} and {rightContract}).";
        }

        var leftBatch = aDimensions.Length == 1
            ? Array.Empty<long>()
            : aDimensions.Take(aDimensions.Length - 2).ToArray();
        var rightBatch = bDimensions.Length == 1
            ? Array.Empty<long>()
            : bDimensions.Take(bDimensions.Length - 2).ToArray();
        if (!CanMultidirectionallyBroadcast(leftBatch, rightBatch))
        {
            return "ONNX MatMul batch dimensions cannot be broadcast together.";
        }

        return null;
    }

    protected override string PrintTorchSharp(Onnxify.MatMul node, CompilerSourceSpan? span)
    {
        return $"torch.matmul({node.A.Name}, {node.B.Name})";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        var operands = expression is CompilerInvocationExpression invocation
            ? (context.Receiver is null ? Array.Empty<CompilerExpression>() : [context.Receiver]).Concat(invocation.Arguments).ToArray()
            : Array.Empty<CompilerExpression>();
        if (operands.Length != 2)
        {
            throw context.Unsupported(expression, $"TorchSharp call '{context.CallName}' requires exactly two tensor operands.");
        }

        var node = new Onnxify.MatMul(
            name: OnnxName.ToLowerInvariant(),
            options: new MatMulInputOutputOptions
            {
                A = new OnnxEdge(context.RequireTensorReference(operands[0], context.CallName).Name),
                B = new OnnxEdge(context.RequireTensorReference(operands[1], context.CallName).Name),
                Y = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }
}

