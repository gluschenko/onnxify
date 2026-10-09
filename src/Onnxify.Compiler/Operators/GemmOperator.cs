using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class GemmOperator() : MatrixMultiplicationOperator<Onnxify.Gemm>("Gemm")
{
    protected override bool AllowsEmptyOptionalInputs => true;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.addmm"),
        CompilerTorchSharpForm.Call("Tensor.addmm"),
        CompilerTorchSharpForm.Call("torch.nn.functional.linear"),
    ];
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 3;
    public override IReadOnlyList<string> AttributeNames { get; } = ["alpha", "beta", "transA", "transB"];

    protected override bool IsSupportedElementType(CompilerElementType elementType)
    {
        return elementType is CompilerElementType.Float32 or CompilerElementType.Float64;
    }

    protected override string? ValidateTypedNode(
        Onnxify.Gemm node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node.TransA is not (null or 0 or 1))
        {
            return "ONNX Gemm attribute 'transA' must be 0 or 1.";
        }

        if (node.TransB is not (null or 0 or 1))
        {
            return "ONNX Gemm attribute 'transB' must be 0 or 1.";
        }

        var left = GetTensorType(node.A.Name, knownTypes);
        var right = GetTensorType(node.B.Name, knownTypes);
        var bias = node.C is null ? null : GetTensorType(node.C.Name, knownTypes);
        if (left is null || right is null)
        {
            return null;
        }

        if (ValidateElementTypes(left, right, [bias]) is { } elementTypeError)
        {
            return elementTypeError;
        }

        var aDimensions = GetFixedDimensions(left);
        var bDimensions = GetFixedDimensions(right);
        if (aDimensions is null || bDimensions is null)
        {
            return null;
        }

        if (aDimensions.Length != 2 || bDimensions.Length != 2)
        {
            return "ONNX Gemm requires rank-2 A and B inputs.";
        }

        var aTransposed = node.TransA == 1;
        var bTransposed = node.TransB == 1;
        var aRows = aDimensions[aTransposed ? 1 : 0];
        var aColumns = aDimensions[aTransposed ? 0 : 1];
        var bRows = bDimensions[bTransposed ? 1 : 0];
        var bColumns = bDimensions[bTransposed ? 0 : 1];
        if (aColumns != bRows)
        {
            return $"ONNX Gemm inner dimensions are incompatible ({aColumns} and {bRows}).";
        }

        var biasDimensions = bias is null ? null : GetFixedDimensions(bias);
        if (biasDimensions is null)
        {
            return null;
        }

        if (biasDimensions.Length > 2)
        {
            return "ONNX Gemm bias input C must have rank at most 2.";
        }

        if (!CanBroadcastTo(biasDimensions, [aRows, bColumns]))
        {
            return "ONNX Gemm bias input C cannot be broadcast to the matrix output shape.";
        }

        return null;
    }

    protected override string PrintTorchSharp(Onnxify.Gemm node, CompilerSourceSpan? span)
    {
        var a = CompilerCSharpNaming.Identifier(node.A.Name);
        var b = CompilerCSharpNaming.Identifier(node.B.Name);
        var c = node.C is null ? string.Empty : CompilerCSharpNaming.Identifier(node.C.Name);
        if (node.TransA == 1)
        {
            a += ".transpose(0, 1)";
        }

        if (node.TransB == 1)
        {
            b += ".transpose(0, 1)";
        }

        var product = $"torch.matmul({a}, {b})";
        var alpha = node.Alpha ?? 1f;
        if (alpha != 1f)
        {
            product = $"({Float(alpha)} * {product})";
        }

        if (c.Length != 0)
        {
            product = $"({product} + ({Float(node.Beta ?? 1f)} * {c}))";
        }

        return product;
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        var arguments = (context.Receiver is null ? Array.Empty<CompilerExpression>() : [context.Receiver]).Concat(invocation.Arguments).ToArray();
        string a;
        string b;
        string? c;
        float? alpha = null;
        float? beta = null;
        long? transB = null;
        if (string.Equals(context.CallName, "torch.nn.functional.linear", StringComparison.Ordinal))
        {
            if (context.Receiver is not null || arguments.Length is < 2 or > 3)
            {
                throw context.Unsupported(expression, "torch.nn.functional.linear requires input, weight, and optional bias.");
            }

            a = context.RequireTensorReference(arguments[0], context.CallName).Name;
            b = context.RequireTensorReference(arguments[1], context.CallName).Name;
            c = arguments.Length == 3
                ? context.RequireTensorReference(arguments[2], context.CallName).Name
                : null;
            transB = 1;
        }
        else
        {
            if (arguments.Length is < 3 or > 5)
            {
                throw context.Unsupported(expression, "torch.addmm requires input, mat1, mat2, and optional constant beta and alpha values.");
            }

            c = context.RequireTensorReference(arguments[0], context.CallName).Name;
            a = context.RequireTensorReference(arguments[1], context.CallName).Name;
            b = context.RequireTensorReference(arguments[2], context.CallName).Name;
            beta = arguments.Length >= 4 ? context.RequireNumericLiteral(arguments[3], "beta") : 1f;
            alpha = arguments.Length >= 5 ? context.RequireNumericLiteral(arguments[4], "alpha") : 1f;
        }

        var node = new Onnxify.Gemm(
            name: OnnxName.ToLowerInvariant(),
            options: new GemmInputOutputOptions
            {
                A = new OnnxEdge(a),
                B = new OnnxEdge(b),
                C = c is null ? null : new OnnxEdge(c),
                Alpha = alpha,
                Beta = beta,
                TransA = null,
                TransB = transB,
                Y = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }
}

