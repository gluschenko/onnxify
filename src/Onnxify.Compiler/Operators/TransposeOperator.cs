using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class TransposeOperator() : CompilerOperator<Onnxify.Transpose>(CompilerOperatorIdentity.Onnx("Transpose"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.permute"),
        CompilerTorchSharpForm.Call("Tensor.permute"),
    ];

    public override IReadOnlyList<string> AttributeNames { get; } = ["perm"];

    protected override string PrintTorchSharp(Onnxify.Transpose node, CompilerSourceSpan? span)
    {
        if (node.Perm is not { Length: > 0 } permutation)
        {
            throw Unsupported(span, "Compiler Transpose requires an explicit permutation.");
        }

        var input = CompilerCSharpNaming.Identifier(node.Data.Name);
        var dimensions = string.Join(", ", permutation.Select(static dimension => dimension.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        return $"{input}.permute(new long[] {{ {dimensions} }})";
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Transpose transpose)
        {
            return $"Transpose requires the generated node type '{nameof(Onnxify.Transpose)}'.";
        }

        if (!IsPermutation(transpose.Perm))
        {
            return "Compiler Transpose requires an explicit permutation without duplicate or negative dimensions.";
        }

        return knownTypes.TryGetValue(transpose.Data.Name, out var inputType)
            && inputType is CompilerTensorType { Dimensions: not null } tensorType
            && tensorType.Dimensions.Count != transpose.Perm!.Length
            ? "Transpose permutation length must match the input tensor rank."
            : null;
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        var inputExpression = context.Receiver ?? invocation.Arguments.FirstOrDefault();
        if (inputExpression is null)
        {
            throw context.Unsupported(expression, "TorchSharp permute requires a tensor input.");
        }

        var permutationExpression = context.Receiver is null
            ? invocation.Arguments.Skip(1).FirstOrDefault()
            : invocation.Arguments.FirstOrDefault();
        if (permutationExpression is not CompilerArrayExpression permutationArray)
        {
            throw context.Unsupported(expression, "TorchSharp permute requires a compile-time integer array.");
        }

        var permutation = permutationArray.Items.Select(item => RequireDimension(item, context)).ToArray();
        if (!IsPermutation(permutation))
        {
            throw context.Unsupported(expression, "TorchSharp permute dimensions must form a permutation from zero to rank minus one.");
        }

        if (context.Receiver is null && invocation.Arguments.Count != 2)
        {
            throw context.Unsupported(expression, "torch.permute accepts only an input tensor and one permutation array.");
        }

        if (context.Receiver is not null && invocation.Arguments.Count != 1)
        {
            throw context.Unsupported(expression, "Tensor.permute accepts only one permutation array.");
        }

        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var inputType = context.Inputs
            .FirstOrDefault(value => string.Equals(value.Name, input.Name, StringComparison.Ordinal))
            ?.Type as CompilerTensorType;
        if (inputType?.Dimensions is { } dimensions && dimensions.Count != permutation.Length)
        {
            throw context.Unsupported(expression, "TorchSharp permute permutation length must match the input tensor rank.");
        }

        var node = new Onnxify.Transpose(
            OnnxName.ToLowerInvariant(),
            new Onnxify.TransposeInputOutputOptions
            {
                Data = new OnnxEdge(input.Name),
                Perm = permutation,
                Transposed = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }

    private static long RequireDimension(CompilerExpression expression, TorchSharpOperatorScanContext context)
    {
        return expression switch
        {
            CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signed } => signed.Value,
            CompilerLiteralExpression { Literal: CompilerUnsignedIntegerLiteral unsigned } when unsigned.Value <= long.MaxValue => (long)unsigned.Value,
            _ => throw context.Unsupported(expression, "TorchSharp permute dimensions must be compile-time integers."),
        };
    }

    private static bool IsPermutation(IReadOnlyList<long>? permutation)
    {
        return permutation is { Count: > 0 }
            && permutation.All(dimension => dimension >= 0 && dimension < permutation.Count)
            && permutation.Distinct().Count() == permutation.Count;
    }
}
