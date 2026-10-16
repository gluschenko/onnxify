using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class MatrixMultiplicationOperator<TNode>(string name)
    : CompilerOperator<TNode>(CompilerOperatorIdentity.Onnx(name))
    where TNode : OnnxNode
{
    public sealed override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not TNode typedNode)
        {
            return $"ONNX {OnnxName} requires generated node type '{typeof(TNode).Name}'.";
        }

        return ValidateTypedNode(typedNode, knownTypes);
    }

    protected abstract string? ValidateTypedNode(
        TNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes);

    protected static CompilerTensorType? GetTensorType(
        string inputName,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return knownTypes.TryGetValue(inputName, out var type) ? type as CompilerTensorType : null;
    }

    protected string? ValidateElementTypes(
        CompilerTensorType left,
        CompilerTensorType right,
        IReadOnlyCollection<CompilerTensorType?>? additionalInputs = null)
    {
        if (left.ElementType != right.ElementType)
        {
            return $"ONNX {OnnxName} requires both matrix operands to have the same element type.";
        }

        if (!IsSupportedElementType(left.ElementType))
        {
            return $"ONNX {OnnxName} does not have a registered runtime-verified mapping for element type '{left.ElementType}'.";
        }

        if (additionalInputs?.Any(input => input is not null && input.ElementType != left.ElementType) == true)
        {
            return $"ONNX {OnnxName} requires all tensor inputs to have the same element type.";
        }

        return null;
    }

    protected abstract bool IsSupportedElementType(CompilerElementType elementType);

    protected static long[]? GetFixedDimensions(CompilerTensorType tensorType)
    {
        if (tensorType.Dimensions is null
            || tensorType.Dimensions.Any(static dimension => dimension is not CompilerFixedDimension))
        {
            return null;
        }

        return tensorType.Dimensions
            .Cast<CompilerFixedDimension>()
            .Select(static dimension => dimension.Value)
            .ToArray();
    }

    protected static bool CanMultidirectionallyBroadcast(long[] left, long[] right)
    {
        var rank = Math.Max(left.Length, right.Length);
        for (var offset = 1; offset <= rank; offset++)
        {
            var leftDimension = offset <= left.Length ? left[left.Length - offset] : 1;
            var rightDimension = offset <= right.Length ? right[right.Length - offset] : 1;
            if (leftDimension != rightDimension && leftDimension != 1 && rightDimension != 1)
            {
                return false;
            }
        }

        return true;
    }

    protected static bool CanBroadcastTo(long[] source, long[] target)
    {
        var rank = Math.Max(source.Length, target.Length);
        for (var offset = 1; offset <= rank; offset++)
        {
            var sourceDimension = offset <= source.Length ? source[source.Length - offset] : 1;
            var targetDimension = offset <= target.Length ? target[target.Length - offset] : 1;
            if (sourceDimension != targetDimension && sourceDimension != 1)
            {
                return false;
            }
        }

        return true;
    }
}
