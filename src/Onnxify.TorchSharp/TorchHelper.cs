namespace Onnxify.TorchSharp;

internal static class TorchHelper
{
    internal static long[] GetShape(this global::TorchSharp.torch.Tensor tensor)
    {
        return [.. tensor.shape];
    }

    internal static float[] GetFloatData(this global::TorchSharp.torch.Tensor tensor)
    {
        return [.. tensor.detach().cpu().data<float>()];
    }

    internal static long[] ToLongArray(IEnumerable<long>? value)
    {
        return value?.ToArray() ?? [];
    }

    internal static long[] ToLongArray(long value)
    {
        return [value];
    }

    internal static long[] ToLongArray(long? value)
    {
        return value is long x ? [x] : [];
    }

    internal static long[] ExpandPadding(long[] padding, int spatialRank)
    {
        if (padding.Length == 0)
        {
            return [];
        }

        if (padding.Length == spatialRank * 2)
        {
            return padding;
        }

        if (padding.Length == 1 && spatialRank == 1)
        {
            return [padding[0], padding[0]];
        }

        return (padding.Length, spatialRank) switch
        {
            (1, 2) => [padding[0], padding[0], padding[0], padding[0]],
            (2, 2) => [padding[0], padding[1], padding[0], padding[1]],
            (1, 3) => [padding[0], padding[0], padding[0], padding[0], padding[0], padding[0]],
            (3, 3) => [padding[0], padding[1], padding[2], padding[0], padding[1], padding[2]],
            _ => throw new NotSupportedException(
                $"Unsupported padding rank: padding.Length={padding.Length}, spatialRank={spatialRank}."
            ),
        };
    }
}

