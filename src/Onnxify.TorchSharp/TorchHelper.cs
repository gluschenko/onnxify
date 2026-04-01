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

    internal static long[] ExpandPadding(long[] padding)
    {
        return padding.Length switch
        {
            1 => [padding[0], padding[0], padding[0], padding[0]],
            2 => [padding[0], padding[1], padding[0], padding[1]],
            4 => padding,
            _ => throw new NotSupportedException($"Unsupported padding rank: {padding.Length}."),
        };
    }
}

