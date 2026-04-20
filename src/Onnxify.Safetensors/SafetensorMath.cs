namespace Onnxify.Safetensors;

internal static class SafetensorMath
{
    public static ulong ComputeElementCount(IReadOnlyList<ulong> shape)
    {
        ArgumentNullException.ThrowIfNull(shape);

        ulong n = 1;
        checked
        {
            foreach (var dim in shape)
            {
                n *= dim;
            }
        }

        return n;
    }

    public static ulong ComputeSizeInBytes(DataType dtype, IReadOnlyList<ulong> shape, bool allowMisaligned)
    {
        try
        {
            var elementCount = ComputeElementCount(shape);
            var bitCount = checked(elementCount * (ulong)dtype.Bitsize());

            if (!allowMisaligned && bitCount % 8 != 0)
            {
                throw SafetensorException.MisalignedSlice();
            }

            return bitCount / 8;
        }
        catch (OverflowException)
        {
            throw SafetensorException.ValidationOverflow();
        }
    }
}
