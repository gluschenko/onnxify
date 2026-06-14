namespace Onnxify.Helpers;

internal static class FloatBitConverter
{
    public static int SingleToInt32Bits(float value)
    {
#if NETSTANDARD2_0
        return BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
#else
        return BitConverter.SingleToInt32Bits(value);
#endif
    }

    public static float Int32BitsToSingle(int value)
    {
#if NETSTANDARD2_0
        return BitConverter.ToSingle(BitConverter.GetBytes(value), 0);
#else
        return BitConverter.Int32BitsToSingle(value);
#endif
    }
}
