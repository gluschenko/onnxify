using Onnxify.Data.Numerics;
#if NET5_0_OR_GREATER
using Half = System.Half;
#else
using Half = Onnxify.Data.Numerics.Float16;
#endif

namespace Onnxify.Helpers;

/// <summary>Central conversions for CLR numeric types whose representation differs by target framework.</summary>
internal static class OnnxNumericConversions
{
    public static Half Float16FromSingle(float value)
    {
#if NET5_0_OR_GREATER
        return (Half)value;
#else
        return Float16.FromSingle(value);
#endif
    }

    public static float Float16ToSingle(Half value)
    {
#if NET5_0_OR_GREATER
        return (float)value;
#else
        return value.ToSingle();
#endif
    }

    public static Half Float16FromBits(ushort value)
    {
#if NET5_0_OR_GREATER
        return BitConverter.UInt16BitsToHalf(value);
#else
        return Float16.FromEncoded(value);
#endif
    }

    public static ushort Float16ToBits(Half value)
    {
#if NET5_0_OR_GREATER
        return BitConverter.HalfToUInt16Bits(value);
#else
        return value.Value;
#endif
    }
}
