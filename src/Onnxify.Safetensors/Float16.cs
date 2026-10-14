using System.Runtime.InteropServices;

namespace Onnxify.Safetensors;

/// <summary>Represents an IEEE 754 binary16 value using a portable encoded payload.</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct Float16 : IEquatable<Float16>, IFormattable
{
    private readonly ushort _value;

    private Float16(ushort value) => _value = value;

    /// <summary>Gets the encoded 16-bit payload.</summary>
    public ushort Value => _value;

    /// <summary>Creates a value from its encoded IEEE 754 binary16 payload.</summary>
    public static Float16 FromEncoded(ushort value) => new(value);

    /// <summary>Converts a single-precision value to binary16 using round-to-nearest, ties-to-even.</summary>
    public static Float16 FromSingle(float value)
    {
#if NET5_0_OR_GREATER
        return new Float16(BitConverter.HalfToUInt16Bits((Half)value));
#else
        var bits = unchecked((uint)Float32Bits.ToInt32(value));
        var sign = (ushort)((bits >> 16) & 0x8000);
        var exponent = (int)((bits >> 23) & 0xff) - 127 + 15;
        var mantissa = bits & 0x7fffff;

        if (((bits >> 23) & 0xff) == 0xff)
        {
            return new Float16((ushort)(sign | (mantissa == 0 ? 0x7c00 : 0x7e00)));
        }

        if (exponent <= 0)
        {
            if (exponent < -10)
            {
                return new Float16(sign);
            }

            mantissa |= 0x800000;
            var shift = 14 - exponent;
            var rounded = mantissa >> shift;
            var remainder = mantissa & ((1u << shift) - 1);
            var halfway = 1u << (shift - 1);
            if (remainder > halfway || remainder == halfway && (rounded & 1) != 0)
            {
                rounded++;
            }

            return new Float16((ushort)((uint)sign | rounded));
        }

        if (exponent >= 31)
        {
            return new Float16((ushort)(sign | 0x7c00));
        }

        var halfMantissa = mantissa >> 13;
        var halfRemainder = mantissa & 0x1fff;
        if (halfRemainder > 0x1000 || halfRemainder == 0x1000 && (halfMantissa & 1) != 0)
        {
            halfMantissa++;
            if (halfMantissa == 0x400)
            {
                halfMantissa = 0;
                exponent++;
                if (exponent >= 31)
                {
                    return new Float16((ushort)(sign | 0x7c00));
                }
            }
        }

        return new Float16((ushort)((uint)sign | (uint)(exponent << 10) | halfMantissa));
#endif
    }

    /// <summary>Converts this value to single precision.</summary>
    public float ToSingle()
    {
#if NET5_0_OR_GREATER
        return (float)BitConverter.UInt16BitsToHalf(_value);
#else
        var sign = (uint)(_value & 0x8000) << 16;
        var exponent = (_value >> 10) & 0x1f;
        var mantissa = (uint)(_value & 0x03ff);
        uint bits;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                var adjustedExponent = -14;
                while ((mantissa & 0x400) == 0)
                {
                    mantissa <<= 1;
                    adjustedExponent--;
                }

                mantissa &= 0x3ff;
                bits = sign | (uint)(adjustedExponent + 127) << 23 | mantissa << 13;
            }
        }
        else if (exponent == 0x1f)
        {
            bits = sign | 0x7f800000 | mantissa << 13;
        }
        else
        {
            bits = sign | (uint)(exponent - 15 + 127) << 23 | mantissa << 13;
        }

        return Float32Bits.ToSingle(unchecked((int)bits));
#endif
    }

    public bool Equals(Float16 other) => _value == other._value;

    public override bool Equals(object? obj) => obj is Float16 other && Equals(other);

    public override int GetHashCode() => _value.GetHashCode();

    public override string ToString() => ToSingle().ToString(System.Globalization.CultureInfo.InvariantCulture);

    public string ToString(string? format, IFormatProvider? formatProvider) => ToSingle().ToString(format, formatProvider);

}

internal static class Float32Bits
{
    [StructLayout(LayoutKind.Explicit)]
    private struct SingleIntegerUnion
    {
        [FieldOffset(0)] public int Integer;
        [FieldOffset(0)] public float Single;
    }

    public static int ToInt32(float value) => new SingleIntegerUnion { Single = value }.Integer;

    public static float ToSingle(int bits) => new SingleIntegerUnion { Integer = bits }.Single;
}
