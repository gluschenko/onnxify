using Onnxify.Helpers;

namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents the ONNX 8-bit <c>float8e5m2</c> tensor element format.
/// </summary>
/// <remarks>
/// Compared with e4m3 formats, e5m2 keeps more exponent range and less mantissa precision. It can represent infinities and NaN encodings.
/// </remarks>
public readonly struct Float8E5M2
{
    /// <summary>
    /// Gets the encoded 8-bit float payload.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Quantizes a single-precision value to e5m2.
    /// </summary>
    /// <param name="value">Value to encode.</param>
    public Float8E5M2(float value)
    {
        Value = Encode(value);
    }

    /// <summary>
    /// Expands the encoded e5m2 value to a single-precision approximation.
    /// </summary>
    /// <returns>The decoded <see cref="float"/> value.</returns>
    public float ToSingle() => Decode(Value);

    private static byte Encode(float f)
    {
        if (float.IsNaN(f)) return 0xFF;
        if (float.IsInfinity(f)) return (byte)(f > 0 ? 0x7C : 0xFC);

        int sign = f < 0 ? 1 : 0;
        f = MathF.Abs(f);

        if (f == 0) return (byte)(sign << 7);

        int exp;
        float mant = MathHelper.Frexp(f, out exp);

        exp -= 1;
        int e = exp + 15; // bias=15

        if (e <= 0) return (byte)(sign << 7);
        if (e >= 31) return (byte)((sign << 7) | (0x1F << 2));

        int m = (int)((mant * 2 - 1) * 4); // 2 bits

        return (byte)((sign << 7) | (e << 2) | (m & 0x3));
    }

    private static float Decode(byte v)
    {
        int sign = (v >> 7) & 1;
        int exp = (v >> 2) & 0x1F;
        int mant = v & 0x3;

        if (exp == 0 && mant == 0) return sign == 1 ? -0f : 0f;
        if (exp == 0x1F)
        {
            if (mant == 0) return sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;
            return float.NaN;
        }

        float m = 1f + mant / 4f;
        int e = exp - 15;

        float result = MathF.Pow(2, e) * m;
        return sign == 1 ? -result : result;
    }
}
