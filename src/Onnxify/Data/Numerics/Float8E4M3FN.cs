using Onnxify.Helpers;

namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents the ONNX 8-bit <c>float8e4m3fn</c> tensor element format.
/// </summary>
/// <remarks>
/// This format trades range and precision for compact storage and uses finite-number semantics for overflow. Use it when preserving or authoring ONNX tensors that explicitly use this element type rather than as a general-purpose arithmetic type.
/// </remarks>
public readonly struct Float8E4M3FN
{
    /// <summary>
    /// Gets the encoded 8-bit float payload.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Quantizes a single-precision value to e4m3fn.
    /// </summary>
    /// <param name="value">Value to encode.</param>
    public Float8E4M3FN(float value)
    {
        Value = Encode(value);
    }

    /// <summary>
    /// Expands the encoded e4m3fn value to a single-precision approximation.
    /// </summary>
    /// <returns>The decoded <see cref="float"/> value.</returns>
    public float ToSingle() => Decode(Value);

    private static byte Encode(float f)
    {
        if (float.IsNaN(f)) return 0x7F;

        int sign = f < 0 ? 1 : 0;
        f = MathF.Abs(f);

        if (f == 0) return (byte)(sign << 7);

        int exp;
        float mant = MathHelper.Frexp(f, out exp);

        exp -= 1;
        int e = exp + 7; // bias=7

        if (e <= 0) return (byte)(sign << 7);
        if (e >= 15) return (byte)((sign << 7) | (0xF << 3));

        int m = (int)((mant * 2 - 1) * 8); // 3 bits

        return (byte)((sign << 7) | (e << 3) | (m & 0x7));
    }

    private static float Decode(byte v)
    {
        int sign = (v >> 7) & 1;
        int exp = (v >> 3) & 0xF;
        int mant = v & 0x7;

        if (exp == 0 && mant == 0) return sign == 1 ? -0f : 0f;
        if (exp == 0xF) return float.NaN;

        float m = 1f + mant / 8f;
        int e = exp - 7;

        float result = MathF.Pow(2, e) * m;
        return sign == 1 ? -result : result;
    }
}
