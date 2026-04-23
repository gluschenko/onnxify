using Onnxify.Helpers;

namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents the ONNX 4-bit <c>float4e2m1</c> tensor element format.
/// </summary>
/// <remarks>
/// Values are stored in the low four bits of <see cref="Value"/>. ONNX raw tensor data packs two elements per byte; Onnxify unpacks them into one wrapper value per tensor element.
/// </remarks>
public readonly struct Float4E2M1
{
    /// <summary>
    /// Gets the encoded 4-bit payload in the low nibble.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Quantizes a single-precision value to the e2m1 format.
    /// </summary>
    /// <param name="value">Value to encode.</param>
    public Float4E2M1(float value)
    {
        Value = Encode(value);
    }

    /// <summary>
    /// Expands the encoded e2m1 value to a single-precision approximation.
    /// </summary>
    /// <returns>The decoded <see cref="float"/> value.</returns>
    public float ToSingle() => Decode(Value);

    private static byte Encode(float f)
    {
        int sign = f < 0 ? 1 : 0;
        f = MathF.Abs(f);

        if (f == 0) return (byte)(sign << 3);

        int exp;
        float mant = MathHelper.Frexp(f, out exp);

        exp -= 1;
        int e = exp + 1; // bias=1

        if (e <= 0) return (byte)(sign << 3);
        if (e >= 3) return (byte)((sign << 3) | (0x3 << 1));

        int m = (int)((mant * 2 - 1) * 2); // 1 bit

        return (byte)((sign << 3) | (e << 1) | (m & 0x1));
    }

    private static float Decode(byte v)
    {
        int sign = (v >> 3) & 1;
        int exp = (v >> 1) & 0x3;
        int mant = v & 0x1;

        if (exp == 0 && mant == 0) return sign == 1 ? -0f : 0f;

        float m = 1f + mant / 2f;
        int e = exp - 1;

        float result = MathF.Pow(2, e) * m;
        return sign == 1 ? -result : result;
    }
}
