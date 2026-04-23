using Onnxify.Helpers;

namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents the ONNX 8-bit <c>float8e5m2fnuz</c> tensor element format.
/// </summary>
/// <remarks>
/// FNUZ semantics remove negative zero and use a finite-only range with a canonical NaN encoding. The wrapper exposes the encoded byte so decoded tensors can be round-tripped or inspected without widening every value.
/// </remarks>
public readonly struct Float8E5M2FNUZ
{
    /// <summary>
    /// Gets the encoded 8-bit float payload.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Creates a value from an already encoded ONNX byte.
    /// </summary>
    /// <param name="value">Encoded e5m2fnuz payload.</param>
    public Float8E5M2FNUZ(byte value)
    {
        Value = value;
    }

    /// <summary>
    /// Quantizes a single-precision value to e5m2fnuz.
    /// </summary>
    /// <param name="value">Value to encode.</param>
    public Float8E5M2FNUZ(float value)
    {
        Value = Encode(value);
    }

    /// <summary>
    /// Expands the encoded e5m2fnuz value to a single-precision approximation.
    /// </summary>
    /// <returns>The decoded <see cref="float"/> value.</returns>
    public float ToSingle()
    {
        return Decode(Value);
    }

    /// <summary>
    /// Extracts the encoded byte payload.
    /// </summary>
    /// <param name="value">Encoded float8 wrapper.</param>
    public static implicit operator byte(Float8E5M2FNUZ value) => value.Value;

    /// <summary>
    /// Wraps an already encoded e5m2fnuz byte.
    /// </summary>
    /// <param name="value">Encoded byte payload.</param>
    public static implicit operator Float8E5M2FNUZ(byte value) => new(value);

    private static byte Encode(float value)
    {
        if (float.IsNaN(value))
            return 0x80; // canonical NaN

        if (value == 0f)
            return 0x00;

        bool negative = value < 0;
        float abs = MathF.Abs(value);

        float mantissa = MathHelper.Frexp(abs, out int exponent);

        exponent -= 1;

        int encodedExponent = exponent + 16;

        if (encodedExponent <= 0)
            return 0x00;

        if (encodedExponent >= 0x1F)
            return (byte)((negative ? 0x80 : 0x00) | 0x7F);

        float fractional = mantissa * 2f - 1f;
        int encodedMantissa = (int)MathF.Round(fractional * 4f, MidpointRounding.ToEven);

        if (encodedMantissa == 4)
        {
            encodedMantissa = 0;
            encodedExponent++;

            if (encodedExponent >= 0x1F)
                return (byte)((negative ? 0x80 : 0x00) | 0x7F);
        }

        return (byte)(
            (negative ? 0x80 : 0x00) |
            ((encodedExponent & 0x1F) << 2) |
            (encodedMantissa & 0x03));
    }

    private static float Decode(byte value)
    {
        if (value == 0x80)
            return float.NaN;

        bool negative = (value & 0x80) != 0;
        int exponentBits = (value >> 2) & 0x1F;
        int mantissaBits = value & 0x03;

        if (exponentBits == 0 && mantissaBits == 0)
            return 0f;

        int exponent = exponentBits - 16;
        float significand = 1f + mantissaBits / 4f;

        float result = significand * MathF.Pow(2f, exponent);
        return negative ? -result : result;
    }

    /// <summary>
    /// Returns the decoded value using the current culture's numeric formatting.
    /// </summary>
    /// <returns>A diagnostic string for the decoded value.</returns>
    public override string ToString()
    {
        return ToSingle().ToString();
    }
}
