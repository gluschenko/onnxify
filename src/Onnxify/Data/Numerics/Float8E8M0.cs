namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents the ONNX 8-bit <c>float8e8m0</c> tensor element format.
/// </summary>
/// <remarks>
/// This exponent-only format stores powers of two. Encoding rounds the base-2 logarithm of the input, so non-power-of-two values are intentionally approximate.
/// </remarks>
public readonly struct Float8E8M0
{
    /// <summary>
    /// Gets the encoded exponent payload.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Encodes a positive single-precision value as an exponent-only float8 value.
    /// </summary>
    /// <param name="value">Value to encode. Non-positive values encode to zero.</param>
    public Float8E8M0(float value)
    {
        if (value <= 0) Value = 0;
        else
        {
            int exp = (int)MathF.Round(MathF.Log2(value));
            Value = (byte)(exp + 127);
        }
    }

    /// <summary>
    /// Expands the encoded exponent to a single-precision power-of-two value.
    /// </summary>
    /// <returns>The decoded <see cref="float"/> value.</returns>
    public float ToSingle()
    {
        int exp = Value - 127;
        return MathF.Pow(2, exp);
    }
}
