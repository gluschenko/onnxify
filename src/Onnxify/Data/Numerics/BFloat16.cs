namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents the ONNX <c>bfloat16</c> tensor element format using the upper 16 bits of an IEEE 754 single-precision value.
/// </summary>
/// <remarks>
/// Converting from <see cref="float"/> truncates the lower mantissa bits; use <see cref="ToSingle"/> when you need a CLR value for inspection or preprocessing.
/// </remarks>
public readonly struct BFloat16
{
    /// <summary>
    /// Gets the encoded 16-bit bfloat payload.
    /// </summary>
    public ushort Value { get; }

    /// <summary>
    /// Encodes a single-precision value as bfloat16 by keeping the high-order sign, exponent, and mantissa bits.
    /// </summary>
    /// <param name="value">Value to encode.</param>
    public BFloat16(float value)
    {
        var bits = (uint)BitConverter.SingleToInt32Bits(value);
        Value = (ushort)(bits >> 16);
    }

    /// <summary>
    /// Expands the encoded bfloat16 payload to a single-precision value.
    /// </summary>
    /// <returns>The nearest representable <see cref="float"/> implied by the stored bfloat bits.</returns>
    public float ToSingle()
    {
        var bits = (uint)Value << 16;
        return BitConverter.Int32BitsToSingle((int)bits);
    }
}
