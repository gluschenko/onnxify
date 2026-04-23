namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents one signed 2-bit ONNX integer tensor element.
/// </summary>
/// <remarks>
/// Values are stored in two's-complement form in the low two bits of <see cref="Value"/>. ONNX raw tensor data packs four elements per byte; Onnxify exposes each unpacked element as a wrapper value.
/// </remarks>
public readonly struct Int2
{
    /// <summary>
    /// Gets the encoded two-bit payload in the low bits.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Encodes a signed value by keeping its low two bits.
    /// </summary>
    /// <param name="value">Value to encode, normally in the range -2 through 1.</param>
    public Int2(sbyte value)
    {
        Value = (byte)(value & 0x3);
    }

    /// <summary>
    /// Decodes the stored two-bit payload as a signed value.
    /// </summary>
    /// <returns>A value in the range -2 through 1.</returns>
    public sbyte ToSByte()
    {
        int v = Value & 0x3;
        if (v >= 2) v -= 4;
        return (sbyte)v;
    }
}
