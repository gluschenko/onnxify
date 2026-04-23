namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents one signed 4-bit ONNX integer tensor element.
/// </summary>
/// <remarks>
/// Values are stored in two's-complement form in the low nibble of <see cref="Value"/>. ONNX raw tensor data packs two elements per byte; Onnxify exposes each unpacked element as a wrapper value.
/// </remarks>
public readonly struct Int4
{
    /// <summary>
    /// Gets the encoded four-bit payload in the low nibble.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Encodes a signed value by keeping its low four bits.
    /// </summary>
    /// <param name="value">Value to encode, normally in the range -8 through 7.</param>
    public Int4(sbyte value)
    {
        Value = (byte)(value & 0xF);
    }

    /// <summary>
    /// Decodes the stored four-bit payload as a signed value.
    /// </summary>
    /// <returns>A value in the range -8 through 7.</returns>
    public sbyte ToSByte()
    {
        int v = Value & 0xF;
        if (v >= 8) v -= 16;
        return (sbyte)v;
    }
}
