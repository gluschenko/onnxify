namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents one unsigned 4-bit ONNX integer tensor element.
/// </summary>
/// <remarks>
/// Values are stored in the low nibble of <see cref="Value"/>. ONNX raw tensor data packs two elements per byte; Onnxify exposes each unpacked element as a wrapper value.
/// </remarks>
public readonly struct UInt4
{
    /// <summary>
    /// Gets the encoded four-bit payload in the low nibble.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Encodes an unsigned value by keeping its low four bits.
    /// </summary>
    /// <param name="value">Value to encode, normally in the range 0 through 15.</param>
    public UInt4(byte value)
    {
        Value = (byte)(value & 0xF);
    }

    /// <summary>
    /// Decodes the stored payload as an unsigned value.
    /// </summary>
    /// <returns>A value in the range 0 through 15.</returns>
    public byte ToByte() => (byte)(Value & 0xF);
}
