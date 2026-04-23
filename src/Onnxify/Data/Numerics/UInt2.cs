namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents one unsigned 2-bit ONNX integer tensor element.
/// </summary>
/// <remarks>
/// Values are stored in the low two bits of <see cref="Value"/>. ONNX raw tensor data packs four elements per byte; Onnxify exposes each unpacked element as a wrapper value.
/// </remarks>
public readonly struct UInt2
{
    /// <summary>
    /// Gets the encoded two-bit payload in the low bits.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Encodes an unsigned value by keeping its low two bits.
    /// </summary>
    /// <param name="value">Value to encode, normally in the range 0 through 3.</param>
    public UInt2(byte value)
    {
        Value = (byte)(value & 0x3);
    }

    /// <summary>
    /// Decodes the stored payload as an unsigned value.
    /// </summary>
    /// <returns>A value in the range 0 through 3.</returns>
    public byte ToByte() => (byte)(Value & 0x3);
}
