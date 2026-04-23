namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents the ONNX 8-bit <c>float8e4m3fnuz</c> tensor element format.
/// </summary>
/// <remarks>
/// FNUZ formats do not preserve negative zero. This wrapper is intended for preserving ONNX tensor element identity and for lightweight conversion at model boundaries.
/// </remarks>
public readonly struct Float8E4M3FNUZ
{
    /// <summary>
    /// Gets the encoded 8-bit float payload.
    /// </summary>
    public byte Value { get; }

    /// <summary>
    /// Quantizes a single-precision value to e4m3fnuz, normalizing negative zero to positive zero.
    /// </summary>
    /// <param name="value">Value to encode.</param>
    public Float8E4M3FNUZ(float value)
    {
        if (value == 0) value = 0; // removes -0
        Value = new Float8E4M3FN(value).Value;
    }

    /// <summary>
    /// Expands the stored value to the closest single-precision representation supported by this wrapper.
    /// </summary>
    /// <returns>The decoded <see cref="float"/> value with negative zero normalized away.</returns>
    public float ToSingle()
    {
        var f = new Float8E4M3FN { }.ToSingle();
        return f == 0 ? 0 : f;
    }
}
