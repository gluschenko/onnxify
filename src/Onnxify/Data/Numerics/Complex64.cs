namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents an ONNX <c>complex64</c> tensor element as single-precision real and imaginary components.
/// </summary>
/// <remarks>
/// ONNX raw data stores complex numbers as interleaved real/imaginary values. This wrapper keeps that pairing explicit when tensors are decoded into CLR objects.
/// </remarks>
public readonly struct Complex64
{
    /// <summary>
    /// Gets the real component.
    /// </summary>
    public float Real { get; }

    /// <summary>
    /// Gets the imaginary component.
    /// </summary>
    public float Imaginary { get; }

    /// <summary>
    /// Creates a complex tensor element.
    /// </summary>
    /// <param name="real">Real component.</param>
    /// <param name="imaginary">Imaginary component.</param>
    public Complex64(float real, float imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }
}
