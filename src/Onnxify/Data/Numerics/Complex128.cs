namespace Onnxify.Data.Numerics;

/// <summary>
/// Represents an ONNX <c>complex128</c> tensor element as double-precision real and imaginary components.
/// </summary>
/// <remarks>
/// ONNX raw data stores complex numbers as interleaved real/imaginary values. This wrapper keeps that pairing explicit when tensors are decoded into CLR objects.
/// </remarks>
public readonly struct Complex128
{
    /// <summary>
    /// Gets the real component.
    /// </summary>
    public double Real { get; }

    /// <summary>
    /// Gets the imaginary component.
    /// </summary>
    public double Imaginary { get; }

    /// <summary>
    /// Creates a complex tensor element.
    /// </summary>
    /// <param name="real">Real component.</param>
    /// <param name="imaginary">Imaginary component.</param>
    public Complex128(double real, double imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }
}
