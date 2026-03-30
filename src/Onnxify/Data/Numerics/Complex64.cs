namespace Onnxify.Data.Numerics;

public readonly struct Complex64
{
    public float Real { get; }
    public float Imaginary { get; }

    public Complex64(float real, float imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }
}

