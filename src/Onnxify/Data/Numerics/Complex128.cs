namespace Onnxify.Data.Numerics;

public readonly struct Complex128
{
    public double Real { get; }
    public double Imaginary { get; }

    public Complex128(double real, double imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }
}

