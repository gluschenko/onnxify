namespace Onnxify.Data;

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

