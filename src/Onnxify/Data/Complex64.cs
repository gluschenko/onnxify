namespace Onnxify.Data;

public readonly struct Complex64
{
    public double Real { get; }
    public double Imaginary { get; }

    public Complex64(double real, double imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }
}
