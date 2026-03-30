namespace Onnxify.Data.Numerics;

public readonly struct Float8E8M0
{
    public byte Value { get; }

    public Float8E8M0(float value)
    {
        if (value <= 0) Value = 0;
        else
        {
            int exp = (int)MathF.Round(MathF.Log2(value));
            Value = (byte)(exp + 127);
        }
    }

    public float ToSingle()
    {
        int exp = Value - 127;
        return MathF.Pow(2, exp);
    }
}

