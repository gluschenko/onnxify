namespace Onnxify.Data;

public readonly struct Float8E4M3FNUZ
{
    public byte Value { get; }

    public Float8E4M3FNUZ(float value)
    {
        if (value == 0) value = 0; // убираем -0
        Value = new Float8E4M3FN(value).Value;
    }

    public float ToSingle()
    {
        var f = new Float8E4M3FN { }.ToSingle(); // не критично, можно inline
        return f == 0 ? 0 : f;
    }
}

