namespace Onnxify.Data;

public readonly struct Float8E4M3FN
{
    public byte Value { get; }

    public Float8E4M3FN(float value)
    {
        Value = Encode(value);
    }

    public float ToSingle() => Decode(Value);

    private static byte Encode(float f)
    {
        if (float.IsNaN(f)) return 0x7F;

        int sign = f < 0 ? 1 : 0;
        f = MathF.Abs(f);

        if (f == 0) return (byte)(sign << 7);

        int exp;
        float mant = MathHelper.Frexp(f, out exp); // mant ∈ [0.5,1)

        exp -= 1; // нормализация
        int e = exp + 7; // bias=7

        if (e <= 0) return (byte)(sign << 7); // underflow
        if (e >= 15) return (byte)((sign << 7) | (0xF << 3)); // saturate

        int m = (int)((mant * 2 - 1) * 8); // 3 бита

        return (byte)((sign << 7) | (e << 3) | (m & 0x7));
    }

    private static float Decode(byte v)
    {
        int sign = (v >> 7) & 1;
        int exp = (v >> 3) & 0xF;
        int mant = v & 0x7;

        if (exp == 0 && mant == 0) return sign == 1 ? -0f : 0f;
        if (exp == 0xF) return float.NaN;

        float m = 1f + mant / 8f;
        int e = exp - 7;

        float result = MathF.Pow(2, e) * m;
        return sign == 1 ? -result : result;
    }
}

