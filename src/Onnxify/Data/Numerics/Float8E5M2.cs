using Onnxify.Data;

namespace Onnxify.Data.Numerics;

public readonly struct Float8E5M2
{
    public byte Value { get; }

    public Float8E5M2(float value)
    {
        Value = Encode(value);
    }

    public float ToSingle() => Decode(Value);

    private static byte Encode(float f)
    {
        if (float.IsNaN(f)) return 0xFF;
        if (float.IsInfinity(f)) return (byte)(f > 0 ? 0x7C : 0xFC);

        int sign = f < 0 ? 1 : 0;
        f = MathF.Abs(f);

        if (f == 0) return (byte)(sign << 7);

        int exp;
        float mant = MathHelper.Frexp(f, out exp);

        exp -= 1;
        int e = exp + 15; // bias=15

        if (e <= 0) return (byte)(sign << 7);
        if (e >= 31) return (byte)((sign << 7) | (0x1F << 2)); // inf

        int m = (int)((mant * 2 - 1) * 4); // 2 бита

        return (byte)((sign << 7) | (e << 2) | (m & 0x3));
    }

    private static float Decode(byte v)
    {
        int sign = (v >> 7) & 1;
        int exp = (v >> 2) & 0x1F;
        int mant = v & 0x3;

        if (exp == 0 && mant == 0) return sign == 1 ? -0f : 0f;
        if (exp == 0x1F)
        {
            if (mant == 0) return sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;
            return float.NaN;
        }

        float m = 1f + mant / 4f;
        int e = exp - 15;

        float result = MathF.Pow(2, e) * m;
        return sign == 1 ? -result : result;
    }
}

