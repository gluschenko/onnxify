using Onnxify.Helpers;

namespace Onnxify.Data.Numerics;

public readonly struct Float4E2M1
{
    public byte Value { get; }

    public Float4E2M1(float value)
    {
        Value = Encode(value);
    }

    public float ToSingle() => Decode(Value);

    private static byte Encode(float f)
    {
        int sign = f < 0 ? 1 : 0;
        f = MathF.Abs(f);

        if (f == 0) return (byte)(sign << 3);

        int exp;
        float mant = MathHelper.Frexp(f, out exp);

        exp -= 1;
        int e = exp + 1; // bias=1

        if (e <= 0) return (byte)(sign << 3);
        if (e >= 3) return (byte)((sign << 3) | (0x3 << 1));

        int m = (int)((mant * 2 - 1) * 2); // 1 бит

        return (byte)((sign << 3) | (e << 1) | (m & 0x1));
    }

    private static float Decode(byte v)
    {
        int sign = (v >> 3) & 1;
        int exp = (v >> 1) & 0x3;
        int mant = v & 0x1;

        if (exp == 0 && mant == 0) return sign == 1 ? -0f : 0f;

        float m = 1f + mant / 2f;
        int e = exp - 1;

        float result = MathF.Pow(2, e) * m;
        return sign == 1 ? -result : result;
    }
}

