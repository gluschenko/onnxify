using Onnxify.Helpers;

namespace Onnxify.Data.Numerics;

public readonly struct Float8E5M2FNUZ
{
    public byte Value { get; }

    public Float8E5M2FNUZ(byte value)
    {
        Value = value;
    }

    public Float8E5M2FNUZ(float value)
    {
        Value = Encode(value);
    }

    public float ToSingle()
    {
        return Decode(Value);
    }

    public static implicit operator byte(Float8E5M2FNUZ value) => value.Value;

    public static implicit operator Float8E5M2FNUZ(byte value) => new(value);

    private static byte Encode(float value)
    {
        if (float.IsNaN(value))
            return 0x80; // canonical NaN

        // FNUZ: no negative zero
        if (value == 0f)
            return 0x00;

        bool negative = value < 0;
        float abs = MathF.Abs(value);

        float mantissa = MathHelper.Frexp(abs, out int exponent);
        // frexp: abs = mantissa * 2^exponent, 0.5 <= mantissa < 1

        // Переводим к виду 1.x * 2^e
        exponent -= 1;

        // Для FNUZ bias = 16
        int encodedExponent = exponent + 16;

        // Subnormal / underflow
        if (encodedExponent <= 0)
            return 0x00;

        // FNUZ: no infinity, overflow -> max finite
        if (encodedExponent >= 0x1F)
            return (byte)((negative ? 0x80 : 0x00) | 0x7F);

        float fractional = mantissa * 2f - 1f;   // [0, 1)
        int encodedMantissa = (int)MathF.Round(fractional * 4f, MidpointRounding.ToEven);

        // округление могло переполнить мантиссу
        if (encodedMantissa == 4)
        {
            encodedMantissa = 0;
            encodedExponent++;

            if (encodedExponent >= 0x1F)
                return (byte)((negative ? 0x80 : 0x00) | 0x7F);
        }

        return (byte)(
            (negative ? 0x80 : 0x00) |
            ((encodedExponent & 0x1F) << 2) |
            (encodedMantissa & 0x03));
    }

    private static float Decode(byte value)
    {
        // FNUZ: 0x80 обычно трактуется как NaN
        if (value == 0x80)
            return float.NaN;

        bool negative = (value & 0x80) != 0;
        int exponentBits = (value >> 2) & 0x1F;
        int mantissaBits = value & 0x03;

        if (exponentBits == 0 && mantissaBits == 0)
            return 0f; // no negative zero in FNUZ

        // Для FNUZ bias = 16
        int exponent = exponentBits - 16;
        float significand = 1f + mantissaBits / 4f;

        float result = significand * MathF.Pow(2f, exponent);
        return negative ? -result : result;
    }

    public override string ToString()
    {
        return ToSingle().ToString();
    }
}

