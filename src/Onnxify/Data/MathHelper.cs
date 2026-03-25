namespace Onnxify.Data;

public static class MathHelper
{
    public static float Frexp(float value, out int exponent)
    {
        const int MantissaBits = 23;
        const int ExponentBias = 127;
        const int ExponentMask = 0x7F800000;
        const int MantissaMask = 0x007FFFFF;
        const int SignMask = unchecked((int)0x80000000);

        int bits = BitConverter.SingleToInt32Bits(value);
        int sign = bits & SignMask;
        int expBits = (bits & ExponentMask) >> MantissaBits;
        int mantBits = bits & MantissaMask;

        // NaN / ±Infinity
        if (expBits == 0xFF)
        {
            exponent = 0;
            return value;
        }

        // ±0
        if (expBits == 0 && mantBits == 0)
        {
            exponent = 0;
            return value;
        }

        // Subnormal
        if (expBits == 0)
        {
            // Нормализуем вручную
            exponent = -126;

            while ((mantBits & (1 << MantissaBits)) == 0)
            {
                mantBits <<= 1;
                exponent--;
            }

            // убираем ведущую 1
            mantBits &= MantissaMask;

            // Для frexp нужно вернуть exponent так, чтобы mantissa была в [0.5, 1)
            exponent++;

            int resultBits = sign | ((ExponentBias - 1) << MantissaBits) | mantBits;
            return BitConverter.Int32BitsToSingle(resultBits);
        }

        // Normal
        exponent = expBits - ExponentBias + 1;

        int frexpBits = sign | ((ExponentBias - 1) << MantissaBits) | mantBits;
        return BitConverter.Int32BitsToSingle(frexpBits);
    }
}

