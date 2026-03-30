namespace Onnxify.Data;

public static class MathHelper
{
    /// <summary>
    /// Bit-mask used for extracting the exponent bits of a <see cref="float"/> (<c>0x7f800000</c>).
    /// </summary>
    public const int FLT_EXP_MASK = 0x7f800000;

    /// <summary>
    /// The number of bits in the mantissa of a <see cref="float"/>, excludes the implicit leading <c>1</c> bit (<c>23</c>).
    /// </summary>
    public const int FLT_MANT_BITS = 23;

    /// <summary>
    /// Bit-mask used for extracting the sign bit of a <see cref="float"/> (<c>0x80000000</c>).
    /// </summary>
    public const int FLT_SGN_MASK = -1 - 0x7fffffff;

    /// <summary>
    /// Bit-mask used for extracting the mantissa bits of a <see cref="float"/> (<c>0x007fffff</c>).
    /// </summary>
    public const int FLT_MANT_MASK = 0x007fffff;

    /// <summary>
    /// Bit-mask used for clearing the exponent bits of a <see cref="float"/> (<c>0x807fffff</c>).
    /// </summary>
    public const int FLT_EXP_CLR_MASK = FLT_SGN_MASK | FLT_MANT_MASK;

    /// <summary>
    /// Decomposes the given floating-point <paramref name="number"/> into a normalized fraction and an integral power of two.
    /// </summary>
    /// <param name="number">A floating-point number.</param>
    /// <param name="exponent">Reference to an <see cref="int"/> value to store the exponent to.</param>
    /// <returns>A <c>fraction</c> in the range <c>[0.5, 1)</c> so that <c><paramref name="number"/> = fraction * 2^<paramref name="exponent"/></c>.</returns>
    /// <remarks>
    /// <para>
    /// Special values are treated as follows.
    /// </para>
    /// <list type="bullet" >
    /// <item>If <paramref name="number"/> is <c>±0</c>, it is returned, and <c>0</c> is returned in <paramref name="exponent"/>.</item>
    /// <item>If <paramref name="number"/> is infinite, it is returned, and an undefined value is returned in <paramref name="exponent"/>.</item>
    /// <item>If <paramref name="number"/> is NaN, it is returned, and an undefined value is returned in <paramref name="exponent"/>.</item>
    /// </list>
    /// <para>
    /// See <a href="http://en.cppreference.com/w/c/numeric/math/frexp">frexp</a> in the C standard documentation.
    /// </para>
    /// </remarks>
    /// <para>
    /// Source: https://github.com/Quansight-Labs/numpy.net/blob/abb78faba57f1df3ed6a5c1a8af22bdacf6055e8/src/NumpyDotNet/NumpyDotNet/MathByMachineCognitus.cs#L647
    /// </para>
    public static float Frexp(float number, out int exponent)
    {
        var bits = BitConverter.SingleToInt32Bits(number);
        var exp = (int)((bits & FLT_EXP_MASK) >> FLT_MANT_BITS);
        exponent = 0;

        if (exp == 0xff || number == 0F)
        {
            number += number;
        }
        else
        {
            // Not zero and finite.
            exponent = exp - 126;
            if (exp == 0)
            {
                // Subnormal, scale number so that it is in [1, 2).
                number *= BitConverter.Int32BitsToSingle(0x4c000000); // 2^25
                bits = BitConverter.SingleToInt32Bits(number);
                exp = (int)((bits & FLT_EXP_MASK) >> FLT_MANT_BITS);
                exponent = exp - 126 - 25;
            }
            // Set exponent to -1 so that number is in [0.5, 1).
            number = BitConverter.Int32BitsToSingle((bits & FLT_EXP_CLR_MASK) | 0x3f000000);
        }

        return number;
    }
}

