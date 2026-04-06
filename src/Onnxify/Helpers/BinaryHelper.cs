using System.Runtime.InteropServices;
using Google.Protobuf;
using Onnxify.Data.Numerics;

namespace Onnxify.Helpers;

public static class BinaryHelper
{
    public static T[] Decode<T>(ReadOnlySpan<byte> span) where T : struct
    {
        if (typeof(T) == typeof(byte))
        {
            return (T[])(object)span.ToArray();
        }

        return MemoryMarshal.Cast<byte, T>(span).ToArray();
    }

    public static ByteString Encode<T>(T[] data) where T : struct
    {
        if (data is byte[])
        {
            return ByteString.CopyFrom((byte[])(object)data);
        }

        var span = MemoryMarshal.AsBytes(data.AsSpan());
        return ByteString.CopyFrom(span.ToArray());
    }

    public static bool[] DecodeBoolArray(ReadOnlySpan<byte> span)
    {
        return [.. span.ToArray().Select(x => x != 0)];
    }

    public static ByteString EncodeBoolArray(bool[] data)
    {
        return ByteString.CopyFrom([.. data.Select(x => (byte)(x ? 1 : 0))]);
    }

    internal static Float8E4M3FN[] DecodeFloat8E4M3FN(ReadOnlySpan<byte> span)
    {
        var result = new Float8E4M3FN[span.Length];
        for (var i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E4M3FN(span[i]);
        }

        return result;
    }

    internal static Float8E4M3FNUZ[] DecodeFloat8E4M3FNUZ(ReadOnlySpan<byte> span)
    {
        var result = new Float8E4M3FNUZ[span.Length];
        for (var i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E4M3FNUZ(span[i]);
        }

        return result;
    }

    internal static Float8E5M2[] DecodeFloat8E5M2(ReadOnlySpan<byte> span)
    {
        var result = new Float8E5M2[span.Length];
        for (var i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E5M2(span[i]);
        }

        return result;
    }

    internal static Float8E5M2FNUZ[] DecodeFloat8E5M2FNUZ(ReadOnlySpan<byte> span)
    {
        var result = new Float8E5M2FNUZ[span.Length];
        for (var i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E5M2FNUZ(span[i]);
        }

        return result;
    }

    internal static Float8E8M0[] DecodeFloat8E8M0(ReadOnlySpan<byte> span)
    {
        var result = new Float8E8M0[span.Length];
        for (var i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E8M0(span[i]);
        }

        return result;
    }

    internal static Float4E2M1[] DecodeFloat4(ReadOnlySpan<byte> span)
    {
        var result = new Float4E2M1[span.Length * 2];

        var j = 0;
        for (var i = 0; i < span.Length; i++)
        {
            var b = span[i];

            result[j++] = new Float4E2M1((byte)(b & 0x0F));
            result[j++] = new Float4E2M1((byte)(b >> 4));
        }

        return result;
    }

    internal static UInt4[] DecodeUInt4(ReadOnlySpan<byte> span)
    {
        var result = new UInt4[span.Length * 2];

        var j = 0;
        for (var i = 0; i < span.Length; i++)
        {
            var b = span[i];

            result[j++] = new UInt4((byte)(b & 0x0F));
            result[j++] = new UInt4((byte)(b >> 4));
        }

        return result;
    }

    internal static Int4[] DecodeInt4(ReadOnlySpan<byte> span)
    {
        var result = new Int4[span.Length * 2];

        var j = 0;
        for (var i = 0; i < span.Length; i++)
        {
            var b = span[i];

            result[j++] = new Int4((sbyte)(b & 0x0F));
            result[j++] = new Int4((sbyte)(b >> 4));
        }

        return result;
    }

    internal static UInt2[] DecodeUInt2(ReadOnlySpan<byte> span)
    {
        var result = new UInt2[span.Length * 4];

        var j = 0;
        for (var i = 0; i < span.Length; i++)
        {
            var b = span[i];

            result[j++] = new UInt2((byte)(b & 0x03));
            result[j++] = new UInt2((byte)((b >> 2) & 0x03));
            result[j++] = new UInt2((byte)((b >> 4) & 0x03));
            result[j++] = new UInt2((byte)((b >> 6) & 0x03));
        }

        return result;
    }

    internal static Int2[] DecodeInt2(ReadOnlySpan<byte> span)
    {
        var result = new Int2[span.Length * 4];

        var j = 0;
        for (var i = 0; i < span.Length; i++)
        {
            var b = span[i];

            result[j++] = new Int2((sbyte)(b & 0x03));
            result[j++] = new Int2((sbyte)((b >> 2) & 0x03));
            result[j++] = new Int2((sbyte)((b >> 4) & 0x03));
            result[j++] = new Int2((sbyte)((b >> 6) & 0x03));
        }

        return result;
    }

    internal static BFloat16[] DecodeBFloat16(ReadOnlySpan<byte> data)
    {
        var ushortSpan = MemoryMarshal.Cast<byte, ushort>(data);
        var result = new BFloat16[ushortSpan.Length];

        for (var i = 0; i < ushortSpan.Length; i++)
        {
            var value = (uint)ushortSpan[i] << 16;
            result[i] = new BFloat16(BitConverter.Int32BitsToSingle((int)value));
        }

        return result;
    }

    internal static ByteString EncodeBFloat16(BFloat16[] data)
    {
        var buffer = new byte[data.Length * 2];

        for (var i = 0; i < data.Length; i++)
        {
            var bits = (uint)BitConverter.SingleToInt32Bits(data[i].ToSingle());
            var bf = (ushort)(bits >> 16);

            buffer[i * 2] = (byte)(bf & 0xFF);
            buffer[i * 2 + 1] = (byte)(bf >> 8);
        }

        return ByteString.CopyFrom(buffer);
    }

    internal static Half[] DecodeHalf(ReadOnlySpan<byte> data)
    {
        var ushortSpan = MemoryMarshal.Cast<byte, ushort>(data);
        var result = new Half[ushortSpan.Length];

        for (var i = 0; i < ushortSpan.Length; i++)
        {
            result[i] = BitConverter.UInt16BitsToHalf(ushortSpan[i]);
        }

        return result;
    }

    internal static ByteString EncodeHalf(Half[] data)
    {
        var buffer = new byte[data.Length * 2];

        for (var i = 0; i < data.Length; i++)
        {
            var bits = BitConverter.HalfToUInt16Bits(data[i]);
            buffer[i * 2] = (byte)(bits & 0xFF);
            buffer[i * 2 + 1] = (byte)(bits >> 8);
        }

        return ByteString.CopyFrom(buffer);
    }

    internal static Complex64[] DecodeComplex64(ReadOnlySpan<byte> data)
    {
        var floatSpan = MemoryMarshal.Cast<byte, float>(data);
        var result = new Complex64[floatSpan.Length / 2];

        for (var i = 0; i < result.Length; i++)
        {
            result[i] = new Complex64(floatSpan[i * 2], floatSpan[i * 2 + 1]);
        }

        return result;
    }

    internal static ByteString EncodeComplex64(Complex64[] data)
    {
        var buffer = new float[data.Length * 2];

        for (var i = 0; i < data.Length; i++)
        {
            buffer[i * 2] = data[i].Real;
            buffer[i * 2 + 1] = data[i].Imaginary;
        }

        return BinaryHelper.Encode(buffer);
    }

    internal static Complex128[] DecodeComplex128(ReadOnlySpan<byte> data)
    {
        var doubleSpan = MemoryMarshal.Cast<byte, double>(data);
        var result = new Complex128[doubleSpan.Length / 2];

        for (var i = 0; i < result.Length; i++)
        {
            result[i] = new Complex128(
                doubleSpan[i * 2],
                doubleSpan[i * 2 + 1]
            );
        }

        return result;
    }

    internal static ByteString EncodeComplex128(Complex128[] data)
    {
        var buffer = new double[data.Length * 2];

        for (var i = 0; i < data.Length; i++)
        {
            buffer[i * 2] = data[i].Real;
            buffer[i * 2 + 1] = data[i].Imaginary;
        }

        return BinaryHelper.Encode(buffer);
    }
}
