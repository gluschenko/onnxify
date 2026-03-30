using System.Runtime.InteropServices;
using Google.Protobuf;
using Google.Protobuf.Collections;
using Onnx;
using Onnxify.Data;
using Onnxify.Data.Numerics;

namespace Onnxify;

public static class OnnxHelper
{
    private static readonly (Type SystemType, TensorProto.Types.DataType DataType)[] _typePairs =
    [
        (typeof(float), TensorProto.Types.DataType.Float),
        (typeof(byte), TensorProto.Types.DataType.Uint8),
        (typeof(sbyte), TensorProto.Types.DataType.Int8),
        (typeof(ushort), TensorProto.Types.DataType.Uint16),
        (typeof(short), TensorProto.Types.DataType.Int16),
        (typeof(int), TensorProto.Types.DataType.Int32),
        (typeof(long), TensorProto.Types.DataType.Int64),
        (typeof(string), TensorProto.Types.DataType.String),
        (typeof(bool), TensorProto.Types.DataType.Bool),
        (typeof(Half), TensorProto.Types.DataType.Float16),
        (typeof(double), TensorProto.Types.DataType.Double),
        (typeof(uint), TensorProto.Types.DataType.Uint32),
        (typeof(ulong), TensorProto.Types.DataType.Uint64),
        (typeof(Complex64), TensorProto.Types.DataType.Complex64),
        (typeof(Complex128), TensorProto.Types.DataType.Complex128),
        (typeof(BFloat16), TensorProto.Types.DataType.Bfloat16),
        (typeof(Float8E4M3FN), TensorProto.Types.DataType.Float8E4M3Fn),
        (typeof(Float8E4M3FNUZ), TensorProto.Types.DataType.Float8E4M3Fnuz),
        (typeof(Float8E5M2), TensorProto.Types.DataType.Float8E5M2),
        (typeof(Float8E5M2FNUZ), TensorProto.Types.DataType.Float8E5M2Fnuz),
        (typeof(Float4E2M1), TensorProto.Types.DataType.Float4E2M1),
        (typeof(Float8E8M0), TensorProto.Types.DataType.Float8E8M0),
        (typeof(UInt4), TensorProto.Types.DataType.Uint4),
        (typeof(Int4), TensorProto.Types.DataType.Int4),
        (typeof(UInt2), TensorProto.Types.DataType.Uint2),
        (typeof(Int2), TensorProto.Types.DataType.Int2),
        (typeof(object), TensorProto.Types.DataType.Undefined),
    ];

    private static readonly Dictionary<Type, TensorProto.Types.DataType> _typeToDataType =
        _typePairs.ToDictionary(x => x.SystemType, x => x.DataType);

    private static readonly Dictionary<TensorProto.Types.DataType, Type> _dataTypeToType =
        _typePairs.ToDictionary(x => x.DataType, x => x.SystemType);

    internal static TensorProto.Types.DataType GetDataType(Type type)
    {
        if (_typeToDataType.TryGetValue(type, out var dataType))
        {
            return dataType;
        }

        throw new NotImplementedException($"Type '{type}' is not supported");
    }

    internal static Type GetSystemType(TensorProto.Types.DataType type)
    {
        if (_dataTypeToType.TryGetValue(type, out var systemType))
        {
            return systemType;
        }

        throw new NotImplementedException($"DataType '{type}' is not supported");
    }

    internal static OnnxTensor FromProto(TensorProto tensor, OnnxModelBaseOptions options)
    {
        var type = (TensorProto.Types.DataType)tensor.DataType;

        return type switch
        {
            TensorProto.Types.DataType.Float => OnnxTensor.FromProto<float>(tensor, options),
            TensorProto.Types.DataType.Uint8 => OnnxTensor.FromProto<byte>(tensor, options),
            TensorProto.Types.DataType.Int8 => OnnxTensor.FromProto<sbyte>(tensor, options),
            TensorProto.Types.DataType.Uint16 => OnnxTensor.FromProto<ushort>(tensor, options),
            TensorProto.Types.DataType.Int16 => OnnxTensor.FromProto<short>(tensor, options),
            TensorProto.Types.DataType.Int32 => OnnxTensor.FromProto<int>(tensor, options),
            TensorProto.Types.DataType.Int64 => OnnxTensor.FromProto<long>(tensor, options),
            TensorProto.Types.DataType.String => OnnxTensor.FromProto<string>(tensor, options),
            TensorProto.Types.DataType.Bool => OnnxTensor.FromProto<bool>(tensor, options),
            TensorProto.Types.DataType.Float16 => OnnxTensor.FromProto<Half>(tensor, options),
            TensorProto.Types.DataType.Double => OnnxTensor.FromProto<double>(tensor, options),
            TensorProto.Types.DataType.Uint32 => OnnxTensor.FromProto<uint>(tensor, options),
            TensorProto.Types.DataType.Uint64 => OnnxTensor.FromProto<ulong>(tensor, options),
            TensorProto.Types.DataType.Complex64 => OnnxTensor.FromProto<Complex64>(tensor, options),
            TensorProto.Types.DataType.Complex128 => OnnxTensor.FromProto<Complex128>(tensor, options),
            TensorProto.Types.DataType.Bfloat16 => OnnxTensor.FromProto<BFloat16>(tensor, options),
            TensorProto.Types.DataType.Float8E4M3Fn => OnnxTensor.FromProto<Float8E4M3FN>(tensor, options),
            TensorProto.Types.DataType.Float8E4M3Fnuz => OnnxTensor.FromProto<Float8E4M3FNUZ>(tensor, options),
            TensorProto.Types.DataType.Float8E5M2 => OnnxTensor.FromProto<Float8E5M2>(tensor, options),
            TensorProto.Types.DataType.Float8E5M2Fnuz => OnnxTensor.FromProto<Float8E5M2FNUZ>(tensor, options),
            TensorProto.Types.DataType.Float4E2M1 => OnnxTensor.FromProto<Float4E2M1>(tensor, options),
            TensorProto.Types.DataType.Float8E8M0 => OnnxTensor.FromProto<Float8E8M0>(tensor, options),
            TensorProto.Types.DataType.Uint4 => OnnxTensor.FromProto<UInt4>(tensor, options),
            TensorProto.Types.DataType.Int4 => OnnxTensor.FromProto<Int4>(tensor, options),
            TensorProto.Types.DataType.Uint2 => OnnxTensor.FromProto<UInt2>(tensor, options),
            TensorProto.Types.DataType.Int2 => OnnxTensor.FromProto<Int2>(tensor, options),
            TensorProto.Types.DataType.Undefined => OnnxTensor.FromProto<object>(tensor, options),
            _ => throw new NotImplementedException($"Not implemented for '{type}'"),
        };
    }

    internal static OnnxSparseTensor FromProto(SparseTensorProto tensor, OnnxModelBaseOptions options)
    {
        var type = (TensorProto.Types.DataType)tensor.Values.DataType;

        return type switch
        {
            TensorProto.Types.DataType.Float => new OnnxSparseTensor<float>(tensor, options),
            TensorProto.Types.DataType.Uint8 => new OnnxSparseTensor<byte>(tensor, options),
            TensorProto.Types.DataType.Int8 => new OnnxSparseTensor<sbyte>(tensor, options),
            TensorProto.Types.DataType.Uint16 => new OnnxSparseTensor<ushort>(tensor, options),
            TensorProto.Types.DataType.Int16 => new OnnxSparseTensor<short>(tensor, options),
            TensorProto.Types.DataType.Int32 => new OnnxSparseTensor<int>(tensor, options),
            TensorProto.Types.DataType.Int64 => new OnnxSparseTensor<long>(tensor, options),
            TensorProto.Types.DataType.String => new OnnxSparseTensor<string>(tensor, options),
            TensorProto.Types.DataType.Bool => new OnnxSparseTensor<bool>(tensor, options),
            TensorProto.Types.DataType.Float16 => new OnnxSparseTensor<Half>(tensor, options),
            TensorProto.Types.DataType.Double => new OnnxSparseTensor<double>(tensor, options),
            TensorProto.Types.DataType.Uint32 => new OnnxSparseTensor<uint>(tensor, options),
            TensorProto.Types.DataType.Uint64 => new OnnxSparseTensor<ulong>(tensor, options),
            TensorProto.Types.DataType.Complex64 => new OnnxSparseTensor<Complex64>(tensor, options),
            TensorProto.Types.DataType.Complex128 => new OnnxSparseTensor<Complex128>(tensor, options),
            TensorProto.Types.DataType.Bfloat16 => new OnnxSparseTensor<BFloat16>(tensor, options),
            TensorProto.Types.DataType.Float8E4M3Fn => new OnnxSparseTensor<Float8E4M3FN>(tensor, options),
            TensorProto.Types.DataType.Float8E4M3Fnuz => new OnnxSparseTensor<Float8E4M3FNUZ>(tensor, options),
            TensorProto.Types.DataType.Float8E5M2 => new OnnxSparseTensor<Float8E5M2>(tensor, options),
            TensorProto.Types.DataType.Float8E5M2Fnuz => new OnnxSparseTensor<Float8E5M2FNUZ>(tensor, options),
            TensorProto.Types.DataType.Float4E2M1 => new OnnxSparseTensor<Float4E2M1>(tensor, options),
            TensorProto.Types.DataType.Float8E8M0 => new OnnxSparseTensor<Float8E8M0>(tensor, options),
            TensorProto.Types.DataType.Uint4 => new OnnxSparseTensor<UInt4>(tensor, options),
            TensorProto.Types.DataType.Int4 => new OnnxSparseTensor<Int4>(tensor, options),
            TensorProto.Types.DataType.Uint2 => new OnnxSparseTensor<UInt2>(tensor, options),
            TensorProto.Types.DataType.Int2 => new OnnxSparseTensor<Int2>(tensor, options),
            TensorProto.Types.DataType.Undefined => new OnnxSparseTensor<object>(tensor, options),
            _ => throw new NotImplementedException($"Not implemented for '{type}'"),
        };
    }

    internal static object GetValue(this TensorProto tensor, OnnxModelBaseOptions options)
    {
        var type = (TensorProto.Types.DataType)tensor.DataType;

        // TODO: data reader abstraction
        if (tensor.DataLocation == TensorProto.Types.DataLocation.External)
        {
            if (!Directory.Exists(options.DataLocation))
            {
                throw new IOException($"Data location is not a directory: {options.DataLocation ?? "<null>"}");
            }

            var external = tensor.ExternalData.ToDictionary(x => x.Key, x => x.Value);

            if (!external.TryGetValue("location", out var location))
            {
                throw new InvalidOperationException("External tensor missing 'location'");
            }

            var offset = external.TryGetValue("offset", out var offsetString)
                ? long.Parse(offsetString)
                : 0;

            var length = external.TryGetValue("length", out var lengthString)
                ? long.Parse(lengthString)
                : -1;

            var path = Path.Combine(options.DataLocation ?? "", location);

            using var fs = File.OpenRead(path);
            fs.Seek(offset, SeekOrigin.Begin);

            if (length < 0)
            {
                length = fs.Length - offset;
            }

            var buffer = new byte[length];
            fs.ReadExactly(buffer);

            var data = ConvertRaw(buffer, type);

            return data;
        }

        if (tensor.RawData.Length > 0)
        {
            var span = tensor.RawData.Span;
            return ConvertRaw(span, type);
        }

        return type switch
        {
            TensorProto.Types.DataType.Float => tensor.FloatData.ToArray(),
            TensorProto.Types.DataType.Double => tensor.DoubleData.ToArray(),
            TensorProto.Types.DataType.Uint8 => tensor.Int32Data.Select(x => (byte)x).ToArray(),
            TensorProto.Types.DataType.Int8 => tensor.Int32Data.Select(x => (sbyte)x).ToArray(),
            TensorProto.Types.DataType.Int32 => tensor.Int32Data.ToArray(),
            TensorProto.Types.DataType.Int64 => tensor.Int64Data.ToArray(),
            TensorProto.Types.DataType.String => tensor.StringData.Select(x => x.ToStringUtf8()).ToArray(),
            _ => throw new NotImplementedException($"Unsupported non-raw tensor type {type}")
        };
    }

    internal static object ConvertRaw(
        ReadOnlySpan<byte> span,
        TensorProto.Types.DataType type
    )
    {
        return type switch
        {
            TensorProto.Types.DataType.Float => Unpack<float>(span),
            TensorProto.Types.DataType.Double => Unpack<double>(span),
            TensorProto.Types.DataType.Int32 => Unpack<int>(span),
            TensorProto.Types.DataType.Int64 => Unpack<long>(span),
            TensorProto.Types.DataType.Uint32 => Unpack<uint>(span),
            TensorProto.Types.DataType.Uint64 => Unpack<ulong>(span),
            TensorProto.Types.DataType.Int16 => Unpack<short>(span),
            TensorProto.Types.DataType.Uint16 => Unpack<ushort>(span),
            TensorProto.Types.DataType.Int8 => Unpack<sbyte>(span),
            TensorProto.Types.DataType.Uint8 => span.ToArray(),
            TensorProto.Types.DataType.Bool => span.ToArray().Select(x => x != 0).ToArray(),
            TensorProto.Types.DataType.Float16 => ConvertHalf(span),
            TensorProto.Types.DataType.Bfloat16 => ConvertBFloat16(span),
            TensorProto.Types.DataType.Complex64 => ConvertComplex64(span),
            TensorProto.Types.DataType.Complex128 => ConvertComplex128(span),
            TensorProto.Types.DataType.Float8E4M3Fn => ConvertFloat8E4M3FN(span),
            TensorProto.Types.DataType.Float8E4M3Fnuz => ConvertFloat8E4M3FNUZ(span),
            TensorProto.Types.DataType.Float8E5M2 => ConvertFloat8E5M2(span),
            TensorProto.Types.DataType.Float8E5M2Fnuz => ConvertFloat8E5M2FNUZ(span),
            TensorProto.Types.DataType.Float4E2M1 => ConvertFloat4(span),
            TensorProto.Types.DataType.Float8E8M0 => ConvertFloat8E8M0(span),
            TensorProto.Types.DataType.Uint4 => ConvertUInt4(span),
            TensorProto.Types.DataType.Int4 => ConvertInt4(span),
            TensorProto.Types.DataType.Uint2 => ConvertUInt2(span),
            TensorProto.Types.DataType.Int2 => ConvertInt2(span),
            _ => throw new NotImplementedException($"Unsupported raw tensor type {type}")
        };
    }

    internal static Float8E4M3FN[] ConvertFloat8E4M3FN(ReadOnlySpan<byte> span)
    {
        var result = new Float8E4M3FN[span.Length];
        for (int i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E4M3FN(span[i]);
        }

        return result;
    }

    internal static Float8E4M3FNUZ[] ConvertFloat8E4M3FNUZ(ReadOnlySpan<byte> span)
    {
        var result = new Float8E4M3FNUZ[span.Length];
        for (int i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E4M3FNUZ(span[i]);
        }

        return result;
    }

    internal static Float8E5M2[] ConvertFloat8E5M2(ReadOnlySpan<byte> span)
    {
        var result = new Float8E5M2[span.Length];
        for (int i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E5M2(span[i]);
        }

        return result;
    }

    internal static Float8E5M2FNUZ[] ConvertFloat8E5M2FNUZ(ReadOnlySpan<byte> span)
    {
        var result = new Float8E5M2FNUZ[span.Length];
        for (int i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E5M2FNUZ(span[i]);
        }

        return result;
    }

    internal static Float8E8M0[] ConvertFloat8E8M0(ReadOnlySpan<byte> span)
    {
        var result = new Float8E8M0[span.Length];
        for (int i = 0; i < span.Length; i++)
        {
            result[i] = new Float8E8M0(span[i]);
        }

        return result;
    }

    internal static Float4E2M1[] ConvertFloat4(ReadOnlySpan<byte> span)
    {
        var result = new Float4E2M1[span.Length * 2];

        int j = 0;
        for (int i = 0; i < span.Length; i++)
        {
            byte b = span[i];

            result[j++] = new Float4E2M1((byte)(b & 0x0F));       // low nibble
            result[j++] = new Float4E2M1((byte)(b >> 4));         // high nibble
        }

        return result;
    }

    internal static UInt4[] ConvertUInt4(ReadOnlySpan<byte> span)
    {
        var result = new UInt4[span.Length * 2];

        int j = 0;
        for (int i = 0; i < span.Length; i++)
        {
            byte b = span[i];

            result[j++] = new UInt4((byte)(b & 0x0F));
            result[j++] = new UInt4((byte)(b >> 4));
        }

        return result;
    }

    internal static Int4[] ConvertInt4(ReadOnlySpan<byte> span)
    {
        var result = new Int4[span.Length * 2];

        int j = 0;
        for (int i = 0; i < span.Length; i++)
        {
            byte b = span[i];

            result[j++] = new Int4((sbyte)(b & 0x0F));
            result[j++] = new Int4((sbyte)(b >> 4));
        }

        return result;
    }

    internal static UInt2[] ConvertUInt2(ReadOnlySpan<byte> span)
    {
        var result = new UInt2[span.Length * 4];

        int j = 0;
        for (int i = 0; i < span.Length; i++)
        {
            byte b = span[i];

            result[j++] = new UInt2((byte)(b & 0x03));
            result[j++] = new UInt2((byte)((b >> 2) & 0x03));
            result[j++] = new UInt2((byte)((b >> 4) & 0x03));
            result[j++] = new UInt2((byte)((b >> 6) & 0x03));
        }

        return result;
    }

    internal static Int2[] ConvertInt2(ReadOnlySpan<byte> span)
    {
        var result = new Int2[span.Length * 4];

        int j = 0;
        for (int i = 0; i < span.Length; i++)
        {
            byte b = span[i];

            result[j++] = new Int2((sbyte)(b & 0x03));
            result[j++] = new Int2((sbyte)((b >> 2) & 0x03));
            result[j++] = new Int2((sbyte)((b >> 4) & 0x03));
            result[j++] = new Int2((sbyte)((b >> 6) & 0x03));
        }

        return result;
    }

    private static T[] Unpack<T>(ReadOnlySpan<byte> span) where T : struct
    {
        return MemoryMarshal.Cast<byte, T>(span).ToArray();
    }

    private static ByteString Pack<T>(T[] data) where T : struct
    {
        var span = MemoryMarshal.AsBytes(data.AsSpan());
        return ByteString.CopyFrom(span.ToArray());
    }

    private static ByteString PackHalf(Half[] data)
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

    private static ByteString PackBFloat16(BFloat16[] data)
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

    private static ByteString PackComplex64(Complex64[] data)
    {
        var buffer = new float[data.Length * 2];

        for (var i = 0; i < data.Length; i++)
        {
            buffer[i * 2] = data[i].Real;
            buffer[i * 2 + 1] = data[i].Imaginary;
        }

        return Pack(buffer);
    }

    private static ByteString PackComplex128(Complex128[] data)
    {
        var buffer = new double[data.Length * 2];

        for (var i = 0; i < data.Length; i++)
        {
            buffer[i * 2] = data[i].Real;
            buffer[i * 2 + 1] = data[i].Imaginary;
        }

        return Pack(buffer);
    }

    internal static IEnumerable<T> GetValue<T>(this TensorProto tensor, OnnxModelBaseOptions options)
    {
        var value = GetValue(tensor, options);

        if (value is IEnumerable<T> typed)
        {
            return typed;
        }

        throw new InvalidCastException($"Tensor '{tensor.Name}' is {value.GetType().Name}, not {typeof(T).Name}");
    }

    private static BFloat16[] ConvertBFloat16(ReadOnlySpan<byte> data)
    {
        var ushortSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(data);
        var result = new BFloat16[ushortSpan.Length];

        for (var i = 0; i < ushortSpan.Length; i++)
        {
            var value = (uint)ushortSpan[i] << 16;
            result[i] = new BFloat16(BitConverter.Int32BitsToSingle((int)value));
        }

        return result;
    }

    private static Half[] ConvertHalf(ReadOnlySpan<byte> data)
    {
        var ushortSpan = MemoryMarshal.Cast<byte, ushort>(data);
        var result = new Half[ushortSpan.Length];

        for (var i = 0; i < ushortSpan.Length; i++)
        {
            result[i] = BitConverter.UInt16BitsToHalf(ushortSpan[i]);
        }

        return result;
    }

    private static Complex64[] ConvertComplex64(ReadOnlySpan<byte> data)
    {
        var floatSpan = MemoryMarshal.Cast<byte, float>(data);
        var result = new Complex64[floatSpan.Length / 2];

        for (var i = 0; i < result.Length; i++)
        {
            result[i] = new Complex64(floatSpan[i * 2], floatSpan[i * 2 + 1]);
        }

        return result;
    }

    private static Complex128[] ConvertComplex128(ReadOnlySpan<byte> data)
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

    internal static T GetValue<T>(this AttributeProto attribute, OnnxModelBaseOptions options)
    {
        var value = GetValue(attribute, options);

        if (value is T typed)
        {
            return typed;
        }

        throw new InvalidCastException($"Attribute '{attribute.Name}' is {value.GetType().Name}, not {typeof(T).Name}");
    }

    internal static OnnxAttribute FromProto(AttributeProto attribute, OnnxModelBaseOptions options)
    {
        return attribute.Type switch
        {
            AttributeProto.Types.AttributeType.Float => OnnxAttribute.FromProto<float>(attribute, options),
            AttributeProto.Types.AttributeType.Int => OnnxAttribute.FromProto<long>(attribute, options),
            AttributeProto.Types.AttributeType.String => OnnxAttribute.FromProto<string>(attribute, options),

            AttributeProto.Types.AttributeType.Tensor => OnnxAttribute.FromProto<OnnxTensor>(attribute, options),
            AttributeProto.Types.AttributeType.Graph => OnnxAttribute.FromProto<OnnxGraph>(attribute, options),
            AttributeProto.Types.AttributeType.SparseTensor => OnnxAttribute.FromProto<OnnxSparseTensor>(attribute, options),
            AttributeProto.Types.AttributeType.TypeProto => OnnxAttribute.FromProto<OnnxValueType>(attribute, options),

            AttributeProto.Types.AttributeType.Floats => OnnxAttribute.FromProto<float[]>(attribute, options),
            AttributeProto.Types.AttributeType.Ints => OnnxAttribute.FromProto<long[]>(attribute, options),
            AttributeProto.Types.AttributeType.Strings => OnnxAttribute.FromProto<string[]>(attribute, options),

            AttributeProto.Types.AttributeType.Tensors => OnnxAttribute.FromProto<OnnxTensor[]>(attribute, options),
            AttributeProto.Types.AttributeType.Graphs => OnnxAttribute.FromProto<OnnxGraph[]>(attribute, options),
            AttributeProto.Types.AttributeType.SparseTensors => OnnxAttribute.FromProto<OnnxSparseTensor[]>(attribute, options),
            AttributeProto.Types.AttributeType.TypeProtos => OnnxAttribute.FromProto<OnnxValueType[]>(attribute, options),

            _ => throw new NotImplementedException($"Not implemented for '{attribute.Type}'"),
        };
    }

    internal static object GetValue(this AttributeProto attribute, OnnxModelBaseOptions options)
    {
        return attribute.Type switch
        {
            AttributeProto.Types.AttributeType.Float => attribute.F,
            AttributeProto.Types.AttributeType.Int => attribute.I,
            AttributeProto.Types.AttributeType.String => attribute.S.ToStringUtf8(),

            AttributeProto.Types.AttributeType.Tensor => FromProto(attribute.T, options),
            AttributeProto.Types.AttributeType.Graph => new OnnxGraph(attribute.G, options),
            AttributeProto.Types.AttributeType.SparseTensor => FromProto(attribute.SparseTensor, options),
            AttributeProto.Types.AttributeType.TypeProto => OnnxValueType.FromProto(attribute.Tp),

            AttributeProto.Types.AttributeType.Floats => attribute.Floats.ToArray(),
            AttributeProto.Types.AttributeType.Ints => attribute.Ints.ToArray(),
            AttributeProto.Types.AttributeType.Strings => attribute.Strings.Select(x => x.ToStringUtf8()).ToArray(),

            AttributeProto.Types.AttributeType.Tensors => attribute.Tensors.Select(x => FromProto(x, options)).ToArray(),
            AttributeProto.Types.AttributeType.Graphs => attribute.Graphs.Select(x => new OnnxGraph(x, options)).ToArray(),
            AttributeProto.Types.AttributeType.SparseTensors => attribute.SparseTensors.Select(x => FromProto(x, options)).ToArray(),
            AttributeProto.Types.AttributeType.TypeProtos => attribute.TypeProtos.Select(x => OnnxValueType.FromProto(x)).ToArray(),

            _ => throw new NotImplementedException($"Unsupported attribute type {attribute.Type}")
        };
    }

    internal static AttributeProto.Types.AttributeType GetAttributeType(this Type type)
    {
        if (type == typeof(object))
            return AttributeProto.Types.AttributeType.Undefined;

        if (type == typeof(float))
            return AttributeProto.Types.AttributeType.Float;

        if (type == typeof(long) || type == typeof(int) || type == typeof(short) || type == typeof(byte))
            return AttributeProto.Types.AttributeType.Int;

        if (type == typeof(string))
            return AttributeProto.Types.AttributeType.String;

        if (type == typeof(TensorProto))
            return AttributeProto.Types.AttributeType.Tensor;

        if (type == typeof(GraphProto))
            return AttributeProto.Types.AttributeType.Graph;

        if (type == typeof(SparseTensorProto))
            return AttributeProto.Types.AttributeType.SparseTensor;

        if (type == typeof(OnnxValueType))
            return AttributeProto.Types.AttributeType.TypeProto;

        var elementType = TypeHelper.GetCollectionElementType(type);

        if (elementType == typeof(float))
            return AttributeProto.Types.AttributeType.Floats;

        if (elementType == typeof(long) || elementType == typeof(int) || elementType == typeof(short) || elementType == typeof(byte))
            return AttributeProto.Types.AttributeType.Ints;

        if (elementType == typeof(string))
            return AttributeProto.Types.AttributeType.Strings;

        if (elementType == typeof(TensorProto))
            return AttributeProto.Types.AttributeType.Tensors;

        if (elementType == typeof(GraphProto))
            return AttributeProto.Types.AttributeType.Graphs;

        if (elementType == typeof(SparseTensorProto))
            return AttributeProto.Types.AttributeType.SparseTensors;

        if (elementType == typeof(OnnxValueType))
            return AttributeProto.Types.AttributeType.TypeProtos;

        throw new NotImplementedException($"Type '{type}' is not supported");
    }

    private static void ResetValue(this AttributeProto attribute)
    {
        ArgumentNullException.ThrowIfNull(attribute);

        attribute.F = default;
        attribute.I = default;
        attribute.S = ByteString.Empty;
        attribute.T = null;
        attribute.G = null;
        attribute.SparseTensor = null;
        attribute.Tp = null;

        attribute.Floats.Clear();
        attribute.Ints.Clear();
        attribute.Strings.Clear();
        attribute.Tensors.Clear();
        attribute.Graphs.Clear();
        attribute.SparseTensors.Clear();
        attribute.TypeProtos.Clear();

        attribute.Type = AttributeProto.Types.AttributeType.Undefined;
    }

    internal static void SetValue<T>(this AttributeProto attribute, T value)
    {
        ArgumentNullException.ThrowIfNull(attribute);
        ArgumentNullException.ThrowIfNull(value);

        attribute.ResetValue();

        var type = GetAttributeType(typeof(T));
        attribute.Type = type;

        switch (type)
        {
            case AttributeProto.Types.AttributeType.Float:
                attribute.F = Convert.ToSingle(value);
                break;
            case AttributeProto.Types.AttributeType.Int:
                attribute.I = Convert.ToInt64(value);
                break;
            case AttributeProto.Types.AttributeType.String:
                attribute.S = ByteString.CopyFromUtf8(Convert.ToString(value)!);
                break;
            case AttributeProto.Types.AttributeType.Tensor:
                attribute.T = TypeHelper.Require<OnnxTensor>(value).ToProto();
                break;
            case AttributeProto.Types.AttributeType.Graph:
                attribute.G = TypeHelper.Require<OnnxGraph>(value).ToProto();
                break;
            case AttributeProto.Types.AttributeType.SparseTensor:
                attribute.SparseTensor = TypeHelper.Require<OnnxSparseTensor>(value).ToProto();
                break;
            case AttributeProto.Types.AttributeType.TypeProto:
                attribute.Tp = TypeHelper.Require<OnnxValueType>(value).ToProto();
                break;
            case AttributeProto.Types.AttributeType.Floats:
                attribute.Floats.Set(TypeHelper.RequireMany<float>(value));
                break;
            case AttributeProto.Types.AttributeType.Ints:
                attribute.Ints.Set(TypeHelper.RequireMany<long>(value));
                break;
            case AttributeProto.Types.AttributeType.Strings:
                attribute.Strings.Set(TypeHelper.RequireMany<string>(value).Select(ByteString.CopyFromUtf8));
                break;
            case AttributeProto.Types.AttributeType.Tensors:
                attribute.Tensors.Set(TypeHelper.RequireMany<OnnxTensor>(value).Select(x => x.ToProto()));
                break;
            case AttributeProto.Types.AttributeType.Graphs:
                attribute.Graphs.Set(TypeHelper.RequireMany<OnnxGraph>(value).Select(x => x.ToProto()));
                break;
            case AttributeProto.Types.AttributeType.SparseTensors:
                attribute.SparseTensors.Set(TypeHelper.RequireMany<OnnxSparseTensor>(value).Select(x => x.ToProto()));
                break;
            case AttributeProto.Types.AttributeType.TypeProtos:
                attribute.TypeProtos.Set(TypeHelper.RequireMany<OnnxValueType>(value).Select(x => x.ToProto()));
                break;
            default:
                throw new NotSupportedException($"Unsupported attribute type {typeof(T).Name}");
        }
    }

    internal static void SetValue<T>(this TensorProto tensor, T value, params long[] shape)
    {
        tensor.Dims.Set(shape);

        tensor.DoubleData.Clear();
        tensor.FloatData.Clear();
        tensor.Int32Data.Clear();
        tensor.Int64Data.Clear();
        tensor.Uint64Data.Clear();
        tensor.StringData.Clear();
        tensor.RawData = ByteString.Empty;

        switch (value)
        {
            case float[] f:
                tensor.DataType = (int)TensorProto.Types.DataType.Float;
                tensor.RawData = Pack(f);
                break;

            case double[] d:
                tensor.DataType = (int)TensorProto.Types.DataType.Double;
                tensor.RawData = Pack(d);
                break;

            case int[] i32:
                tensor.DataType = (int)TensorProto.Types.DataType.Int32;
                tensor.RawData = Pack(i32);
                break;

            case long[] i64:
                tensor.DataType = (int)TensorProto.Types.DataType.Int64;
                tensor.RawData = Pack(i64);
                break;

            case byte[] u8:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint8;
                tensor.RawData = ByteString.CopyFrom(u8);
                break;

            case sbyte[] i8:
                tensor.DataType = (int)TensorProto.Types.DataType.Int8;
                tensor.RawData = Pack(i8);
                break;

            case short[] i16:
                tensor.DataType = (int)TensorProto.Types.DataType.Int16;
                tensor.RawData = Pack(i16);
                break;

            case ushort[] u16:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint16;
                tensor.RawData = Pack(u16);
                break;

            case uint[] u32:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint32;
                tensor.RawData = Pack(u32);
                break;

            case ulong[] u64:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint64;
                tensor.RawData = Pack(u64);
                break;

            case bool[] b:
                tensor.DataType = (int)TensorProto.Types.DataType.Bool;
                tensor.RawData = ByteString.CopyFrom(b.Select(x => (byte)(x ? 1 : 0)).ToArray());
                break;

            case Half[] h:
                tensor.DataType = (int)TensorProto.Types.DataType.Float16;
                tensor.RawData = PackHalf(h);
                break;

            case BFloat16[] bf when typeof(T) == typeof(BFloat16[]):
                tensor.DataType = (int)TensorProto.Types.DataType.Bfloat16;
                tensor.RawData = PackBFloat16(bf);
                break;

            case Complex64[] c64:
                tensor.DataType = (int)TensorProto.Types.DataType.Complex64;
                tensor.RawData = PackComplex64(c64);
                break;

            case Complex128[] c128:
                tensor.DataType = (int)TensorProto.Types.DataType.Complex128;
                tensor.RawData = PackComplex128(c128);
                break;

            case string[] s:
                tensor.DataType = (int)TensorProto.Types.DataType.String;
                tensor.StringData.Set(s.Select(ByteString.CopyFromUtf8));
                break;

            default:
                throw new NotSupportedException($"Unsupported tensor type {typeof(T)}");
        }
    }

    internal static void Set<T>(this RepeatedField<T> collection, IEnumerable<T> items)
    {
        collection.Clear();
        foreach (var x in items)
        {
            collection.Add(x);
        }
    }
}
