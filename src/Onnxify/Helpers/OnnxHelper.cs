using Google.Protobuf;
using Google.Protobuf.Collections;
using Onnx;
using Onnxify.Data.Numerics;

namespace Onnxify.Helpers;

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
        var systemType = GetSystemType(type);

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

            var path = Path.Combine(options.DataLocation ?? string.Empty, location);

            var result = options.DataReader.ReadTensorValue(
                location: path,
                offset: offset,
                length: length,
                type: systemType
            );

            return result;
        }

        if (tensor.RawData.Length > 0)
        {
            var span = tensor.RawData.Span;
            var data = DecodeRawData(span, systemType);
            return data;
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

    internal static object DecodeRawData(
        ReadOnlySpan<byte> span,
        Type type
    )
    {
        var tensorType = GetDataType(type);

        return tensorType switch
        {
            TensorProto.Types.DataType.Float => BinaryHelper.Decode<float>(span),
            TensorProto.Types.DataType.Double => BinaryHelper.Decode<double>(span),
            TensorProto.Types.DataType.Int32 => BinaryHelper.Decode<int>(span),
            TensorProto.Types.DataType.Int64 => BinaryHelper.Decode<long>(span),
            TensorProto.Types.DataType.Uint32 => BinaryHelper.Decode<uint>(span),
            TensorProto.Types.DataType.Uint64 => BinaryHelper.Decode<ulong>(span),
            TensorProto.Types.DataType.Int16 => BinaryHelper.Decode<short>(span),
            TensorProto.Types.DataType.Uint16 => BinaryHelper.Decode<ushort>(span),
            TensorProto.Types.DataType.Int8 => BinaryHelper.Decode<sbyte>(span),
            TensorProto.Types.DataType.Uint8 => BinaryHelper.Decode<byte>(span),
            TensorProto.Types.DataType.Bool => BinaryHelper.DecodeBoolArray(span),
            TensorProto.Types.DataType.Float16 => BinaryHelper.DecodeHalf(span),
            TensorProto.Types.DataType.Bfloat16 => BinaryHelper.DecodeBFloat16(span),
            TensorProto.Types.DataType.Complex64 => BinaryHelper.DecodeComplex64(span),
            TensorProto.Types.DataType.Complex128 => BinaryHelper.DecodeComplex128(span),
            TensorProto.Types.DataType.Float8E4M3Fn => BinaryHelper.DecodeFloat8E4M3FN(span),
            TensorProto.Types.DataType.Float8E4M3Fnuz => BinaryHelper.DecodeFloat8E4M3FNUZ(span),
            TensorProto.Types.DataType.Float8E5M2 => BinaryHelper.DecodeFloat8E5M2(span),
            TensorProto.Types.DataType.Float8E5M2Fnuz => BinaryHelper.DecodeFloat8E5M2FNUZ(span),
            TensorProto.Types.DataType.Float4E2M1 => BinaryHelper.DecodeFloat4(span),
            TensorProto.Types.DataType.Float8E8M0 => BinaryHelper.DecodeFloat8E8M0(span),
            TensorProto.Types.DataType.Uint4 => BinaryHelper.DecodeUInt4(span),
            TensorProto.Types.DataType.Int4 => BinaryHelper.DecodeInt4(span),
            TensorProto.Types.DataType.Uint2 => BinaryHelper.DecodeUInt2(span),
            TensorProto.Types.DataType.Int2 => BinaryHelper.DecodeInt2(span),
            _ => throw new NotImplementedException($"Unsupported raw tensor type {type}")
        };
    }

    internal static void SetValue<T>(this TensorProto tensor, T value, params long[] shape)
    {
        tensor.ResetValue();
        tensor.Dims.Set(shape);

        switch (value)
        {
            case float[] f:
                tensor.DataType = (int)TensorProto.Types.DataType.Float;
                tensor.RawData = BinaryHelper.Encode(f);
                break;

            case double[] d:
                tensor.DataType = (int)TensorProto.Types.DataType.Double;
                tensor.RawData = BinaryHelper.Encode(d);
                break;

            case int[] i32:
                tensor.DataType = (int)TensorProto.Types.DataType.Int32;
                tensor.RawData = BinaryHelper.Encode(i32);
                break;

            case long[] i64:
                tensor.DataType = (int)TensorProto.Types.DataType.Int64;
                tensor.RawData = BinaryHelper.Encode(i64);
                break;

            case byte[] u8:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint8;
                tensor.RawData = BinaryHelper.Encode(u8);
                break;

            case sbyte[] i8:
                tensor.DataType = (int)TensorProto.Types.DataType.Int8;
                tensor.RawData = BinaryHelper.Encode(i8);
                break;

            case short[] i16:
                tensor.DataType = (int)TensorProto.Types.DataType.Int16;
                tensor.RawData = BinaryHelper.Encode(i16);
                break;

            case ushort[] u16:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint16;
                tensor.RawData = BinaryHelper.Encode(u16);
                break;

            case uint[] u32:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint32;
                tensor.RawData = BinaryHelper.Encode(u32);
                break;

            case ulong[] u64:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint64;
                tensor.RawData = BinaryHelper.Encode(u64);
                break;

            case bool[] b:
                tensor.DataType = (int)TensorProto.Types.DataType.Bool;
                tensor.RawData = BinaryHelper.EncodeBoolArray(b);
                break;

            case Half[] h:
                tensor.DataType = (int)TensorProto.Types.DataType.Float16;
                tensor.RawData = BinaryHelper.EncodeHalf(h);
                break;

            case BFloat16[] bf when typeof(T) == typeof(BFloat16[]):
                tensor.DataType = (int)TensorProto.Types.DataType.Bfloat16;
                tensor.RawData = BinaryHelper.EncodeBFloat16(bf);
                break;

            case Float8E4M3FN[] f8e4m3fn:
                tensor.DataType = (int)TensorProto.Types.DataType.Float8E4M3Fn;
                tensor.RawData = BinaryHelper.EncodeFloat8E4M3FN(f8e4m3fn);
                break;

            case Float8E4M3FNUZ[] f8e4m3fnuz:
                tensor.DataType = (int)TensorProto.Types.DataType.Float8E4M3Fnuz;
                tensor.RawData = BinaryHelper.EncodeFloat8E4M3FNUZ(f8e4m3fnuz);
                break;

            case Float8E5M2[] f8e5m2:
                tensor.DataType = (int)TensorProto.Types.DataType.Float8E5M2;
                tensor.RawData = BinaryHelper.EncodeFloat8E5M2(f8e5m2);
                break;

            case Float8E5M2FNUZ[] f8e5m2fnuz:
                tensor.DataType = (int)TensorProto.Types.DataType.Float8E5M2Fnuz;
                tensor.RawData = BinaryHelper.EncodeFloat8E5M2FNUZ(f8e5m2fnuz);
                break;

            case Float4E2M1[] f4e2m1:
                tensor.DataType = (int)TensorProto.Types.DataType.Float4E2M1;
                tensor.RawData = BinaryHelper.EncodeFloat4(f4e2m1);
                break;

            case Float8E8M0[] f8e8m0:
                tensor.DataType = (int)TensorProto.Types.DataType.Float8E8M0;
                tensor.RawData = BinaryHelper.EncodeFloat8E8M0(f8e8m0);
                break;

            case UInt4[] u4:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint4;
                tensor.RawData = BinaryHelper.EncodeUInt4(u4);
                break;

            case Int4[] i4:
                tensor.DataType = (int)TensorProto.Types.DataType.Int4;
                tensor.RawData = BinaryHelper.EncodeInt4(i4);
                break;

            case UInt2[] u2:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint2;
                tensor.RawData = BinaryHelper.EncodeUInt2(u2);
                break;

            case Int2[] i2:
                tensor.DataType = (int)TensorProto.Types.DataType.Int2;
                tensor.RawData = BinaryHelper.EncodeInt2(i2);
                break;

            case Complex64[] c64:
                tensor.DataType = (int)TensorProto.Types.DataType.Complex64;
                tensor.RawData = BinaryHelper.EncodeComplex64(c64);
                break;

            case Complex128[] c128:
                tensor.DataType = (int)TensorProto.Types.DataType.Complex128;
                tensor.RawData = BinaryHelper.EncodeComplex128(c128);
                break;

            case string[] s:
                tensor.DataType = (int)TensorProto.Types.DataType.String;
                tensor.StringData.Set(s.Select(ByteString.CopyFromUtf8));
                break;

            default:
                throw new NotSupportedException($"Unsupported tensor type {typeof(T)}");
        }
    }

    internal static void ResetValue(this TensorProto tensor)
    {
        tensor.DoubleData.Clear();
        tensor.FloatData.Clear();
        tensor.Int32Data.Clear();
        tensor.Int64Data.Clear();
        tensor.Uint64Data.Clear();
        tensor.StringData.Clear();
        tensor.RawData = ByteString.Empty;
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

    internal static void Set<T>(this RepeatedField<T> collection, IEnumerable<T> items)
    {
        collection.Clear();
        foreach (var x in items)
        {
            collection.Add(x);
        }
    }
}
