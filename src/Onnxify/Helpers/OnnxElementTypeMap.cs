using Onnx;
using Onnxify.Data.Numerics;
#if NET5_0_OR_GREATER
using Half = System.Half;
#else
using Half = Onnxify.Data.Numerics.Float16;
#endif

namespace Onnxify.Helpers;

/// <summary>Shared CLR-to-ONNX tensor element type mapping.</summary>
internal static class OnnxElementTypeMap
{
    private static readonly (Type ClrType, TensorProto.Types.DataType OnnxType)[] TYPE_PAIRS =
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

    private static readonly IReadOnlyDictionary<Type, TensorProto.Types.DataType> CLR_TO_ONNX =
        TYPE_PAIRS.ToDictionary(static pair => pair.ClrType, static pair => pair.OnnxType);

    private static readonly IReadOnlyDictionary<TensorProto.Types.DataType, Type> ONNX_TO_CLR =
        TYPE_PAIRS.ToDictionary(static pair => pair.OnnxType, static pair => pair.ClrType);

    public static TensorProto.Types.DataType GetDataType(Type clrType)
        => CLR_TO_ONNX.TryGetValue(clrType, out var dataType)
            ? dataType
            : throw new NotImplementedException($"Type '{clrType}' is not supported.");

    public static Type GetSystemType(TensorProto.Types.DataType dataType)
        => ONNX_TO_CLR.TryGetValue(dataType, out var clrType)
            ? clrType
            : throw new NotImplementedException($"Data type '{dataType}' is not supported.");
}
