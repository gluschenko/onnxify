using System.Collections.Immutable;
using Onnx;
using Onnxify.Helpers;

namespace Onnxify;

/// <summary>
/// Base type for ONNX value type descriptors used by graph inputs, outputs, and intermediate values.
/// </summary>
/// <remarks>
/// ONNX supports tensor, sparse tensor, sequence, map, opaque, and optional value types. The currently usable construction surface focuses on tensor types through <see cref="OnnxTensorType"/>.
/// </remarks>
public abstract class OnnxValueType
{
    /// <summary>
    /// Gets the optional ONNX denotation string that describes the semantic meaning of the value type.
    /// </summary>
    public string Denotation { get; }

    /// <summary>
    /// Initializes a value-type descriptor with an ONNX denotation.
    /// </summary>
    /// <param name="denotation">Semantic denotation to write into ONNX type metadata, or an empty string when none is needed.</param>
    protected OnnxValueType(string denotation)
    {
        Denotation = denotation;
    }

    internal virtual TypeProto ToProto()
    {
        var proto = new TypeProto
        {
            Denotation = Denotation,
        };

        return proto;
    }

    internal static OnnxValueType FromProto(TypeProto type)
    {
        return type.ValueCase switch
        {
            TypeProto.ValueOneofCase.TensorType => FromProto(type.TensorType, type),
            TypeProto.ValueOneofCase.SparseTensorType => FromProto(type.SparseTensorType, type),
            TypeProto.ValueOneofCase.OpaqueType => FromProto(type.OpaqueType, type),
            TypeProto.ValueOneofCase.SequenceType => FromProto(type.SequenceType, type),
            TypeProto.ValueOneofCase.MapType => FromProto(type.MapType, type),
            TypeProto.ValueOneofCase.OptionalType => FromProto(type.OptionalType, type),
            TypeProto.ValueOneofCase.None => throw new NotImplementedException($"Not implemented for '{type.ValueCase}'"),
            _ => throw new NotImplementedException($"Not implemented for '{type.ValueCase}'"),
        };
    }

    /// <summary>
    /// Returns a compact diagnostic label for the value-type descriptor.
    /// </summary>
    /// <returns>A string intended for graph inspection.</returns>
    public override string ToString()
    {
        return GetType().Name;
    }

    internal static OnnxTensorType FromProto(TypeProto.Types.Tensor proto, TypeProto typeProto)
    {
        var type = OnnxHelper.GetSystemType((TensorProto.Types.DataType)proto.ElemType);
        var shape = OnnxTensorShape.FromProto(proto.Shape);
        var denotation = typeProto.Denotation;

        return new OnnxTensorType(type, shape, denotation);
    }

    internal static OnnxValueType FromProto(TypeProto.Types.SparseTensor tensor, TypeProto typeProto)
    {
        throw new Exception("TODO");
    }

    internal static OnnxValueType FromProto(TypeProto.Types.Opaque tensor, TypeProto typeProto)
    {
        throw new Exception("TODO");
    }

    internal static OnnxValueType FromProto(TypeProto.Types.Sequence tensor, TypeProto typeProto)
    {
        throw new Exception("TODO");
    }

    internal static OnnxValueType FromProto(TypeProto.Types.Map tensor, TypeProto typeProto)
    {
        throw new Exception("TODO");
    }

    internal static OnnxValueType FromProto(TypeProto.Types.Optional tensor, TypeProto typeProto)
    {
        throw new Exception("TODO");
    }
}

/// <summary>
/// Describes an ONNX tensor value: element type, optional shape, and optional denotation.
/// </summary>
/// <remarks>
/// Use an unknown shape when a model accepts any rank or when shape inference is expected to fill details later. Use symbolic dimensions to preserve dynamic axes such as <c>batch</c> or <c>sequence</c>.
/// </remarks>
public sealed class OnnxTensorType : OnnxValueType
{
    /// <summary>
    /// Gets the CLR element type that maps to the ONNX tensor element type.
    /// </summary>
    public Type Type { get; }

    /// <summary>
    /// Gets the optional tensor shape. A <see langword="null"/> shape means the tensor rank is unspecified.
    /// </summary>
    public OnnxTensorShape? Shape { get; }

    /// <summary>
    /// Creates a tensor type descriptor from a CLR element type and optional shape.
    /// </summary>
    /// <param name="type">CLR element type supported by Onnxify tensor serialization.</param>
    /// <param name="shape">Tensor shape, or <see langword="null"/> to leave rank unspecified.</param>
    /// <param name="denotation">Optional ONNX denotation string.</param>
    public OnnxTensorType(
        Type type,
        OnnxTensorShape? shape,
        string denotation
    ) : base(denotation)
    {
        Type = type;
        Shape = shape;
    }

    /// <summary>
    /// Creates a shaped tensor type from a CLR element type.
    /// </summary>
    /// <typeparam name="T">CLR element type supported by Onnxify tensor serialization.</typeparam>
    /// <param name="shape">Dimensions in ONNX order; values may be fixed sizes or symbolic names.</param>
    /// <param name="denotation">Optional ONNX denotation string.</param>
    /// <returns>A tensor type descriptor suitable for graph inputs, outputs, and value-info entries.</returns>
    public static OnnxTensorType Create<T>(IEnumerable<OnnxDimension> shape, string denotation = "")
    {
        var tensorShape = OnnxTensorShape.Create(shape);
        var type = typeof(T);
        var result = new OnnxTensorType(
            type: type,
            shape: tensorShape,
            denotation: denotation
        );

        return result;
    }

    /// <summary>
    /// Creates a tensor type with an unspecified rank.
    /// </summary>
    /// <typeparam name="T">CLR element type supported by Onnxify tensor serialization.</typeparam>
    /// <param name="denotation">Optional ONNX denotation string.</param>
    /// <returns>A tensor type descriptor without shape metadata.</returns>
    public static OnnxTensorType Create<T>(string denotation = "")
    {
        var type = typeof(T);
        var result = new OnnxTensorType(
            type: type,
            shape: null,
            denotation: denotation
        );

        return result;
    }

    internal override TypeProto ToProto()
    {
        var proto = base.ToProto();

        proto.TensorType = new TypeProto.Types.Tensor
        {
            ElemType = (int)OnnxHelper.GetDataType(Type),
            Shape = Shape?.ToProto(),
        };

        return proto;
    }

    /// <summary>
    /// Returns a compact diagnostic representation including element type, shape when known, and denotation when present.
    /// </summary>
    /// <returns>A string intended for graph inspection.</returns>
    public override string ToString()
    {
        var shape = Shape is null
            ? string.Empty
            : $"[{string.Join(", ", Shape.Dimensions)}]";

        var denotation = string.IsNullOrWhiteSpace(Denotation)
            ? string.Empty
            : $" ({Denotation})";

        return $"{Type.Name}{shape}{denotation}";
    }
}

/// <summary>
/// Represents the ordered dimensions of an ONNX tensor shape.
/// </summary>
/// <remarks>
/// Dimensions can be fixed integer sizes or symbolic names. Symbolic dimensions are useful for dynamic axes that should keep their identity in exported ONNX metadata.
/// </remarks>
public class OnnxTensorShape
{
    /// <summary>
    /// Gets the immutable ordered dimension descriptors.
    /// </summary>
    public ImmutableArray<OnnxDimension> Dimensions { get; }

    /// <summary>
    /// Creates a tensor shape from explicit ONNX dimension descriptors.
    /// </summary>
    /// <param name="dimentions">Dimensions in ONNX order.</param>
    public OnnxTensorShape(IEnumerable<OnnxDimension> dimentions)
    {
        Dimensions = dimentions.ToImmutableArray();
    }

    /// <summary>
    /// Creates a tensor shape from explicit ONNX dimension descriptors.
    /// </summary>
    /// <param name="dimensions">Dimensions in ONNX order.</param>
    /// <returns>A shape preserving fixed and symbolic dimensions.</returns>
    public static OnnxTensorShape Create(IEnumerable<OnnxDimension> dimensions)
    {
        return new OnnxTensorShape(dimensions);
    }

    /// <summary>
    /// Creates a tensor shape from fixed dimension sizes.
    /// </summary>
    /// <param name="shape">Fixed dimension sizes in ONNX order.</param>
    /// <returns>A shape with integer dimensions.</returns>
    public static OnnxTensorShape Create(IEnumerable<long> shape)
    {
        return new OnnxTensorShape(shape.Select(x => new OnnxDimension<long>(x)).ToArray());
    }

    /// <summary>
    /// Creates a tensor shape from symbolic dimension names.
    /// </summary>
    /// <param name="shape">Symbolic dimension names in ONNX order.</param>
    /// <returns>A shape with symbolic dimensions.</returns>
    public static OnnxTensorShape Create(IEnumerable<string> shape)
    {
        return new OnnxTensorShape(shape.Select(x => new OnnxDimension<string>(x)).ToArray());
    }

    internal static OnnxTensorShape? FromProto(TensorShapeProto? proto)
    {
        if (proto is null)
        {
            return null;
        }

        var dimentions = proto.Dim.Select(x => OnnxDimension.FromProto(x)).ToArray();
        return new OnnxTensorShape(dimentions);
    }

    internal TensorShapeProto ToProto()
    {
        var proto = new TensorShapeProto();
        proto.Dim.Set(Dimensions.Select(x => x.ToProto()));
        return proto;
    }
}

/// <summary>
/// Base type for one ONNX tensor dimension, either a fixed size or a symbolic parameter.
/// </summary>
public abstract class OnnxDimension
{
    /// <summary>
    /// Gets the dimension payload as either <see cref="long"/> for fixed sizes or <see cref="string"/> for symbolic dimensions.
    /// </summary>
    /// <returns>The dimension value to serialize into ONNX shape metadata.</returns>
    public abstract object GetValue();

    internal abstract TensorShapeProto.Types.Dimension ToProto();
    internal static OnnxDimension FromProto(TensorShapeProto.Types.Dimension x)
    {
        return x.ValueCase switch
        {
            TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue => new OnnxDimension<long>(x.DimValue),
            TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam => new OnnxDimension<string>(x.DimParam),
            _ => throw new NotImplementedException($"Not implemented for '{x.ValueCase}'"),
        };
    }

    /// <summary>
    /// Converts a fixed integer size into an ONNX dimension.
    /// </summary>
    /// <param name="value">Fixed dimension size.</param>
    public static implicit operator OnnxDimension(long value) => new OnnxDimension<long>(value);

    /// <summary>
    /// Converts a symbolic dimension name into an ONNX dimension.
    /// </summary>
    /// <param name="value">Symbolic dimension parameter name.</param>
    public static implicit operator OnnxDimension(string value) => new OnnxDimension<string>(value);

    /// <summary>
    /// Returns the dimension payload as it should appear in diagnostic shape strings.
    /// </summary>
    /// <returns>A fixed size or symbolic dimension name.</returns>
    public override string ToString()
    {
        return GetValue().ToString() ?? string.Empty;
    }
}

/// <summary>
/// Represents a fixed or symbolic ONNX tensor dimension.
/// </summary>
/// <typeparam name="T">Use <see cref="long"/> for fixed sizes or <see cref="string"/> for symbolic dimensions.</typeparam>
public class OnnxDimension<T> : OnnxDimension where T : notnull
{
    /// <summary>
    /// Gets the fixed size or symbolic dimension name.
    /// </summary>
    public T Value { get; }

    /// <inheritdoc />
    public override object GetValue()
    {
        return Value;
    }

    /// <summary>
    /// Creates a dimension from an ONNX-supported dimension payload.
    /// </summary>
    /// <param name="value">Fixed dimension size or symbolic dimension name.</param>
    /// <exception cref="NotSupportedException">Thrown when <typeparamref name="T"/> is neither <see cref="long"/> nor <see cref="string"/>.</exception>
    public OnnxDimension(T value)
    {
        if (value is not long and not string)
        {
            throw new NotSupportedException($"Not supported for ${value.GetType().Name}");
        }

        Value = value;
    }

    internal override TensorShapeProto.Types.Dimension ToProto()
    {
        var proto = new TensorShapeProto.Types.Dimension();

        if (Value is long longValue)
        {
            proto.DimValue = longValue;
        }
        else if (Value is string stringValue)
        {
            proto.DimParam = stringValue;
        }
        else
        {
            throw new NotImplementedException($"Not implemented for '{Value.GetType().Name}'");
        }

        return proto;
    }
}


