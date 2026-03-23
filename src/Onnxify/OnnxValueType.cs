using System.Collections.Immutable;
using Onnx;

namespace Onnxify;

public abstract class OnnxValueType
{
    public string Denotation { get; }

    protected OnnxValueType(string denotation)
    {
        Denotation = denotation;
    }

    internal virtual TypeProto ToProto()
    {
        var proto = new TypeProto
        {
            Denotation = Denotation
        };

        return proto;
    }

    internal static OnnxValueType FromProto(TypeProto type)
    {
        return type.ValueCase switch
        {
            TypeProto.ValueOneofCase.TensorType => FromProto(type.TensorType, type),
            TypeProto.ValueOneofCase.SparseTensorType => FromProto(type.SparseTensorType),
            TypeProto.ValueOneofCase.OpaqueType => FromProto(type.OpaqueType),
            TypeProto.ValueOneofCase.SequenceType => FromProto(type.SequenceType),
            TypeProto.ValueOneofCase.MapType => FromProto(type.MapType),
            TypeProto.ValueOneofCase.None => throw new NotImplementedException($"Not implemented for '{type.ValueCase}'"),
            _ => throw new NotImplementedException($"Not implemented for '{type.ValueCase}'"),
        };
    }

    internal static OnnxTensorType FromProto(TypeProto.Types.Tensor proto, TypeProto typeProto)
    {
        var type = OnnxHelper.GetSystemType((TensorProto.Types.DataType)proto.ElemType);
        var shape = OnnxTensorShape.FromProto(proto.Shape);
        var denotation = typeProto.Denotation;

        return new OnnxTensorType(type, shape, denotation);
    }

    internal static OnnxValueType FromProto(TypeProto.Types.SparseTensor tensor)
    {
        throw new Exception("TODO");
    }

    internal static OnnxValueType FromProto(TypeProto.Types.Opaque tensor)
    {
        throw new Exception("TODO");
    }

    internal static OnnxValueType FromProto(TypeProto.Types.Sequence tensor)
    {
        throw new Exception("TODO");
    }

    internal static OnnxValueType FromProto(TypeProto.Types.Map tensor)
    {
        throw new Exception("TODO");
    }
}

public sealed class OnnxTensorType : OnnxValueType
{
    public Type Type { get; }
    public OnnxTensorShape? Shape { get; }

    public OnnxTensorType(
        Type type,
        OnnxTensorShape? shape,
        string denotation
    ) : base(denotation)
    {
        Type = type;
        Shape = shape;
    }

    public static OnnxTensorType Create<T>(long[] shape, string denotation = "")
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
}

public class OnnxTensorShape
{
    public ImmutableArray<OnnxDimension> Dimensions { get; }

    public OnnxTensorShape(IEnumerable<OnnxDimension> dimentions)
    {
        Dimensions = dimentions.ToImmutableArray();
    }

    public static OnnxTensorShape Create(IEnumerable<long> shape)
    {
        return new OnnxTensorShape(shape.Select(x => new OnnxDimension<long>(x)).ToArray());
    }

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

public abstract class OnnxDimension
{
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
}

public class OnnxDimension<T> : OnnxDimension where T : notnull
{
    public T Value { get; }

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


