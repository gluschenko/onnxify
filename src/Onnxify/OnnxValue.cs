using System.Collections.Immutable;
using Onnx;

namespace Onnxify;

public abstract class OnnxValue : IOnnxGraphEdge
{
    public abstract string Name { get; }
    public abstract OnnxValueType Type { get; }

    internal abstract ValueInfoProto ToProto();

    internal static OnnxValue<T> FromProto<T>(ValueInfoProto valueInfo) where T : OnnxValueType
    {
        var name = valueInfo.Name;
        var type = (T)OnnxValueType.FromProto(valueInfo.Type);

        return new OnnxValue<T>(
            name: name,
            type: type,
            valueInfo: valueInfo
        );
    }
}

public class OnnxValue<T> : OnnxValue where T : OnnxValueType
{
    public override string Name { get; }
    public override T Type { get; }

    private readonly ValueInfoProto _valueInfo;

    internal OnnxValue(
        string name,
        T type,
        ValueInfoProto? valueInfo
    )
    {
        Name = name;
        Type = type;

        _valueInfo = valueInfo ?? new ValueInfoProto 
        {
            Name = name,
        };
    }

    internal override ValueInfoProto ToProto()
    {
        var newValue = _valueInfo.Clone();
        newValue.Name = Name;

        return newValue;
    }
}

public abstract class OnnxValueType
{
    public string Denotation { get; }

    protected OnnxValueType(string denotation)
    {
        Denotation = denotation;
    }

    internal abstract TypeProto ToProto();

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
        var type = (TensorProto.Types.DataType)proto.ElemType;
        var shape = OnnxTensorShape.FromProto(proto.Shape);

        return new OnnxTensorType(type, shape, typeProto.Denotation);
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
    public TensorProto.Types.DataType Type { get; }
    public OnnxTensorShape Shape { get; }

    internal OnnxTensorType(
        TensorProto.Types.DataType type,
        OnnxTensorShape shape,
        string denotation
    ) : base(denotation)
    {
        Type = type;
        Shape = shape;
    }

    internal override TypeProto ToProto()
    {
        var proto = new TypeProto
        {
            TensorType = new TypeProto.Types.Tensor
            {
                ElemType = (int)Type,
                Shape = Shape.ToProto(),
            }
        };

        return proto;
    }
}

public class OnnxTensorShape
{
    public ImmutableArray<OnnxDimension> Dimensions { get; }

    internal OnnxTensorShape(IEnumerable<OnnxDimension> dimentions)
    {
        Dimensions = dimentions.ToImmutableArray();
    }

    internal static OnnxTensorShape FromProto(TensorShapeProto proto)
    {
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

