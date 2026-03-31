using Onnx;

namespace Onnxify;

public abstract class OnnxValue : IOnnxGraphEdge
{
    public abstract string Name { get; }
    public abstract OnnxValueType Type { get; }

    internal abstract ValueInfoProto ToProto();

    internal static OnnxValue FromProto(ValueInfoProto proto)
    {
        var type = proto.Type.ValueCase;

        return type switch
        {
            TypeProto.ValueOneofCase.TensorType => FromProto<OnnxTensorType>(proto),
            TypeProto.ValueOneofCase.SequenceType => throw new NotImplementedException("TODO"),
            TypeProto.ValueOneofCase.MapType => throw new NotImplementedException("TODO"),
            TypeProto.ValueOneofCase.SparseTensorType => throw new NotImplementedException("TODO"),
            TypeProto.ValueOneofCase.OpaqueType => throw new NotImplementedException("TODO"),
            TypeProto.ValueOneofCase.OptionalType => throw new NotImplementedException("TODO"),
            TypeProto.ValueOneofCase.None => throw new NotImplementedException($"Not implemented for '{type}'"),
            _ => throw new NotImplementedException($"Not implemented for '{type}'"),
        };
    }

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
        newValue.Type = Type.ToProto();

        return newValue;
    }

    public override string ToString()
    {
        return $"{Name}: {Type}";
    }
}
