using Onnx;

namespace Onnxify;

/// <summary>
/// Base type for graph values that carry ONNX type metadata, such as inputs, outputs, and intermediate value-info entries.
/// </summary>
/// <remarks>
/// This represents ONNX <c>ValueInfoProto</c>. It is distinct from <see cref="OnnxTensor"/>, which carries initializer data, and from <see cref="OnnxEdge"/>, which carries only a wire name.
/// </remarks>
public abstract class OnnxValue : IOnnxGraphEdge
{
    /// <summary>
    /// Gets the graph wire name described by this value-info entry.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the ONNX type descriptor associated with this graph value.
    /// </summary>
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

/// <summary>
/// Represents a graph value with a specific Onnxify value-type descriptor.
/// </summary>
/// <typeparam name="T">Concrete value-type descriptor, typically <see cref="OnnxTensorType"/>.</typeparam>
public class OnnxValue<T> : OnnxValue where T : OnnxValueType
{
    /// <inheritdoc />
    public override string Name { get; }

    /// <inheritdoc />
    public override T Type { get; }

    private readonly ValueInfoProto _valueInfo;

    public OnnxValue(string name, T type) : this(name, type, null) { }

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

    /// <summary>
    /// Returns a compact diagnostic representation of the value name and ONNX type.
    /// </summary>
    /// <returns>A string intended for graph inspection.</returns>
    public override string ToString()
    {
        return $"{Name}: {Type}";
    }
}
