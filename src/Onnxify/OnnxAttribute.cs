using Onnx;

namespace Onnxify;

public abstract class OnnxAttribute
{
    public abstract string Name { get; }
    internal abstract AttributeProto ToProto();

    public abstract object GetValue();

    internal static OnnxAttribute<T> FromProto<T>(AttributeProto attribute) where T : notnull
    {
        var name = attribute.Name;
        var value = OnnxHelper.GetValue<T>(attribute);

        return new OnnxAttribute<T>(
            name: name,
            value: value,
            proto: attribute
        );
    }
}

public class OnnxAttribute<T> : OnnxAttribute where T : notnull
{
    public override string Name { get; }
    public Type Type => typeof(T);
    public T Value { get; private set; }

    private readonly AttributeProto _attribute;

    public OnnxAttribute(string name, T value) : this(name, value, null) { }

    internal OnnxAttribute(
        string name,
        T value,
        AttributeProto? proto
    )
    {
        Name = name;
        Value = value;

        _attribute = proto ?? new AttributeProto
        {
            Name = Name,
        };
    }

    internal override AttributeProto ToProto()
    {
        var newAttribute = _attribute.Clone();
        newAttribute.Name = Name;
        newAttribute.Type = OnnxHelper.GetAttributeType(Type);
        newAttribute.SetValue(Value);

        return newAttribute;
    }

    public override object GetValue()
    {
        return Value;
    }
}
