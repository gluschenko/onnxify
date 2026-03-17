using Onnx;

namespace Onnxify;

public class OnnxAttribute<T> : OnnxAttribute
{
    public override string Name => _attribute.Name;
    public AttributeProto.Types.AttributeType Type { get; init; }
    public T Value { get; set; }

    private readonly AttributeProto _attribute;

    public OnnxAttribute(AttributeProto attribute)
    {
        _attribute = attribute;

        Type = _attribute.Type;
        Value = OnnxHelper.GetValue<T>(attribute);
    }

    internal override AttributeProto ToProto()
    {
        var newAttribute = _attribute.Clone();
        newAttribute.Name = Name;
        newAttribute.Type = Type;
        newAttribute.SetValue(Value);

        return newAttribute;
    }
}
