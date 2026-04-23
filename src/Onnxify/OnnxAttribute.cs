using System.Collections;
using Onnx;
using Onnxify.Helpers;

namespace Onnxify;

/// <summary>
/// Base type for ONNX node attributes whose concrete value type may only be known after parsing the model.
/// </summary>
/// <remarks>
/// Attribute values can be scalars, tensors, graphs, sparse tensors, type descriptors, or ONNX-supported arrays of those values. Use <see cref="GetValue"/> when inspecting generic nodes and <see cref="OnnxAttribute{T}"/> when constructing attributes with a known CLR type.
/// </remarks>
public abstract class OnnxAttribute
{
    /// <summary>
    /// Gets the ONNX attribute name as defined by the operator schema.
    /// </summary>
    public abstract string Name { get; }
    internal abstract AttributeProto ToProto();

    /// <summary>
    /// Gets the attribute payload as a boxed CLR value.
    /// </summary>
    /// <returns>The value represented by this attribute.</returns>
    public abstract object GetValue();

    internal static OnnxAttribute<T> FromProto<T>(AttributeProto attribute, OnnxModelBaseOptions options) where T : notnull
    {
        var name = attribute.Name;
        var value = OnnxHelper.GetValue<T>(attribute, options);

        return new OnnxAttribute<T>(
            name: name,
            value: value,
            proto: attribute
        );
    }
}

/// <summary>
/// Represents an ONNX attribute with a known CLR value type.
/// </summary>
/// <typeparam name="T">CLR type used to infer the ONNX attribute kind during serialization.</typeparam>
/// <remarks>
/// Numeric integer attribute values are serialized to ONNX's 64-bit integer attribute representation, while tensor and graph values are serialized through their Onnxify wrappers.
/// </remarks>
public class OnnxAttribute<T> : OnnxAttribute where T : notnull
{
    /// <inheritdoc />
    public override string Name { get; }

    /// <summary>
    /// Gets the CLR type used to choose the ONNX attribute kind.
    /// </summary>
    public Type Type => typeof(T);

    /// <summary>
    /// Gets the typed attribute value.
    /// </summary>
    public T Value { get; private set; }

    private readonly AttributeProto _attribute;

    /// <summary>
    /// Creates an attribute that can be attached to a node.
    /// </summary>
    /// <param name="name">Attribute name exactly as expected by the ONNX operator schema.</param>
    /// <param name="value">Attribute value to serialize.</param>
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

    /// <inheritdoc />
    public override object GetValue()
    {
        return Value;
    }

    /// <summary>
    /// Returns a compact diagnostic representation of the attribute name, type, and value.
    /// </summary>
    /// <returns>A string intended for inspection, not for stable serialization.</returns>
    public override string ToString()
    {
        return $"{Name}: {Type.Name} = {FormatValue(Value)}";
    }

    private static string FormatValue(T value)
    {
        return value switch
        {
            string text => text,
            IEnumerable values when value is not string => $"[{string.Join(", ", values.Cast<object?>().Select(FormatObject))}]",
            _ => FormatObject(value),
        };
    }

    private static string FormatObject(object? value)
    {
        return value?.ToString() ?? "null";
    }
}
