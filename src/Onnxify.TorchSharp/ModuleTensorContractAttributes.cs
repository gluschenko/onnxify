using System.Globalization;

namespace Onnxify.TorchSharp;

/// <summary>
/// Describes one fixed or symbolic tensor dimension in a TorchSharp module contract.
/// </summary>
public readonly record struct TensorDimension
{
    /// <summary>
    /// Gets the dimension payload as either <see cref="long"/> for fixed sizes or <see cref="string"/> for symbolic names.
    /// </summary>
    public object Value { get; }

    /// <summary>
    /// Creates a fixed-size tensor dimension.
    /// </summary>
    public TensorDimension(long value)
    {
        Value = value;
    }

    /// <summary>
    /// Creates a symbolic tensor dimension.
    /// </summary>
    public TensorDimension(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        Value = value;
    }

    public static implicit operator TensorDimension(int value) => new((long)value);

    public static implicit operator TensorDimension(long value) => new(value);

    public static implicit operator TensorDimension(string value) => new(value);

    public override string ToString()
    {
        return Value switch
        {
            long fixedSize => fixedSize.ToString(CultureInfo.InvariantCulture),
            string symbolicName => symbolicName,
            _ => Value.ToString() ?? string.Empty,
        };
    }
}

/// <summary>
/// Declares one tensor input for a TorchSharp module.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
public sealed class ModuleInputAttribute : Attribute
{
    public ModuleInputAttribute(
        string name,
        global::TorchSharp.torch.ScalarType dataType,
        params object[] dimensions
    )
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Name = name;
        DataType = dataType;
        Dimensions = NormalizeDimensions(dimensions);
    }

    public string Name { get; }

    public global::TorchSharp.torch.ScalarType DataType { get; }

    public IReadOnlyList<TensorDimension> Dimensions { get; }

    internal static TensorDimension[] NormalizeDimensions(IReadOnlyList<object> dimensions)
    {
        var result = new TensorDimension[dimensions.Count];
        for (var index = 0; index < dimensions.Count; index++)
        {
            result[index] = dimensions[index] switch
            {
                int fixedSize => fixedSize,
                long fixedSize => fixedSize,
                string symbolicName => symbolicName,
                _ => throw new ArgumentException(
                    $"Unsupported tensor dimension value '{dimensions[index]}' of type '{dimensions[index].GetType().Name}'."
                ),
            };
        }

        return result;
    }
}

/// <summary>
/// Declares one tensor output for a TorchSharp module.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
public sealed class ModuleOutputAttribute : Attribute
{
    public ModuleOutputAttribute(
        string name,
        global::TorchSharp.torch.ScalarType dataType,
        params object[] dimensions
    )
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Name = name;
        DataType = dataType;
        Dimensions = ModuleInputAttribute.NormalizeDimensions(dimensions);
    }

    public string Name { get; }

    public global::TorchSharp.torch.ScalarType DataType { get; }

    public IReadOnlyList<TensorDimension> Dimensions { get; }
}
