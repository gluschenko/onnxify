namespace Onnxify.Compiler;

/// <summary>Base class for typed compiler literals.</summary>
public abstract class CompilerLiteral : IEquatable<CompilerLiteral>
{
    public bool Equals(CompilerLiteral? other)
    {
        var result = other is not null
            && GetType() == other.GetType()
            && EqualsCore(other);
        return result;
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerLiteral);

    public override int GetHashCode() => GetHashCodeCore();

    protected abstract bool EqualsCore(CompilerLiteral other);

    protected abstract int GetHashCodeCore();
}

/// <summary>Represents the C# <c>null</c> literal without a runtime object.</summary>
public sealed class CompilerNullLiteral : CompilerLiteral
{
    protected override bool EqualsCore(CompilerLiteral other) => other is CompilerNullLiteral;

    protected override int GetHashCodeCore() => 17;
}

/// <summary>Base class for scalar literals with an explicit element type.</summary>
public abstract class CompilerScalarLiteral : CompilerLiteral
{
    protected CompilerScalarLiteral(CompilerElementType elementType)
    {
        ElementType = elementType;
    }

    public CompilerElementType ElementType { get; }
}

public sealed class CompilerBooleanLiteral : CompilerScalarLiteral
{
    public CompilerBooleanLiteral(bool value)
        : base(CompilerElementType.Boolean)
    {
        Value = value;
    }

    public bool Value { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var result = Value == ((CompilerBooleanLiteral)other).Value;
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Value);
}

public sealed class CompilerSignedIntegerLiteral : CompilerScalarLiteral
{
    public CompilerSignedIntegerLiteral(CompilerElementType elementType, long value)
        : base(elementType)
    {
        if (elementType is not (
            CompilerElementType.Int8
            or CompilerElementType.Int16
            or CompilerElementType.Int32
            or CompilerElementType.Int64))
        {
            throw new ArgumentException("The element type must be a signed integer type.", nameof(elementType));
        }

        Value = value;
    }

    public long Value { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var integer = (CompilerSignedIntegerLiteral)other;
        var result = ElementType == integer.ElementType && Value == integer.Value;
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Value);
}

public sealed class CompilerUnsignedIntegerLiteral : CompilerScalarLiteral
{
    public CompilerUnsignedIntegerLiteral(CompilerElementType elementType, ulong value)
        : base(elementType)
    {
        if (elementType is not (
            CompilerElementType.UInt8
            or CompilerElementType.UInt16
            or CompilerElementType.UInt32
            or CompilerElementType.UInt64))
        {
            throw new ArgumentException("The element type must be an unsigned integer type.", nameof(elementType));
        }

        Value = value;
    }

    public ulong Value { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var integer = (CompilerUnsignedIntegerLiteral)other;
        var result = ElementType == integer.ElementType && Value == integer.Value;
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Value);
}

public sealed class CompilerFloatingPointLiteral : CompilerScalarLiteral
{
    public CompilerFloatingPointLiteral(CompilerElementType elementType, double value)
        : base(elementType)
    {
        if (elementType is not (
            CompilerElementType.Float16
            or CompilerElementType.BFloat16
            or CompilerElementType.Float32
            or CompilerElementType.Float64))
        {
            throw new ArgumentException("The element type must be a floating-point type.", nameof(elementType));
        }

        Value = value;
    }

    public double Value { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var floatingPoint = (CompilerFloatingPointLiteral)other;
        var result = ElementType == floatingPoint.ElementType && Value.Equals(floatingPoint.Value);
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Value);
}

public sealed class CompilerComplexLiteral : CompilerScalarLiteral
{
    public CompilerComplexLiteral(CompilerElementType elementType, double real, double imaginary)
        : base(elementType)
    {
        if (elementType is not (CompilerElementType.Complex64 or CompilerElementType.Complex128))
        {
            throw new ArgumentException("The element type must be a complex type.", nameof(elementType));
        }

        Real = real;
        Imaginary = imaginary;
    }

    public double Real { get; }

    public double Imaginary { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var complex = (CompilerComplexLiteral)other;
        var result = ElementType == complex.ElementType
            && Real.Equals(complex.Real)
            && Imaginary.Equals(complex.Imaginary);
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Real, Imaginary);
}

public sealed class CompilerStringLiteral : CompilerScalarLiteral
{
    public CompilerStringLiteral(string value)
        : base(CompilerElementType.String)
    {
        Value = value ?? throw new ArgumentNullException(nameof(value));
    }

    public string Value { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var result = string.Equals(Value, ((CompilerStringLiteral)other).Value, StringComparison.Ordinal);
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Value);
}

/// <summary>
/// Preserves the encoded payload of compact ONNX numeric element types.
/// </summary>
public sealed class CompilerPackedScalarLiteral : CompilerScalarLiteral
{
    public CompilerPackedScalarLiteral(CompilerElementType elementType, ulong encodedValue)
        : base(elementType)
    {
        if (elementType is not (
            CompilerElementType.BFloat16
            or CompilerElementType.Float8E4M3FN
            or CompilerElementType.Float8E4M3FNUZ
            or CompilerElementType.Float8E5M2
            or CompilerElementType.Float8E5M2FNUZ
            or CompilerElementType.Float4E2M1
            or CompilerElementType.Float8E8M0
            or CompilerElementType.UInt4
            or CompilerElementType.Int4
            or CompilerElementType.UInt2
            or CompilerElementType.Int2))
        {
            throw new ArgumentException("The element type must be a compact ONNX numeric type.", nameof(elementType));
        }

        EncodedValue = encodedValue;
    }

    public ulong EncodedValue { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var packed = (CompilerPackedScalarLiteral)other;
        var result = ElementType == packed.ElementType && EncodedValue == packed.EncodedValue;
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, EncodedValue);
}

/// <summary>Describes an external tensor payload without depending on ONNX protobuf types.</summary>
public sealed class CompilerExternalTensorData : IEquatable<CompilerExternalTensorData>
{
    public CompilerExternalTensorData(
        string location,
        long offset = 0,
        long? length = null,
        string? checksum = null
    )
    {
        if (string.IsNullOrWhiteSpace(location))
        {
            throw new ArgumentException("External tensor data location is required.", nameof(location));
        }

        if (offset < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        if (length < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        Location = location;
        Offset = offset;
        Length = length;
        Checksum = checksum;
    }

    public string Location { get; }

    public long Offset { get; }

    public long? Length { get; }

    public string? Checksum { get; }

    public bool Equals(CompilerExternalTensorData? other)
    {
        var result = other is not null
            && string.Equals(Location, other.Location, StringComparison.Ordinal)
            && Offset == other.Offset
            && Length == other.Length
            && string.Equals(Checksum, other.Checksum, StringComparison.Ordinal);
        return result;
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerExternalTensorData);

    public override int GetHashCode() => CompilerStructural.Combine(17, Location, Offset, Length, Checksum);
}

/// <summary>Tensor literal or initializer payload represented as typed scalar values.</summary>
public sealed class CompilerTensorLiteral : CompilerLiteral
{
    public CompilerTensorLiteral(
        CompilerElementType elementType,
        IEnumerable<CompilerDimension> dimensions,
        IEnumerable<CompilerScalarLiteral>? values = null,
        CompilerExternalTensorData? externalData = null
    )
    {
        ElementType = elementType;
        Dimensions = CompilerStructural.Copy(dimensions, nameof(dimensions));
        Values = CompilerStructural.Copy(values ?? Array.Empty<CompilerScalarLiteral>(), nameof(values));
        ExternalData = externalData;

        foreach (var value in Values)
        {
            if (value.ElementType != elementType)
            {
                throw new ArgumentException("Tensor literal values must use the tensor element type.", nameof(values));
            }
        }
    }

    public CompilerElementType ElementType { get; }

    public IReadOnlyList<CompilerDimension> Dimensions { get; }

    public IReadOnlyList<CompilerScalarLiteral> Values { get; }

    public CompilerExternalTensorData? ExternalData { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var tensor = (CompilerTensorLiteral)other;
        var result = ElementType == tensor.ElementType
            && CompilerStructural.SequenceEqual(Dimensions, tensor.Dimensions)
            && CompilerStructural.SequenceEqual(Values, tensor.Values)
            && EqualityComparer<CompilerExternalTensorData?>.Default.Equals(ExternalData, tensor.ExternalData);
        return result;
    }

    protected override int GetHashCodeCore()
    {
        return CompilerStructural.Combine(
            17,
            ElementType,
            CompilerStructural.GetHashCode(Dimensions),
            CompilerStructural.GetHashCode(Values),
            ExternalData);
    }
}

public sealed class CompilerArrayLiteral : CompilerLiteral
{
    public CompilerArrayLiteral(
        IEnumerable<CompilerLiteral> items,
        string? itemType = null
    )
    {
        Items = CompilerStructural.Copy(items, nameof(items));
        ItemType = itemType;
    }

    public IReadOnlyList<CompilerLiteral> Items { get; }

    /// <summary>Gets the source element type marker, including for an empty ONNX array.</summary>
    public string? ItemType { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var array = (CompilerArrayLiteral)other;
        var result = CompilerStructural.SequenceEqual(Items, array.Items)
            && string.Equals(ItemType, array.ItemType, StringComparison.Ordinal);
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(
        17,
        CompilerStructural.GetHashCode(Items),
        ItemType);
}

public sealed class CompilerTupleLiteral : CompilerLiteral
{
    public CompilerTupleLiteral(IEnumerable<CompilerLiteral> items)
    {
        Items = CompilerStructural.Copy(items, nameof(items));
    }

    public IReadOnlyList<CompilerLiteral> Items { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var result = CompilerStructural.SequenceEqual(Items, ((CompilerTupleLiteral)other).Items);
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.GetHashCode(Items);
}

/// <summary>Compiler-owned graph-valued ONNX attribute.</summary>
public sealed class CompilerGraphLiteral : CompilerLiteral
{
    public CompilerGraphLiteral(CompilerComputationTree graph)
    {
        CompilerStructural.RequireNotNull(graph, nameof(graph));
        Graph = graph;
    }

    public CompilerComputationTree Graph { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var result = EqualityComparer<CompilerComputationTree>.Default.Equals(
            Graph,
            ((CompilerGraphLiteral)other).Graph);
        return result;
    }

    protected override int GetHashCodeCore() => Graph.GetHashCode();
}

/// <summary>Compiler-owned sparse tensor attribute or initializer payload.</summary>
public sealed class CompilerSparseTensorLiteral : CompilerLiteral
{
    public CompilerSparseTensorLiteral(
        IEnumerable<CompilerDimension> dimensions,
        CompilerTensorLiteral values,
        CompilerTensorLiteral indices
    )
    {
        Dimensions = CompilerStructural.Copy(dimensions, nameof(dimensions));
        CompilerStructural.RequireNotNull(values, nameof(values));
        CompilerStructural.RequireNotNull(indices, nameof(indices));
        Values = values;
        Indices = indices;
    }

    public IReadOnlyList<CompilerDimension> Dimensions { get; }

    public CompilerTensorLiteral Values { get; }

    public CompilerTensorLiteral Indices { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var sparse = (CompilerSparseTensorLiteral)other;
        var result = CompilerStructural.SequenceEqual(Dimensions, sparse.Dimensions)
            && EqualityComparer<CompilerTensorLiteral>.Default.Equals(Values, sparse.Values)
            && EqualityComparer<CompilerTensorLiteral>.Default.Equals(Indices, sparse.Indices);
        return result;
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(
        17,
        CompilerStructural.GetHashCode(Dimensions),
        Values,
        Indices);
}

/// <summary>Compiler-owned ONNX TypeProto attribute payload.</summary>
public sealed class CompilerTypeLiteral : CompilerLiteral
{
    public CompilerTypeLiteral(CompilerType value)
    {
        CompilerStructural.RequireNotNull(value, nameof(value));
        Value = value;
    }

    public CompilerType Value { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        var result = EqualityComparer<CompilerType>.Default.Equals(Value, ((CompilerTypeLiteral)other).Value);
        return result;
    }

    protected override int GetHashCodeCore() => Value.GetHashCode();
}

/// <summary>Normalized operator attribute owned by the compiler intermediate representation.</summary>
public sealed class CompilerAttribute : IEquatable<CompilerAttribute>
{
    public CompilerAttribute(string name, CompilerLiteral value)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Attribute name cannot be empty.", nameof(name));
        }

        CompilerStructural.RequireNotNull(value, nameof(value));
        Name = name;
        Value = value;
    }

    public string Name { get; }

    public CompilerLiteral Value { get; }

    public bool Equals(CompilerAttribute? other)
    {
        var result = other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && EqualityComparer<CompilerLiteral>.Default.Equals(Value, other.Value);
        return result;
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerAttribute);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, Value);
}
