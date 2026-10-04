namespace Onnxify.Compiler;

/// <summary>Base class for typed compiler literals.</summary>
public abstract class CompilerLiteral : IEquatable<CompilerLiteral>
{
    public bool Equals(CompilerLiteral? other)
    {
        return other is not null
            && GetType() == other.GetType()
            && EqualsCore(other);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerLiteral);

    public override int GetHashCode() => GetHashCodeCore();

    protected abstract bool EqualsCore(CompilerLiteral other);

    protected abstract int GetHashCodeCore();
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
        return Value == ((CompilerBooleanLiteral)other).Value;
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
        return ElementType == integer.ElementType && Value == integer.Value;
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
        return ElementType == integer.ElementType && Value == integer.Value;
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
        return ElementType == floatingPoint.ElementType && Value.Equals(floatingPoint.Value);
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
        return ElementType == complex.ElementType
            && Real.Equals(complex.Real)
            && Imaginary.Equals(complex.Imaginary);
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
        return string.Equals(Value, ((CompilerStringLiteral)other).Value, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Value);
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
        return other is not null
            && string.Equals(Location, other.Location, StringComparison.Ordinal)
            && Offset == other.Offset
            && Length == other.Length
            && string.Equals(Checksum, other.Checksum, StringComparison.Ordinal);
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
        return ElementType == tensor.ElementType
            && CompilerStructural.SequenceEqual(Dimensions, tensor.Dimensions)
            && CompilerStructural.SequenceEqual(Values, tensor.Values)
            && EqualityComparer<CompilerExternalTensorData?>.Default.Equals(ExternalData, tensor.ExternalData);
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
    public CompilerArrayLiteral(IEnumerable<CompilerLiteral> items)
    {
        Items = CompilerStructural.Copy(items, nameof(items));
    }

    public IReadOnlyList<CompilerLiteral> Items { get; }

    protected override bool EqualsCore(CompilerLiteral other)
    {
        return CompilerStructural.SequenceEqual(Items, ((CompilerArrayLiteral)other).Items);
    }

    protected override int GetHashCodeCore() => CompilerStructural.GetHashCode(Items);
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
        return CompilerStructural.SequenceEqual(Items, ((CompilerTupleLiteral)other).Items);
    }

    protected override int GetHashCodeCore() => CompilerStructural.GetHashCode(Items);
}

/// <summary>Normalized operator attribute owned by the compiler IR.</summary>
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
        return other is not null
            && string.Equals(Name, other.Name, StringComparison.Ordinal)
            && EqualityComparer<CompilerLiteral>.Default.Equals(Value, other.Value);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerAttribute);

    public override int GetHashCode() => CompilerStructural.Combine(17, Name, Value);
}
