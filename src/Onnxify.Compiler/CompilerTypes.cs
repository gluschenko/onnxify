namespace Onnxify.Compiler;

/// <summary>Scalar element types understood by the compiler IR.</summary>
public enum CompilerElementType
{
    Unknown = 0,
    Boolean = 1,
    Int8 = 2,
    UInt8 = 3,
    Int16 = 4,
    UInt16 = 5,
    Int32 = 6,
    UInt32 = 7,
    Int64 = 8,
    UInt64 = 9,
    Float16 = 10,
    BFloat16 = 11,
    Float32 = 12,
    Float64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    String = 16,
}

/// <summary>Kind of tensor dimension represented by the IR.</summary>
public enum CompilerDimensionKind
{
    Unknown = 0,
    Fixed = 1,
    Symbolic = 2,
}

/// <summary>Base class for compiler-owned value type descriptors.</summary>
public abstract class CompilerType : IEquatable<CompilerType>
{
    protected CompilerType(string? denotation = null)
    {
        Denotation = denotation ?? string.Empty;
    }

    public string Denotation { get; }

    public bool Equals(CompilerType? other)
    {
        return other is not null
            && GetType() == other.GetType()
            && EqualsCore(other);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerType);

    public override int GetHashCode() => GetHashCodeCore();

    protected abstract bool EqualsCore(CompilerType other);

    protected abstract int GetHashCodeCore();
}

/// <summary>Base class for fixed, symbolic, and unknown dimensions.</summary>
public abstract class CompilerDimension : IEquatable<CompilerDimension>
{
    protected CompilerDimension(CompilerDimensionKind kind, string? denotation = null)
    {
        Kind = kind;
        Denotation = denotation ?? string.Empty;
    }

    public CompilerDimensionKind Kind { get; }

    public string Denotation { get; }

    public bool Equals(CompilerDimension? other)
    {
        return other is not null
            && GetType() == other.GetType()
            && EqualsCore(other);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerDimension);

    public override int GetHashCode() => GetHashCodeCore();

    protected abstract bool EqualsCore(CompilerDimension other);

    protected abstract int GetHashCodeCore();
}

public sealed class CompilerFixedDimension : CompilerDimension
{
    public CompilerFixedDimension(long value, string? denotation = null)
        : base(CompilerDimensionKind.Fixed, denotation)
    {
        if (value < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(value));
        }

        Value = value;
    }

    public long Value { get; }

    protected override bool EqualsCore(CompilerDimension other)
    {
        return Value == ((CompilerFixedDimension)other).Value
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Kind, Denotation, Value);
}

public sealed class CompilerSymbolicDimension : CompilerDimension
{
    public CompilerSymbolicDimension(string name, string? denotation = null)
        : base(CompilerDimensionKind.Symbolic, denotation)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("A symbolic dimension name is required.", nameof(name));
        }

        Name = name;
    }

    public string Name { get; }

    protected override bool EqualsCore(CompilerDimension other)
    {
        return string.Equals(Name, ((CompilerSymbolicDimension)other).Name, StringComparison.Ordinal)
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Kind, Denotation, Name);
}

public sealed class CompilerUnknownDimension : CompilerDimension
{
    public CompilerUnknownDimension(string? denotation = null)
        : base(CompilerDimensionKind.Unknown, denotation)
    {
    }

    protected override bool EqualsCore(CompilerDimension other)
    {
        return string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Kind, Denotation);
}

public sealed class CompilerScalarType : CompilerType
{
    public CompilerScalarType(CompilerElementType elementType, string? denotation = null)
        : base(denotation)
    {
        ElementType = elementType;
    }

    public CompilerElementType ElementType { get; }

    protected override bool EqualsCore(CompilerType other)
    {
        return ElementType == ((CompilerScalarType)other).ElementType
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Denotation);
}

public sealed class CompilerTensorType : CompilerType
{
    public CompilerTensorType(
        CompilerElementType elementType,
        IEnumerable<CompilerDimension>? dimensions,
        string? denotation = null
    ) : base(denotation)
    {
        ElementType = elementType;
        Dimensions = CompilerStructural.CopyNullable(dimensions, nameof(dimensions));
    }

    public CompilerElementType ElementType { get; }

    /// <summary>Null means unknown rank; an empty list means a known scalar-shaped tensor.</summary>
    public IReadOnlyList<CompilerDimension>? Dimensions { get; }

    public bool HasKnownRank => Dimensions is not null;

    protected override bool EqualsCore(CompilerType other)
    {
        var tensor = (CompilerTensorType)other;
        return ElementType == tensor.ElementType
            && CompilerStructural.SequenceEqual(Dimensions, tensor.Dimensions)
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore()
    {
        return CompilerStructural.Combine(17, ElementType, Denotation, CompilerStructural.GetHashCode(Dimensions));
    }
}

public sealed class CompilerOptionalType : CompilerType
{
    public CompilerOptionalType(CompilerType elementType, string? denotation = null)
        : base(denotation)
    {
        CompilerStructural.RequireNotNull(elementType, nameof(elementType));
        ElementType = elementType;
    }

    public CompilerType ElementType { get; }

    protected override bool EqualsCore(CompilerType other)
    {
        return EqualityComparer<CompilerType>.Default.Equals(ElementType, ((CompilerOptionalType)other).ElementType)
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Denotation);
}

public sealed class CompilerSequenceType : CompilerType
{
    public CompilerSequenceType(CompilerType elementType, string? denotation = null)
        : base(denotation)
    {
        CompilerStructural.RequireNotNull(elementType, nameof(elementType));
        ElementType = elementType;
    }

    public CompilerType ElementType { get; }

    protected override bool EqualsCore(CompilerType other)
    {
        return EqualityComparer<CompilerType>.Default.Equals(ElementType, ((CompilerSequenceType)other).ElementType)
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, ElementType, Denotation);
}

public sealed class CompilerTupleType : CompilerType
{
    public CompilerTupleType(IEnumerable<CompilerType> elementTypes, string? denotation = null)
        : base(denotation)
    {
        ElementTypes = CompilerStructural.Copy(elementTypes, nameof(elementTypes));
    }

    public IReadOnlyList<CompilerType> ElementTypes { get; }

    protected override bool EqualsCore(CompilerType other)
    {
        var tuple = (CompilerTupleType)other;
        return CompilerStructural.SequenceEqual(ElementTypes, tuple.ElementTypes)
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Denotation, CompilerStructural.GetHashCode(ElementTypes));
}

public sealed class CompilerMapType : CompilerType
{
    public CompilerMapType(CompilerElementType keyType, CompilerType valueType, string? denotation = null)
        : base(denotation)
    {
        CompilerStructural.RequireNotNull(valueType, nameof(valueType));
        KeyType = keyType;
        ValueType = valueType;
    }

    public CompilerElementType KeyType { get; }

    public CompilerType ValueType { get; }

    protected override bool EqualsCore(CompilerType other)
    {
        var map = (CompilerMapType)other;
        return KeyType == map.KeyType
            && EqualityComparer<CompilerType>.Default.Equals(ValueType, map.ValueType)
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, KeyType, ValueType, Denotation);
}

public sealed class CompilerSparseTensorType : CompilerType
{
    public CompilerSparseTensorType(
        CompilerElementType elementType,
        IEnumerable<CompilerDimension>? dimensions,
        string? denotation = null
    ) : base(denotation)
    {
        ElementType = elementType;
        Dimensions = CompilerStructural.CopyNullable(dimensions, nameof(dimensions));
    }

    public CompilerElementType ElementType { get; }

    public IReadOnlyList<CompilerDimension>? Dimensions { get; }

    protected override bool EqualsCore(CompilerType other)
    {
        var sparse = (CompilerSparseTensorType)other;
        return ElementType == sparse.ElementType
            && CompilerStructural.SequenceEqual(Dimensions, sparse.Dimensions)
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore()
    {
        return CompilerStructural.Combine(17, ElementType, Denotation, CompilerStructural.GetHashCode(Dimensions));
    }
}

public sealed class CompilerOpaqueType : CompilerType
{
    public CompilerOpaqueType(string domain, string name, string? denotation = null)
        : base(denotation)
    {
        Domain = domain ?? string.Empty;
        Name = name ?? string.Empty;
    }

    public string Domain { get; }

    public string Name { get; }

    protected override bool EqualsCore(CompilerType other)
    {
        var opaque = (CompilerOpaqueType)other;
        return string.Equals(Domain, opaque.Domain, StringComparison.Ordinal)
            && string.Equals(Name, opaque.Name, StringComparison.Ordinal)
            && string.Equals(Denotation, other.Denotation, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Domain, Name, Denotation);
}
