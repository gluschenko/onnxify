namespace Onnxify;

/// <summary>
/// Identifies one operator signature for runtime-profile validation.
/// </summary>
public sealed class OnnxOperatorSupport : IEquatable<OnnxOperatorSupport>
{
    /// <summary>
    /// Gets the ONNX operator domain; use an empty string for the default domain.
    /// </summary>
    public required string Domain { get; init; }

    /// <summary>
    /// Gets the ONNX operator type, such as <c>Add</c> or <c>Cast</c>.
    /// </summary>
    public required string OpType { get; init; }

    /// <inheritdoc />
    public bool Equals(OnnxOperatorSupport? other)
    {
        return other is not null
            && string.Equals(Domain, other.Domain, StringComparison.Ordinal)
            && string.Equals(OpType, other.OpType, StringComparison.Ordinal);
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        return obj is OnnxOperatorSupport other && Equals(other);
    }

    /// <inheritdoc />
    public override int GetHashCode()
    {
        return HashCode.Combine(Domain, OpType);
    }
}
