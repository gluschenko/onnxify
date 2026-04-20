namespace Onnxify.Safetensors;

/// <summary>
/// Represents one side of a tensor range bound, including whether the bound is inclusive or exclusive.
/// </summary>
/// <param name="Value">The bound value, or <see langword="null"/> for an unbounded side.</param>
/// <param name="Inclusive"><see langword="true"/> when the bound is inclusive; otherwise the value is exclusive.</param>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
/// Original Rust entity: <c>core::ops::Bound&lt;usize&gt;</c> as consumed by <c>TensorIndexer::Narrow</c>.
/// </remarks>
public readonly record struct TensorBounds(ulong? Value, bool Inclusive)
{
    /// <summary>
    /// Creates an unbounded range side.
    /// </summary>
    /// <returns>A bound with no explicit limit.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>Bound::Unbounded</c>.
    /// </remarks>
    public static TensorBounds Unbounded() => new(null, true);

    /// <summary>
    /// Creates an inclusive range side.
    /// </summary>
    /// <param name="value">The included bound value.</param>
    /// <returns>An inclusive bound.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>Bound::Included</c>.
    /// </remarks>
    public static TensorBounds Included(ulong value) => new(value, true);

    /// <summary>
    /// Creates an exclusive range side.
    /// </summary>
    /// <param name="value">The excluded bound value.</param>
    /// <returns>An exclusive bound.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>Bound::Excluded</c>.
    /// </remarks>
    public static TensorBounds Excluded(ulong value) => new(value, false);
}
