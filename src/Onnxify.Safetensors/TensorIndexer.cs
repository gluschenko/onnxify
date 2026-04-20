namespace Onnxify.Safetensors;

/// <summary>
/// Represents a single logical indexing operation that can be applied to one tensor dimension.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
/// Original Rust entity: <c>TensorIndexer</c>.
/// </remarks>
public abstract record TensorIndexer;

/// <summary>
/// Selects a single element from a dimension and removes that dimension from the resulting sliced shape.
/// </summary>
/// <param name="Index">The zero-based element index to select.</param>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
/// Original Rust entity: <c>TensorIndexer::Select</c>.
/// </remarks>
public sealed record SelectTensorIndexer(ulong Index) : TensorIndexer;

/// <summary>
/// Narrows a dimension to a range bounded by optional inclusive or exclusive start and end markers.
/// </summary>
/// <param name="Start">The lower bound of the range.</param>
/// <param name="End">The upper bound of the range.</param>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
/// Original Rust entity: <c>TensorIndexer::Narrow</c>.
/// </remarks>
public sealed record NarrowTensorIndexer(TensorBounds Start, TensorBounds End) : TensorIndexer;
