namespace Onnxify.Safetensors;

/// <summary>
/// Represents a normalized slice bound pair after converting inclusive and exclusive user syntax into an executable range.
/// </summary>
/// <param name="Start">The inclusive start index within the dimension.</param>
/// <param name="Stop">The exclusive end index within the dimension.</param>
/// <param name="IsSelect">Indicates whether the original operation selected a single element instead of narrowing a range.</param>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
/// Original Rust entity: local C# helper extracted from the <c>SliceIterator::new</c> match over <c>TensorIndexer</c>.
/// </remarks>
internal readonly record struct ResolvedSlice(ulong Start, ulong Stop, bool IsSelect);
