namespace Onnxify.Safetensors;

/// <summary>
/// Represents a half-open byte range over a tensor payload buffer.
/// </summary>
/// <param name="Start">The inclusive start byte offset.</param>
/// <param name="Stop">The exclusive end byte offset.</param>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
/// Original Rust entity: local C# helper extracted from the <c>Vec&lt;(usize, usize)&gt;</c> held by <c>SliceIterator</c>.
/// </remarks>
internal readonly record struct SliceRange(int Start, int Stop);
