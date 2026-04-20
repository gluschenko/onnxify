namespace Onnxify.Safetensors;

/// <summary>
/// Represents the half-open byte range occupied by a tensor inside the safetensors payload section.
/// </summary>
/// <param name="Start">The inclusive start byte offset relative to the payload section.</param>
/// <param name="End">The exclusive end byte offset relative to the payload section.</param>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>TensorInfo.data_offsets</c>.
/// </remarks>
public readonly record struct TensorDataOffsets(ulong Start, ulong End);
