namespace Onnxify.Safetensors;

/// <summary>
/// Describes one tensor entry from the safetensors header, including its element type, logical shape, and payload offsets.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>TensorInfo</c>.
/// </remarks>
public sealed class TensorInfo
{
    /// <summary>
    /// Gets or initializes the element type used to interpret the tensor payload.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>TensorInfo.dtype</c>.
    /// </remarks>
    public required DataType DataType { get; init; }

    /// <summary>
    /// Gets or initializes the logical tensor shape.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>TensorInfo.shape</c>.
    /// </remarks>
    public required ulong[] Shape { get; init; }

    /// <summary>
    /// Gets or initializes the half-open payload byte range occupied by the tensor.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>TensorInfo.data_offsets</c>.
    /// </remarks>
    public required TensorDataOffsets DataOffsets { get; init; }
}
