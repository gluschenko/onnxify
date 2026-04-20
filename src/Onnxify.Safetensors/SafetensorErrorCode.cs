namespace Onnxify.Safetensors;

/// <summary>
/// Identifies the normalized failure category for safetensors format, validation, and I/O errors in the managed port.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>SafeTensorError</c>.
/// </remarks>
public enum SafetensorErrorCode
{
    /// <summary>The header bytes are not valid UTF-8.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidHeader</c>.</remarks>
    InvalidHeader = 1,
    /// <summary>The header is valid text but not valid safetensors JSON.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidHeaderDeserialization</c>.</remarks>
    InvalidHeaderDeserialization = 2,
    /// <summary>The declared header exceeds the maximum supported size.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::HeaderTooLarge</c>.</remarks>
    HeaderTooLarge = 3,
    /// <summary>The input buffer is too short to contain the safetensors header length prefix.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::HeaderTooSmall</c>.</remarks>
    HeaderTooSmall = 4,
    /// <summary>The declared header length does not fit inside the provided buffer.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidHeaderLength</c>.</remarks>
    InvalidHeaderLength = 5,
    /// <summary>A requested tensor name does not exist in the archive.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::TensorNotFound</c>.</remarks>
    TensorNotFound = 6,
    /// <summary>A tensor entry has inconsistent shape, data type, or byte offsets.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::TensorInvalidInfo</c>.</remarks>
    TensorInvalidInfo = 7,
    /// <summary>A tensor entry's offsets are not contiguous or otherwise violate safetensors layout rules.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidOffset</c>.</remarks>
    InvalidOffset = 8,
    /// <summary>A filesystem read or write operation failed.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::IoError</c>.</remarks>
    IoError = 9,
    /// <summary>JSON serialization or parsing failed outside the dedicated header-deserialization path.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::JsonError</c>.</remarks>
    JsonError = 10,
    /// <summary>A tensor view cannot be formed because the supplied byte count does not match shape and data type.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidTensorView</c>.</remarks>
    InvalidTensorView = 11,
    /// <summary>The tensor offsets do not cover the data section exactly.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::MetadataIncompleteBuffer</c>.</remarks>
    MetadataIncompleteBuffer = 12,
    /// <summary>An arithmetic overflow occurred while computing tensor sizes from shape and data type.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::ValidationOverflow</c>.</remarks>
    ValidationOverflow = 13,
    /// <summary>A sub-byte tensor slice does not land on byte boundaries.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::MisalignedSlice</c>.</remarks>
    MisalignedSlice = 14,
}
