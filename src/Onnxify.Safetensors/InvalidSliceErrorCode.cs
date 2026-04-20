namespace Onnxify.Safetensors;

/// <summary>
/// Identifies the high-level reason why tensor slicing failed in the managed safetensors slice engine.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
/// Original Rust entity: <c>InvalidSlice</c>.
/// </remarks>
public enum InvalidSliceErrorCode
{
    /// <summary>The caller provided more indexers than there are dimensions in the tensor.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>. Original Rust entity: <c>InvalidSlice::TooManySlices</c>.</remarks>
    TooManySlices = 1,
    /// <summary>The requested slice starts or ends outside the size of the addressed dimension.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>. Original Rust entity: <c>InvalidSlice::SliceOutOfRange</c>.</remarks>
    SliceOutOfRange = 2,
    /// <summary>The requested slice would cut through a sub-byte element at a non-byte boundary.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>. Original Rust entity: <c>InvalidSlice::MisalignedSlice</c>.</remarks>
    MisalignedSlice = 3,
}
