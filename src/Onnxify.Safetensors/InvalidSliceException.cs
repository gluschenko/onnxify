namespace Onnxify.Safetensors;

/// <summary>
/// Represents an invalid tensor slicing request and preserves the same observable failure categories as upstream safetensors.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
/// Original Rust entity: <c>InvalidSlice</c>.
/// </remarks>
public sealed class InvalidSliceException : Exception
{
    /// <summary>Gets the normalized error code for the slicing failure.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>. Original Rust entity: <c>InvalidSlice</c>.</remarks>
    public InvalidSliceErrorCode Code { get; }

    /// <summary>Gets the zero-based dimension index involved in the failure when the error is dimension-specific.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>. Original Rust entity: <c>InvalidSlice::SliceOutOfRange.dim_index</c>.</remarks>
    public int? DimensionIndex { get; }

    /// <summary>Gets the offending index value when the slice is out of range.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>. Original Rust entity: <c>InvalidSlice::SliceOutOfRange.asked</c>.</remarks>
    public ulong? Asked { get; }

    /// <summary>Gets the size of the dimension that was exceeded when the slice is out of range.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>. Original Rust entity: <c>InvalidSlice::SliceOutOfRange.dim_size</c>.</remarks>
    public ulong? DimensionSize { get; }

    private InvalidSliceException(
        InvalidSliceErrorCode code,
        string message,
        int? dimensionIndex = null,
        ulong? asked = null,
        ulong? dimensionSize = null
    ) : base(message)
    {
        Code = code;
        DimensionIndex = dimensionIndex;
        Asked = asked;
        DimensionSize = dimensionSize;
    }

    /// <summary>
    /// Creates an exception that reports the caller supplied more slice indexers than the tensor rank permits.
    /// </summary>
    /// <returns>An exception mirroring the upstream <c>TooManySlices</c> error.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>InvalidSlice::TooManySlices</c>.
    /// </remarks>
    public static InvalidSliceException TooManySlices()
    {
        return new(
            InvalidSliceErrorCode.TooManySlices,
            "more slicing indexes than dimensions in tensor"
        );
    }

    /// <summary>
    /// Creates an exception describing an out-of-range slice bound for a specific tensor dimension.
    /// </summary>
    /// <param name="dimIndex">The zero-based dimension index that rejected the bound.</param>
    /// <param name="asked">The offending element index implied by the requested slice.</param>
    /// <param name="dimSize">The size of the addressed dimension.</param>
    /// <returns>An exception mirroring the upstream <c>SliceOutOfRange</c> error.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>InvalidSlice::SliceOutOfRange</c>.
    /// </remarks>
    public static InvalidSliceException SliceOutOfRange(int dimIndex, ulong asked, ulong dimSize)
    {
        return new(
            code: InvalidSliceErrorCode.SliceOutOfRange,
            message: $"index {asked} out of bounds for tensor dimension #{dimIndex} of size {dimSize}",
            dimensionIndex: dimIndex,
            asked: asked,
            dimensionSize: dimSize
        );
    }

    /// <summary>
    /// Creates an exception describing a slice over a sub-byte data type that is not aligned to byte boundaries.
    /// </summary>
    /// <returns>An exception mirroring the upstream <c>MisalignedSlice</c> error.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>InvalidSlice::MisalignedSlice</c>.
    /// </remarks>
    public static InvalidSliceException MisalignedSlice()
    {
        return new(
            InvalidSliceErrorCode.MisalignedSlice,
            "The slice is slicing for subbytes dtypes, and the slice does not end up at a byte boundary, this is invalid."
        );
    }
}
