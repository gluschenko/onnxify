namespace Onnxify.Safetensors;

/// <summary>
/// Represents a safetensors format or validation failure and carries structured details that mirror the upstream Rust error variants.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>SafeTensorError</c>.
/// </remarks>
public sealed class SafeTensorException : Exception
{
    /// <summary>Gets the normalized safetensors error category.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError</c>.</remarks>
    public SafeTensorErrorCode Code { get; }

    /// <summary>Gets the tensor name involved in the failure when the upstream error carries one.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: name payload on <c>TensorNotFound</c> and <c>InvalidOffset</c>.</remarks>
    public string? TensorName { get; }

    /// <summary>Gets the data type involved in the failure when the upstream error carries one.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidTensorView.0</c>.</remarks>
    public DataType? DataType { get; }

    /// <summary>Gets the tensor shape involved in the failure when the upstream error carries one.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidTensorView.1</c>.</remarks>
    public IReadOnlyList<ulong>? Shape { get; }

    /// <summary>Gets the byte length involved in the failure when the upstream error carries one.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidTensorView.2</c>.</remarks>
    public ulong? ByteLength { get; }

    private SafeTensorException(
        SafeTensorErrorCode code,
        string message,
        Exception? innerException = null,
        string? tensorName = null,
        DataType? dtype = null,
        IReadOnlyList<ulong>? shape = null,
        ulong? byteLength = null
    ) : base(message, innerException)
    {
        Code = code;
        TensorName = tensorName;
        DataType = dtype;
        Shape = shape;
        ByteLength = byteLength;
    }

    /// <summary>Creates an error for invalid UTF-8 in the header bytes.</summary>
    /// <param name="innerException">The UTF-8 decoder failure.</param>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidHeader</c>.</remarks>
    public static SafeTensorException InvalidHeader(Exception innerException)
        => new(SafeTensorErrorCode.InvalidHeader, $"invalid UTF-8 in header: {innerException.Message}", innerException);

    /// <summary>Creates an error for a header that is text but not valid safetensors JSON.</summary>
    /// <param name="innerException">The JSON parser failure.</param>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidHeaderDeserialization</c>.</remarks>
    public static SafeTensorException InvalidHeaderDeserialization(Exception innerException)
        => new(SafeTensorErrorCode.InvalidHeaderDeserialization, $"invalid JSON in header: {innerException.Message}", innerException);

    /// <summary>Creates an error for general JSON serialization or deserialization failures.</summary>
    /// <param name="innerException">The underlying JSON failure.</param>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::JsonError</c>.</remarks>
    public static SafeTensorException JsonError(Exception innerException)
        => new(SafeTensorErrorCode.JsonError, $"JSON error: {innerException.Message}", innerException);

    /// <summary>Creates an error for a header that exceeds the allowed maximum size.</summary>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::HeaderTooLarge</c>.</remarks>
    public static SafeTensorException HeaderTooLarge()
        => new(SafeTensorErrorCode.HeaderTooLarge, "header too large");

    /// <summary>Creates an error for a buffer that is too short to contain the mandatory length prefix.</summary>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::HeaderTooSmall</c>.</remarks>
    public static SafeTensorException HeaderTooSmall()
        => new(SafeTensorErrorCode.HeaderTooSmall, "header too small");

    /// <summary>Creates an error for a declared header length that does not fit inside the supplied buffer.</summary>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidHeaderLength</c>.</remarks>
    public static SafeTensorException InvalidHeaderLength()
        => new(SafeTensorErrorCode.InvalidHeaderLength, "invalid header length");

    /// <summary>Creates an error for a missing tensor lookup.</summary>
    /// <param name="name">The missing tensor name.</param>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::TensorNotFound</c>.</remarks>
    public static SafeTensorException TensorNotFound(string name)
        => new(SafeTensorErrorCode.TensorNotFound, $"tensor `{name}` not found", tensorName: name);

    /// <summary>Creates an error for inconsistent tensor metadata.</summary>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::TensorInvalidInfo</c>.</remarks>
    public static SafeTensorException TensorInvalidInfo()
        => new(SafeTensorErrorCode.TensorInvalidInfo, "invalid shape, data type, or offset for tensor");

    /// <summary>Creates an error for invalid or non-contiguous tensor offsets.</summary>
    /// <param name="name">The tensor name that owns the invalid offsets.</param>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidOffset</c>.</remarks>
    public static SafeTensorException InvalidOffset(string name)
        => new(SafeTensorErrorCode.InvalidOffset, $"invalid offset for tensor `{name}`", tensorName: name);

    /// <summary>Creates an error for filesystem I/O failures during safetensors reads or writes.</summary>
    /// <param name="innerException">The underlying I/O error.</param>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::IoError</c>.</remarks>
    public static SafeTensorException IoError(IOException innerException)
        => new(SafeTensorErrorCode.IoError, $"I/O error: {innerException.Message}", innerException);

    /// <summary>Creates an error for tensor data whose byte length does not match its shape and data type.</summary>
    /// <param name="dtype">The data type used for size computation.</param>
    /// <param name="shape">The tensor shape used for size computation.</param>
    /// <param name="byteLength">The actual byte count presented for the view.</param>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::InvalidTensorView</c>.</remarks>
    public static SafeTensorException InvalidTensorView(DataType dtype, IReadOnlyList<ulong> shape, ulong byteLength)
        => new(
            SafeTensorErrorCode.InvalidTensorView,
            $"tensor of type {dtype.ToWireName()} and shape ({string.Join(", ", shape)}) can't be created from {byteLength} bytes",
            dtype: dtype,
            shape: shape.ToArray(),
            byteLength: byteLength);

    /// <summary>Creates an error for archives whose tensor offsets do not cover the data section exactly.</summary>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::MetadataIncompleteBuffer</c>.</remarks>
    public static SafeTensorException MetadataIncompleteBuffer()
        => new(SafeTensorErrorCode.MetadataIncompleteBuffer, "incomplete metadata, file not fully covered");

    /// <summary>Creates an error for arithmetic overflow while validating shapes or byte lengths.</summary>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::ValidationOverflow</c>.</remarks>
    public static SafeTensorException ValidationOverflow()
        => new(SafeTensorErrorCode.ValidationOverflow, "overflow computing buffer size from shape and/or element type");

    /// <summary>Creates an error for sub-byte slicing that does not align to byte boundaries.</summary>
    /// <returns>A structured safetensors exception.</returns>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>SafeTensorError::MisalignedSlice</c>.</remarks>
    public static SafeTensorException MisalignedSlice()
        => new(SafeTensorErrorCode.MisalignedSlice, "The slice is slicing for subbytes dtypes, and the slice does not end up at a byte boundary, this is invalid.");
}
