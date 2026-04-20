namespace Onnxify.Safetensors;

public sealed class SafetensorException : Exception
{
    public SafetensorErrorCode Code { get; }
    public string? TensorName { get; }
    public DataType? DataType { get; }
    public IReadOnlyList<ulong>? Shape { get; }
    public ulong? ByteLength { get; }

    private SafetensorException(
        SafetensorErrorCode code,
        string message,
        Exception? innerException = null,
        string? tensorName = null,
        DataType? dtype = null,
        IReadOnlyList<ulong>? shape = null,
        ulong? byteLength = null)
        : base(message, innerException)
    {
        Code = code;
        TensorName = tensorName;
        DataType = dtype;
        Shape = shape;
        ByteLength = byteLength;
    }

    public static SafetensorException InvalidHeader(Exception innerException)
        => new(SafetensorErrorCode.InvalidHeader, $"invalid UTF-8 in header: {innerException.Message}", innerException);

    public static SafetensorException InvalidHeaderDeserialization(Exception innerException)
        => new(SafetensorErrorCode.InvalidHeaderDeserialization, $"invalid JSON in header: {innerException.Message}", innerException);

    public static SafetensorException JsonError(Exception innerException)
        => new(SafetensorErrorCode.JsonError, $"JSON error: {innerException.Message}", innerException);

    public static SafetensorException HeaderTooLarge()
        => new(SafetensorErrorCode.HeaderTooLarge, "header too large");

    public static SafetensorException HeaderTooSmall()
        => new(SafetensorErrorCode.HeaderTooSmall, "header too small");

    public static SafetensorException InvalidHeaderLength()
        => new(SafetensorErrorCode.InvalidHeaderLength, "invalid header length");

    public static SafetensorException TensorNotFound(string name)
        => new(SafetensorErrorCode.TensorNotFound, $"tensor `{name}` not found", tensorName: name);

    public static SafetensorException TensorInvalidInfo()
        => new(SafetensorErrorCode.TensorInvalidInfo, "invalid shape, data type, or offset for tensor");

    public static SafetensorException InvalidOffset(string name)
        => new(SafetensorErrorCode.InvalidOffset, $"invalid offset for tensor `{name}`", tensorName: name);

    public static SafetensorException IoError(IOException innerException)
        => new(SafetensorErrorCode.IoError, $"I/O error: {innerException.Message}", innerException);

    public static SafetensorException InvalidTensorView(DataType dtype, IReadOnlyList<ulong> shape, ulong byteLength)
        => new(
            SafetensorErrorCode.InvalidTensorView,
            $"tensor of type {dtype.ToWireName()} and shape ({string.Join(", ", shape)}) can't be created from {byteLength} bytes",
            dtype: dtype,
            shape: shape.ToArray(),
            byteLength: byteLength);

    public static SafetensorException MetadataIncompleteBuffer()
        => new(SafetensorErrorCode.MetadataIncompleteBuffer, "incomplete metadata, file not fully covered");

    public static SafetensorException ValidationOverflow()
        => new(SafetensorErrorCode.ValidationOverflow, "overflow computing buffer size from shape and/or element type");

    public static SafetensorException MisalignedSlice()
        => new(SafetensorErrorCode.MisalignedSlice, "The slice is slicing for subbytes dtypes, and the slice does not end up at a byte boundary, this is invalid.");
}
