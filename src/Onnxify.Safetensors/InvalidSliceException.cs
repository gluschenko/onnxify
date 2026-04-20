namespace Onnxify.Safetensors;

public sealed class InvalidSliceException : Exception
{
    public InvalidSliceErrorCode Code { get; }
    public int? DimensionIndex { get; }
    public ulong? Asked { get; }
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

    public static InvalidSliceException TooManySlices()
    {
        return new(
            InvalidSliceErrorCode.TooManySlices,
            "more slicing indexes than dimensions in tensor"
        );
    }

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

    public static InvalidSliceException MisalignedSlice()
    {
        return new(
            InvalidSliceErrorCode.MisalignedSlice,
            "The slice is slicing for subbytes dtypes, and the slice does not end up at a byte boundary, this is invalid."
        );
    }
}
