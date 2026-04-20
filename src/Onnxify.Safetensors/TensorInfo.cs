namespace Onnxify.Safetensors;

public sealed class TensorInfo
{
    public required DataType DataType { get; init; }
    public required ulong[] Shape { get; init; }
    public required TensorDataOffsets DataOffsets { get; init; }
}
