namespace Onnxify.Safetensors;

public readonly record struct TensorBounds(ulong? Value, bool Inclusive)
{
    public static TensorBounds Unbounded() => new(null, true);
    public static TensorBounds Included(ulong value) => new(value, true);
    public static TensorBounds Excluded(ulong value) => new(value, false);
}
