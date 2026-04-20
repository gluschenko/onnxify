namespace Onnxify.Safetensors;

public abstract record TensorIndexer;

public sealed record SelectTensorIndexer(ulong Index) : TensorIndexer;

public sealed record NarrowTensorIndexer(TensorBounds Start, TensorBounds End) : TensorIndexer;
