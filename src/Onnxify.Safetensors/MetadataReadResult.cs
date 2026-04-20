namespace Onnxify.Safetensors;

public readonly record struct MetadataReadResult(int HeaderLength, Metadata Metadata);
