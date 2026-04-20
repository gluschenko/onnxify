namespace Onnxify.Safetensors;

internal readonly record struct PreparedSafetensorsData(
    byte[] HeaderBytes,
    List<KeyValuePair<string, TensorView>> Tensors);
