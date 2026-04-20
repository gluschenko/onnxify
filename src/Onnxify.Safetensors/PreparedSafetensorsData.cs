namespace Onnxify.Safetensors;

/// <summary>
/// Bundles the serialized header bytes with the tensor views ordered for deterministic safetensors serialization.
/// </summary>
/// <param name="HeaderBytes">The aligned JSON header bytes ready to be prefixed and written.</param>
/// <param name="Tensors">The tensor views in the exact order they must be emitted to preserve upstream behavior.</param>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: local C# helper extracted from <c>prepare</c> and <c>PreparedData</c>.
/// </remarks>
internal readonly record struct PreparedSafetensorsData(
    byte[] HeaderBytes,
    List<KeyValuePair<string, TensorView>> Tensors);
