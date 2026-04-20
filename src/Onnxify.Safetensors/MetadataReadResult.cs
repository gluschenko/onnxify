namespace Onnxify.Safetensors;

/// <summary>
/// Carries the parsed safetensors header length together with the validated metadata model.
/// </summary>
/// <param name="HeaderLength">The JSON header length in bytes, excluding the 8-byte length prefix.</param>
/// <param name="Metadata">The validated metadata parsed from the header.</param>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>SafeTensors::read_metadata</c> return tuple <c>(usize, Metadata)</c>.
/// </remarks>
public readonly record struct MetadataReadResult(int HeaderLength, Metadata Metadata);
