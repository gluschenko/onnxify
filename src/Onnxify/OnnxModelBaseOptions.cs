using Onnxify.Data;

namespace Onnxify;

/// <summary>
/// Shared options for model loading, creation, and tensor-data resolution.
/// </summary>
public class OnnxModelBaseOptions
{
    /// <summary>
    /// Gets the base directory used to resolve relative ONNX external-data locations.
    /// </summary>
    /// <remarks>
    /// <see cref="OnnxModel.FromFile"/> sets this to the directory containing the loaded model. Set it manually when deserializing tensors or building models that reference external files.
    /// </remarks>
    public string? DataLocation { get; init; } = null;

    /// <summary>
    /// Gets the provider used to read external tensor payloads.
    /// </summary>
    /// <remarks>
    /// Override this when tensor bytes live outside the local filesystem or require custom authentication, decompression, or storage lookup.
    /// </remarks>
    public ExternalDataProvider DataReader { get; init; } = OnnxExternalDataProvider.Instance;

    /// <summary>
    /// Gets the provider reserved for writing external tensor payloads.
    /// </summary>
    /// <remarks>
    /// Current serialization embeds tensor data by default; this option exists for callers that need to keep read/write configuration together as external-data writing support grows.
    /// </remarks>
    public ExternalDataProvider DataWriter { get; init; } = OnnxExternalDataProvider.Instance;

    public NodeTypeResolutionStrategy NodeTypeResolutionStrategy { get; init; } = NodeTypeResolutionStrategy.FailFast;

    /// <summary>
    /// Gets the imported opset versions available while materializing graph nodes from protobuf.
    /// </summary>
    internal IReadOnlyDictionary<string, long>? OpsetImports { get; init; }
}

public enum NodeTypeResolutionStrategy
{
    FailFast = 1,
    IgnoreIncompatible = 2,
}
