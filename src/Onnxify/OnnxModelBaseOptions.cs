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
    public string? DataLocation { get; set; } = null;

    /// <summary>
    /// Gets the provider used to read external tensor payloads.
    /// </summary>
    /// <remarks>
    /// Override this when tensor bytes live outside the local filesystem or require custom authentication, decompression, or storage lookup.
    /// </remarks>
    public ExternalDataProvider DataReader { get; set; } = OnnxExternalDataProvider.Instance;

    /// <summary>
    /// Gets the provider reserved for writing external tensor payloads.
    /// </summary>
    /// <remarks>
    /// Current serialization embeds tensor data by default; this option exists for callers that need to keep read/write configuration together as external-data writing support grows.
    /// </remarks>
    public ExternalDataProvider DataWriter { get; set; } = OnnxExternalDataProvider.Instance;

    public NodeTypeResolutionStrategy NodeTypeResolutionStrategy { get; set; } = NodeTypeResolutionStrategy.FailFast;

    /// <summary>
    /// Gets the imported opset versions available while materializing graph nodes from protobuf.
    /// </summary>
    internal IReadOnlyDictionary<string, long>? OpsetImports { get; set; }
}

/// <summary>
/// Controls how loaded protobuf nodes are materialized into the <see cref="OnnxGraph"/> object model.
/// </summary>
/// <remarks>
/// Onnxify can expose a node either as a generated typed wrapper when the operator schema matches the current
/// library model, or as a generic <see cref="OnnxNode"/> that preserves the raw ONNX inputs, outputs, and attributes.
/// This setting only affects graph materialization while loading or cloning models; it does not change ONNX operator
/// semantics or runtime inference behavior.
/// </remarks>
public enum NodeTypeResolutionStrategy
{
    /// <summary>
    /// Throws when a loaded model contains an operator whose protobuf structure is incompatible with Onnxify's
    /// generated typed wrapper for that operator.
    /// </summary>
    /// <remarks>
    /// Use this mode when authoring or editing graphs through typed node APIs and you want schema drift to fail early.
    /// </remarks>
    FailFast = 1,

    /// <summary>
    /// Uses typed node wrappers when they are compatible, but falls back to generic <see cref="OnnxNode"/> instances
    /// for incompatible operators.
    /// </summary>
    /// <remarks>
    /// Use this mode when inspecting or round-tripping models from newer or older ONNX opsets while still taking
    /// advantage of typed wrappers where Onnxify can safely resolve them.
    /// </remarks>
    IgnoreIncompatible = 2,

    /// <summary>
    /// Always materializes loaded nodes as generic <see cref="OnnxNode"/> instances without attempting typed wrapper
    /// resolution.
    /// </summary>
    /// <remarks>
    /// Use this mode when exact preservation of raw ONNX node inputs and attributes is more important than typed API
    /// convenience, such as source generation, canonical graph sorting, diagnostics, and model reconstruction.
    /// </remarks>
    PreserveUntyped = 3,
}
