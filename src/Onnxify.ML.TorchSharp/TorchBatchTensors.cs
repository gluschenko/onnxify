using Tensor = global::TorchSharp.torch.Tensor;

namespace Onnxify.ML.TorchSharp;

/// <summary>
/// Holds tensors created during collation before they are wrapped into a <see cref="TorchMiniBatch{TSample}"/>.
/// </summary>
public sealed class TorchBatchTensors
{
    /// <summary>
    /// Initializes a new tensor bundle for a batch.
    /// </summary>
    public TorchBatchTensors(
        Tensor inputs,
        Tensor? targets = null,
        IReadOnlyDictionary<string, Tensor>? additionalTensors = null)
    {
        Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        Targets = targets;
        AdditionalTensors = additionalTensors ?? new Dictionary<string, Tensor>(StringComparer.Ordinal);
    }

    /// <summary>
    /// Gets the model input tensor.
    /// </summary>
    public Tensor Inputs { get; }

    /// <summary>
    /// Gets the optional supervision target tensor.
    /// </summary>
    public Tensor? Targets { get; }

    /// <summary>
    /// Gets additional named tensors produced during collation.
    /// </summary>
    public IReadOnlyDictionary<string, Tensor> AdditionalTensors { get; }
}
