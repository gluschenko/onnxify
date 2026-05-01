using Tensor = global::TorchSharp.torch.Tensor;

namespace Onnxify.ML.TorchSharp;

public sealed class TorchBatchTensors
{
    public TorchBatchTensors(
        Tensor inputs,
        Tensor? targets = null,
        IReadOnlyDictionary<string, Tensor>? additionalTensors = null)
    {
        Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        Targets = targets;
        AdditionalTensors = additionalTensors ?? new Dictionary<string, Tensor>(StringComparer.Ordinal);
    }

    public Tensor Inputs { get; }

    public Tensor? Targets { get; }

    public IReadOnlyDictionary<string, Tensor> AdditionalTensors { get; }
}
