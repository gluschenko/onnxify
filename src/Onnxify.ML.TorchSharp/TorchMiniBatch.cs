using Tensor = global::TorchSharp.torch.Tensor;

namespace Onnxify.ML.TorchSharp;

/// <summary>
/// Wraps a logical mini-batch together with Torch tensors created for model execution.
/// </summary>
public sealed class TorchMiniBatch<TSample> : IDisposable
{
    /// <summary>
    /// Initializes a new tensor-backed mini-batch.
    /// </summary>
    public TorchMiniBatch(
        MiniBatch<TSample> batch,
        Tensor inputs,
        Tensor? targets = null,
        IReadOnlyDictionary<string, Tensor>? additionalTensors = null)
    {
        Batch = batch ?? throw new ArgumentNullException(nameof(batch));
        Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        Targets = targets;
        AdditionalTensors = additionalTensors ?? new Dictionary<string, Tensor>(StringComparer.Ordinal);
    }

    /// <summary>
    /// Gets the logical mini-batch metadata.
    /// </summary>
    public MiniBatch<TSample> Batch { get; }

    /// <summary>
    /// Gets the original samples contained in the batch.
    /// </summary>
    public IReadOnlyList<TSample> Samples => Batch.Items;

    /// <summary>
    /// Gets the zero-based batch index.
    /// </summary>
    public int BatchIndex => Batch.BatchIndex;

    /// <summary>
    /// Gets a value indicating whether the batch is smaller than the nominal batch size.
    /// </summary>
    public bool IsPartialBatch => Batch.IsPartialBatch;

    /// <summary>
    /// Gets the number of samples in the batch.
    /// </summary>
    public int Count => Batch.Count;

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

    /// <summary>
    /// Disposes all unique tensors owned by the mini-batch.
    /// </summary>
    public void Dispose()
    {
        var disposed = new HashSet<object>();

        DisposeTensor(Inputs);
        DisposeTensor(Targets);

        foreach (var tensor in AdditionalTensors.Values)
        {
            DisposeTensor(tensor);
        }

        void DisposeTensor(Tensor? tensor)
        {
            if (tensor is null || !disposed.Add(tensor))
            {
                return;
            }

            tensor.Dispose();
        }
    }
}
