using Tensor = global::TorchSharp.torch.Tensor;

namespace Onnxify.ML.TorchSharp;

public sealed class TorchMiniBatch<TSample> : IDisposable
{
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

    public MiniBatch<TSample> Batch { get; }

    public IReadOnlyList<TSample> Samples => Batch.Items;

    public int BatchIndex => Batch.BatchIndex;

    public bool IsPartialBatch => Batch.IsPartialBatch;

    public int Count => Batch.Count;

    public Tensor Inputs { get; }

    public Tensor? Targets { get; }

    public IReadOnlyDictionary<string, Tensor> AdditionalTensors { get; }

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
