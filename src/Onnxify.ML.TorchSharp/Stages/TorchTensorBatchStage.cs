using Onnxify.ML.Stages;

namespace Onnxify.ML.TorchSharp.Stages;

/// <summary>
/// Collates samples into tensor-backed Torch mini-batches.
/// </summary>
public class TorchTensorBatchStage<TSample> : BatchingStage<TSample, TorchMiniBatch<TSample>>
{
    private readonly Func<IReadOnlyList<TSample>, PipelineContext, CancellationToken, ValueTask<TorchBatchTensors>> _collate;

    /// <summary>
    /// Initializes the stage from an asynchronous collate function with access to the pipeline context.
    /// </summary>
    public TorchTensorBatchStage(
        int batchSize,
        Func<IReadOnlyList<TSample>, PipelineContext, CancellationToken, ValueTask<TorchBatchTensors>> collate,
        bool includeIncompleteBatch = true,
        PipelineStageOptions? options = null)
        : base(batchSize, includeIncompleteBatch, options ?? new PipelineStageOptions
        {
            Category = PipelineStageCategories.BATCHING,
            Name = "torch-collate"
        })
    {
        _collate = collate ?? throw new ArgumentNullException(nameof(collate));
    }

    /// <summary>
    /// Initializes the stage from an asynchronous collate function that does not need the pipeline context.
    /// </summary>
    public TorchTensorBatchStage(
        int batchSize,
        Func<IReadOnlyList<TSample>, CancellationToken, ValueTask<TorchBatchTensors>> collate,
        bool includeIncompleteBatch = true,
        PipelineStageOptions? options = null)
        : this(
            batchSize,
            (samples, _, token) => collate(samples, token),
            includeIncompleteBatch,
            options)
    {
        ArgumentNullException.ThrowIfNull(collate);
    }

    protected override async ValueTask<TorchMiniBatch<TSample>> CreateBatchAsync(
        IReadOnlyList<TSample> batchItems,
        int batchIndex,
        bool isPartialBatch,
        PipelineContext context,
        CancellationToken token)
    {
        var tensors = await _collate(batchItems, context, token);
        var batch = new MiniBatch<TSample>
        {
            Items = batchItems,
            BatchIndex = batchIndex,
            IsPartialBatch = isPartialBatch
        };

        return new TorchMiniBatch<TSample>(
            batch,
            tensors.Inputs,
            tensors.Targets,
            tensors.AdditionalTensors
        );
    }

    protected override int? CalculateOutputCount(int inputCount)
    {
        if (inputCount == 0)
        {
            return 0;
        }

        var fullBatches = inputCount / BatchSize;
        var hasRemainder = inputCount % BatchSize != 0;

        if (hasRemainder && IncludeIncompleteBatch)
        {
            return fullBatches + 1;
        }

        return fullBatches;
    }
}
