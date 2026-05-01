namespace Onnxify.ML.Stages;

public class MiniBatchStage<TInput> : BatchingStage<TInput, MiniBatch<TInput>>
{
    public MiniBatchStage(
        int batchSize,
        bool includeIncompleteBatch = true,
        PipelineStageOptions? options = null)
        : base(batchSize, includeIncompleteBatch, options)
    {
    }

    protected override ValueTask<MiniBatch<TInput>> CreateBatchAsync(
        IReadOnlyList<TInput> batchItems,
        int batchIndex,
        bool isPartialBatch,
        PipelineContext context,
        CancellationToken token)
    {
        var batch = new MiniBatch<TInput>
        {
            Items = batchItems,
            BatchIndex = batchIndex,
            IsPartialBatch = isPartialBatch
        };

        return ValueTask.FromResult(batch);
    }
}
