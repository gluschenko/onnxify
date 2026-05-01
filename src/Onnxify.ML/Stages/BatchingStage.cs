using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

/// <summary>
/// Extensible batching primitive for classic mini-batches, bucketing and token-budget grouping.
/// </summary>
public abstract class BatchingStage<TInput, TBatch> : PipelineStage<TInput, TBatch>
{
    protected BatchingStage(
        int batchSize,
        bool includeIncompleteBatch = true,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions { Category = PipelineStageCategories.Batching })
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        BatchSize = batchSize;
        IncludeIncompleteBatch = includeIncompleteBatch;
    }

    public int BatchSize { get; }

    public bool IncludeIncompleteBatch { get; }

    public sealed override async IAsyncEnumerable<TBatch> ExecuteAsync(
        IEnumerable<TInput> input,
        PipelineContext context,
        [EnumeratorCancellation] CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(context);

        var total = input.TryGetNonEnumeratedCount(out var knownTotal)
            ? knownTotal
            : -1;
        var current = 0;
        var batchIndex = 0;
        var initialCapacity = BatchSize > 1024
            ? 1024
            : BatchSize;
        var buffer = new List<TInput>(initialCapacity);

        await ReportProgressAsync(current, total);

        foreach (var item in input)
        {
            token.ThrowIfCancellationRequested();

            buffer.Add(item);
            current++;

            if (ShouldFlushBatch(buffer, batchIndex))
            {
                yield return await CreateBatchAsync(
                    buffer.ToArray(),
                    batchIndex,
                    isPartialBatch: false,
                    context,
                    token);

                batchIndex++;
                buffer.Clear();
            }

            await ReportProgressAsync(current, total);
        }

        if (buffer.Count > 0 && IncludeIncompleteBatch)
        {
            yield return await CreateBatchAsync(
                buffer.ToArray(),
                batchIndex,
                isPartialBatch: true,
                context,
                token);
        }
    }

    protected virtual bool ShouldFlushBatch(IReadOnlyList<TInput> buffer, int batchIndex)
    {
        return buffer.Count >= BatchSize;
    }

    protected abstract ValueTask<TBatch> CreateBatchAsync(
        IReadOnlyList<TInput> batchItems,
        int batchIndex,
        bool isPartialBatch,
        PipelineContext context,
        CancellationToken token);
}
