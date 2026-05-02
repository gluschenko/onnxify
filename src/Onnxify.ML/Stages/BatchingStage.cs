using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

/// <summary>
/// Extensible batching primitive for classic mini-batches, bucketing and token-budget grouping.
/// </summary>
public abstract class BatchingStage<TInput, TBatch> : PipelineStage<TInput, TBatch>
{
    /// <summary>
    /// Initializes a batching stage.
    /// </summary>
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

    /// <summary>
    /// Gets the nominal batch size.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    /// Gets a value indicating whether the final incomplete batch should be emitted.
    /// </summary>
    public bool IncludeIncompleteBatch { get; }

    /// <inheritdoc />
    public sealed override IAsyncEnumerable<TBatch> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(context);

        var hasKnownCount = PipelineAsyncEnumerable.TryGetKnownCount(input, out var knownCount);
        var outputCount = hasKnownCount
            ? CalculateOutputCount(knownCount)
            : null;

        return PipelineAsyncEnumerable.WithKnownCount(
            ExecuteCoreAsync(input, context, hasKnownCount ? knownCount : null, token),
            outputCount);
    }

    private async IAsyncEnumerable<TBatch> ExecuteCoreAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        int? knownCount,
        [EnumeratorCancellation] CancellationToken token)
    {
        var total = knownCount ?? -1;
        var current = 0;
        var batchIndex = 0;
        var initialCapacity = BatchSize > 1024
            ? 1024
            : BatchSize;
        var buffer = new List<TInput>(initialCapacity);

        await ReportProgressAsync(context, current, total);

        await foreach (var item in input.WithCancellation(token))
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

            await ReportProgressAsync(context, current, total);
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

        if (knownCount is null)
        {
            await ReportProgressAsync(context, current, current);
        }
    }

    /// <summary>
    /// Determines whether the current buffer should be flushed into a batch.
    /// </summary>
    protected virtual bool ShouldFlushBatch(IReadOnlyList<TInput> buffer, int batchIndex)
    {
        return buffer.Count >= BatchSize;
    }

    /// <summary>
    /// Returns a known output count when it can be determined from the input count alone.
    /// </summary>
    protected virtual int? CalculateOutputCount(int inputCount)
    {
        return null;
    }

    /// <summary>
    /// Creates the emitted batch object for the supplied buffered items.
    /// </summary>
    protected abstract ValueTask<TBatch> CreateBatchAsync(
        IReadOnlyList<TInput> batchItems,
        int batchIndex,
        bool isPartialBatch,
        PipelineContext context,
        CancellationToken token);
}
