using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

/// <summary>
/// Projects items concurrently while optionally preserving the original input order.
/// </summary>
public sealed class ParallelMapStage<TInput, TOutput> : PipelineStage<TInput, TOutput>
{
    private readonly Func<TInput, PipelineContext, CancellationToken, ValueTask<TOutput>> _transform;
    private readonly ConcurrentEnumeratorOptions _options;

    /// <summary>
    /// Initializes the stage from a synchronous transform.
    /// </summary>
    public ParallelMapStage(
        Func<TInput, TOutput> transform,
        ConcurrentEnumeratorOptions? options = null,
        PipelineStageOptions? stageOptions = null)
        : this((input, _, _) => ValueTask.FromResult(transform(input)), options, stageOptions)
    {
        ArgumentNullException.ThrowIfNull(transform);
    }

    /// <summary>
    /// Initializes the stage from an asynchronous transform.
    /// </summary>
    public ParallelMapStage(
        Func<TInput, PipelineContext, CancellationToken, ValueTask<TOutput>> transform,
        ConcurrentEnumeratorOptions? options = null,
        PipelineStageOptions? stageOptions = null)
        : base(stageOptions ?? new PipelineStageOptions
        {
            Category = PipelineStageCategories.DataPreparation,
            Name = "parallel-map"
        })
    {
        _transform = transform ?? throw new ArgumentNullException(nameof(transform));
        _options = options ?? new ConcurrentEnumeratorOptions();
    }

    /// <inheritdoc />
    public override IAsyncEnumerable<TOutput> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(context);

        var hasKnownCount = PipelineAsyncEnumerable.TryGetKnownCount(input, out var knownCount);

        return PipelineAsyncEnumerable.WithKnownCount(
            ExecuteCoreAsync(input, context, hasKnownCount ? knownCount : null, token),
            hasKnownCount ? knownCount : null);
    }

    private async IAsyncEnumerable<TOutput> ExecuteCoreAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        int? knownCount,
        [EnumeratorCancellation] CancellationToken token)
    {
        var current = 0;
        var total = knownCount ?? -1;

        await ReportProgressAsync(context, current, total);

        var enumerator = new ConcurrentEnumerator<TInput, TOutput>(
            input,
            (Func<TInput, CancellationToken, ValueTask<TOutput>>)(async (item, cancellationToken) =>
            {
                var result = await _transform(item, context, cancellationToken);
                await ReportProgressAsync(context, Interlocked.Increment(ref current), total);
                return result;
            }),
            _options);

        await foreach (var item in enumerator.ExecuteAsync(token))
        {
            yield return item;
        }

        if (knownCount is null)
        {
            await ReportProgressAsync(context, current, current);
        }
    }
}
