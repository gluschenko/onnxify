using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

public sealed class ParallelMapStage<TInput, TOutput> : BatchPipelineStage<TInput, TOutput>
{
    private readonly Func<TInput, PipelineContext, CancellationToken, ValueTask<TOutput>> _transform;
    private readonly ConcurrentEnumeratorOptions _options;

    public ParallelMapStage(
        Func<TInput, TOutput> transform,
        ConcurrentEnumeratorOptions? options = null,
        PipelineStageOptions? stageOptions = null)
        : this((input, _, _) => ValueTask.FromResult(transform(input)), options, stageOptions)
    {
        ArgumentNullException.ThrowIfNull(transform);
    }

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

    protected override async IAsyncEnumerable<TOutput> ExecuteBatchAsync(
        IReadOnlyList<TInput> input,
        PipelineContext context,
        [EnumeratorCancellation] CancellationToken token)
    {
        var current = 0;
        var total = input.Count;

        await ReportProgressAsync(current, total);

        var enumerator = new ConcurrentEnumerator<TInput, TOutput>(
            input,
            async (item, cancellationToken) =>
            {
                var result = await _transform(item, context, cancellationToken);
                await ReportProgressAsync(Interlocked.Increment(ref current), total);
                return result;
            },
            _options);

        await foreach (var item in enumerator.ExecuteAsync(token))
        {
            yield return item;
        }
    }
}
