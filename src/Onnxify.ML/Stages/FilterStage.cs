using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

public sealed class FilterStage<TInput> : PipelineStage<TInput, TInput>
{
    private readonly Func<TInput, PipelineContext, CancellationToken, ValueTask<bool>> _predicate;

    public FilterStage(
        Func<TInput, bool> predicate,
        PipelineStageOptions? options = null)
        : this((input, _, _) => ValueTask.FromResult(predicate(input)), options)
    {
        ArgumentNullException.ThrowIfNull(predicate);
    }

    public FilterStage(
        Func<TInput, PipelineContext, CancellationToken, ValueTask<bool>> predicate,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions { Category = PipelineStageCategories.DataPreparation })
    {
        _predicate = predicate ?? throw new ArgumentNullException(nameof(predicate));
    }

    public override async IAsyncEnumerable<TInput> ExecuteAsync(
        IEnumerable<TInput> input,
        PipelineContext context,
        [EnumeratorCancellation] CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(context);

        var current = 0;
        var total = input.TryGetNonEnumeratedCount(out var knownTotal)
            ? knownTotal
            : -1;

        await ReportProgressAsync(current, total);

        foreach (var item in input)
        {
            token.ThrowIfCancellationRequested();

            if (await _predicate(item, context, token))
            {
                yield return item;
            }

            current++;
            await ReportProgressAsync(current, total);
        }
    }
}
