using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

/// <summary>
/// Filters items from the upstream stream using a predicate.
/// </summary>
public sealed class FilterStage<TInput> : PipelineStage<TInput, TInput>
{
    private readonly Func<TInput, PipelineContext, CancellationToken, ValueTask<bool>> _predicate;

    /// <summary>
    /// Initializes the stage from a synchronous predicate.
    /// </summary>
    public FilterStage(
        Func<TInput, bool> predicate,
        PipelineStageOptions? options = null)
        : this((input, _, _) => ValueTask.FromResult(predicate(input)), options)
    {
        ArgumentNullException.ThrowIfNull(predicate);
    }

    /// <summary>
    /// Initializes the stage from an asynchronous predicate.
    /// </summary>
    public FilterStage(
        Func<TInput, PipelineContext, CancellationToken, ValueTask<bool>> predicate,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions { Category = PipelineStageCategories.DATA_PREPARATION })
    {
        _predicate = predicate ?? throw new ArgumentNullException(nameof(predicate));
    }

    /// <inheritdoc />
    public override IAsyncEnumerable<TInput> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(context);

        var hasKnownCount = PipelineAsyncEnumerable.TryGetKnownCount(input, out var knownCount);

        return ExecuteCoreAsync(input, context, hasKnownCount ? knownCount : null, token);
    }

    private async IAsyncEnumerable<TInput> ExecuteCoreAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        int? knownCount,
        [EnumeratorCancellation] CancellationToken token)
    {
        var current = 0;
        var total = knownCount ?? -1;

        await ReportProgressAsync(context, current, total);

        await foreach (var item in input.WithCancellation(token))
        {
            token.ThrowIfCancellationRequested();

            if (await _predicate(item, context, token))
            {
                yield return item;
            }

            current++;
            await ReportProgressAsync(context, current, total);
        }

        if (knownCount is null)
        {
            await ReportProgressAsync(context, current, current);
        }
    }
}
