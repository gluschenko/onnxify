using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

/// <summary>
/// Routes each input item into one of two child stages based on a predicate.
/// </summary>
public sealed class BranchStage<TInput, TOutput> : PipelineStage<TInput, TOutput>
{
    private readonly Func<TInput, PipelineContext, CancellationToken, ValueTask<bool>> _predicate;
    private readonly PipelineStage<TInput, TOutput> _whenTrue;
    private readonly PipelineStage<TInput, TOutput> _whenFalse;

    /// <summary>
    /// Initializes the branch stage from a synchronous predicate.
    /// </summary>
    public BranchStage(
        Func<TInput, bool> predicate,
        PipelineStage<TInput, TOutput> whenTrue,
        PipelineStage<TInput, TOutput> whenFalse,
        PipelineStageOptions? options = null)
        : this(
            (input, _, _) => ValueTask.FromResult(predicate(input)),
            whenTrue,
            whenFalse,
            options)
    {
        ArgumentNullException.ThrowIfNull(predicate);
    }

    /// <summary>
    /// Initializes the branch stage from an asynchronous predicate.
    /// </summary>
    public BranchStage(
        Func<TInput, PipelineContext, CancellationToken, ValueTask<bool>> predicate,
        PipelineStage<TInput, TOutput> whenTrue,
        PipelineStage<TInput, TOutput> whenFalse,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions
        {
            Name = "branch",
            Category = PipelineStageCategories.ORCHESTRATION
        })
    {
        _predicate = predicate ?? throw new ArgumentNullException(nameof(predicate));
        _whenTrue = whenTrue ?? throw new ArgumentNullException(nameof(whenTrue));
        _whenFalse = whenFalse ?? throw new ArgumentNullException(nameof(whenFalse));
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

        return ExecuteCoreAsync(input, context, hasKnownCount ? knownCount : null, token);
    }

    internal override IReadOnlyList<PipelineStage> GetChildren()
    {
        return [_whenTrue, _whenFalse];
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

        await foreach (var item in input.WithCancellation(token))
        {
            token.ThrowIfCancellationRequested();

            var stage = await _predicate(item, context, token)
                ? _whenTrue
                : _whenFalse;

            await foreach (var output in stage.ExecuteSingleAsync(item, context, token))
            {
                yield return output;
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
