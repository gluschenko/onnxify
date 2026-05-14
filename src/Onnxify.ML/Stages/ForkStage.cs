using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

/// <summary>
/// Executes two child stages for each input item and returns both result collections together.
/// </summary>
public sealed class ForkStage<TInput, TLeft, TRight> : PipelineStage<TInput, ForkResult<TInput, TLeft, TRight>>
{
    private readonly PipelineStage<TInput, TLeft> _left;
    private readonly PipelineStage<TInput, TRight> _right;

    /// <summary>
    /// Initializes a fork stage from two child stages.
    /// </summary>
    public ForkStage(
        PipelineStage<TInput, TLeft> left,
        PipelineStage<TInput, TRight> right,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions
        {
            Name = "fork",
            Category = PipelineStageCategories.ORCHESTRATION
        })
    {
        _left = left ?? throw new ArgumentNullException(nameof(left));
        _right = right ?? throw new ArgumentNullException(nameof(right));
    }

    /// <inheritdoc />
    public override IAsyncEnumerable<ForkResult<TInput, TLeft, TRight>> ExecuteAsync(
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

    internal override IReadOnlyList<PipelineStage> GetChildren()
    {
        return [_left, _right];
    }

    private async IAsyncEnumerable<ForkResult<TInput, TLeft, TRight>> ExecuteCoreAsync(
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

            var left = await _left.ExecuteSingleAsync(item, context, token).ToListAsync(token);
            var right = await _right.ExecuteSingleAsync(item, context, token).ToListAsync(token);

            yield return new ForkResult<TInput, TLeft, TRight>
            {
                Input = item,
                Left = left,
                Right = right
            };

            current++;
            await ReportProgressAsync(context, current, total);
        }

        if (knownCount is null)
        {
            await ReportProgressAsync(context, current, current);
        }
    }
}
