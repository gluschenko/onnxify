using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

/// <summary>
/// Base class for one-input to one-output stages that process items independently.
/// </summary>
public abstract class ItemPipelineStage<TInput, TOutput> : PipelineStage<TInput, TOutput>
{
    /// <summary>
    /// Initializes an item-wise stage.
    /// </summary>
    protected ItemPipelineStage(PipelineStageOptions? options = null)
        : base(options)
    {
    }

    /// <summary>
    /// Executes the stage for a single input item.
    /// </summary>
    public ValueTask<TOutput> ExecuteSingleAsync(
        TInput input,
        PipelineContext? context = null,
        CancellationToken token = default)
    {
        return ProcessAsync(input, context ?? PipelineContext.Empty, token);
    }

    /// <inheritdoc />
    public sealed override IAsyncEnumerable<TOutput> ExecuteAsync(
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

        await foreach (var item in input.WithCancellation(token))
        {
            token.ThrowIfCancellationRequested();

            yield return await ProcessAsync(item, context, token);

            current++;
            await ReportProgressAsync(context, current, total);
        }

        if (knownCount is null)
        {
            await ReportProgressAsync(context, current, current);
        }
    }

    /// <summary>
    /// Processes a single input item.
    /// </summary>
    protected abstract ValueTask<TOutput> ProcessAsync(
        TInput input,
        PipelineContext context,
        CancellationToken token);
}
