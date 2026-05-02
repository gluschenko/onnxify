using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

public abstract class ItemPipelineStage<TInput, TOutput> : PipelineStage<TInput, TOutput>
{
    protected ItemPipelineStage(PipelineStageOptions? options = null)
        : base(options)
    {
    }

    public ValueTask<TOutput> ExecuteSingleAsync(
        TInput input,
        PipelineContext? context = null,
        CancellationToken token = default)
    {
        return ProcessAsync(input, context ?? PipelineContext.Empty, token);
    }

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

    protected abstract ValueTask<TOutput> ProcessAsync(
        TInput input,
        PipelineContext context,
        CancellationToken token);
}
