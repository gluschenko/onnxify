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

    public sealed override async IAsyncEnumerable<TOutput> ExecuteAsync(
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

            yield return await ProcessAsync(item, context, token);

            current++;
            await ReportProgressAsync(current, total);
        }
    }

    protected abstract ValueTask<TOutput> ProcessAsync(
        TInput input,
        PipelineContext context,
        CancellationToken token);
}
