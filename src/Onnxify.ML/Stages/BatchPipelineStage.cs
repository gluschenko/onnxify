namespace Onnxify.ML.Stages;

public abstract class BatchPipelineStage<TInput, TOutput> : PipelineStage<TInput, TOutput>
{
    protected BatchPipelineStage(PipelineStageOptions? options = null)
        : base(options)
    {
    }

    public sealed override IAsyncEnumerable<TOutput> ExecuteAsync(
        IEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(context);

        return ExecuteBatchAsync(Materialize(input), context, token);
    }

    protected abstract IAsyncEnumerable<TOutput> ExecuteBatchAsync(
        IReadOnlyList<TInput> input,
        PipelineContext context,
        CancellationToken token);

    protected static IReadOnlyList<TInput> Materialize(IEnumerable<TInput> input)
    {
        return input as IReadOnlyList<TInput> ?? input.ToArray();
    }
}
