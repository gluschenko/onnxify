namespace Onnxify.ML.Stages;

public abstract class BatchPipelineStage<TInput, TOutput> : PipelineStage<TInput, TOutput>
{
    protected BatchPipelineStage(PipelineStageOptions? options = null)
        : base(options)
    {
    }

    public sealed override IAsyncEnumerable<TOutput> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(context);

        return PipelineAsyncEnumerable.WithKnownCount(
            ExecuteBatchAsync(input, context, token),
            GetKnownOutputCount(input));
    }

    protected abstract IAsyncEnumerable<TOutput> ExecuteBatchAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token);

    protected static Task<IReadOnlyList<TInput>> MaterializeAsync(
        IAsyncEnumerable<TInput> input,
        CancellationToken cancellationToken)
    {
        return PipelineExecutionExtensions.ToListAsync(input, cancellationToken);
    }

    protected virtual int? GetKnownOutputCount(IAsyncEnumerable<TInput> input)
    {
        return null;
    }
}
