namespace Onnxify.ML.Stages;

public sealed class DelegateStage<TInput, TOutput> : BatchPipelineStage<TInput, TOutput>
{
    private readonly Func<IReadOnlyList<TInput>, PipelineContext, CancellationToken, IAsyncEnumerable<TOutput>> _execute;

    public DelegateStage(
        Func<IReadOnlyList<TInput>, PipelineContext, CancellationToken, IAsyncEnumerable<TOutput>> execute,
        PipelineStageOptions? options = null)
        : base(options)
    {
        _execute = execute ?? throw new ArgumentNullException(nameof(execute));
    }

    protected override IAsyncEnumerable<TOutput> ExecuteBatchAsync(
        IReadOnlyList<TInput> input,
        PipelineContext context,
        CancellationToken token)
    {
        return _execute(input, context, token);
    }
}
