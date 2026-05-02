namespace Onnxify.ML.Stages;

public sealed class DelegateStage<TInput, TOutput> : PipelineStage<TInput, TOutput>
{
    private readonly Func<IAsyncEnumerable<TInput>, PipelineContext, CancellationToken, IAsyncEnumerable<TOutput>> _execute;

    public DelegateStage(
        Func<IAsyncEnumerable<TInput>, PipelineContext, CancellationToken, IAsyncEnumerable<TOutput>> execute,
        PipelineStageOptions? options = null)
        : base(options)
    {
        _execute = execute ?? throw new ArgumentNullException(nameof(execute));
    }

    public override IAsyncEnumerable<TOutput> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token)
    {
        return _execute(input, context, token);
    }
}
