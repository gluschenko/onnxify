namespace Onnxify.ML.Stages;

/// <summary>
/// Adapts an arbitrary asynchronous delegate into a pipeline stage.
/// </summary>
public sealed class DelegateStage<TInput, TOutput> : PipelineStage<TInput, TOutput>
{
    private readonly Func<IAsyncEnumerable<TInput>, PipelineContext, CancellationToken, IAsyncEnumerable<TOutput>> _execute;

    /// <summary>
    /// Initializes the stage from a delegate that consumes the upstream async stream directly.
    /// </summary>
    public DelegateStage(
        Func<IAsyncEnumerable<TInput>, PipelineContext, CancellationToken, IAsyncEnumerable<TOutput>> execute,
        PipelineStageOptions? options = null)
        : base(options)
    {
        _execute = execute ?? throw new ArgumentNullException(nameof(execute));
    }

    /// <inheritdoc />
    public override IAsyncEnumerable<TOutput> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token)
    {
        return _execute(input, context, token);
    }
}
