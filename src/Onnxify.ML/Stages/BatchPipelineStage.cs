namespace Onnxify.ML.Stages;

/// <summary>
/// Base class for stages that need to reason about the entire upstream stream as a batch-oriented unit.
/// </summary>
public abstract class BatchPipelineStage<TInput, TOutput> : PipelineStage<TInput, TOutput>
{
    /// <summary>
    /// Initializes a batch-oriented stage.
    /// </summary>
    protected BatchPipelineStage(PipelineStageOptions? options = null)
        : base(options)
    {
    }

    /// <inheritdoc />
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

    /// <summary>
    /// Executes the stage against the upstream stream using batch-oriented logic.
    /// </summary>
    protected abstract IAsyncEnumerable<TOutput> ExecuteBatchAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token);

    /// <summary>
    /// Materializes an asynchronous source into memory for stages that require replayable input.
    /// </summary>
    protected static Task<IReadOnlyList<TInput>> MaterializeAsync(
        IAsyncEnumerable<TInput> input,
        CancellationToken cancellationToken)
    {
        return PipelineExecutionExtensions.ToListAsync(input, cancellationToken);
    }

    /// <summary>
    /// Returns a known output count when it can be determined without executing the stage.
    /// </summary>
    protected virtual int? GetKnownOutputCount(IAsyncEnumerable<TInput> input)
    {
        return null;
    }
}
