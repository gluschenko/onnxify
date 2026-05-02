namespace Onnxify.ML.Stages;

/// <summary>
/// Executes side effects for each item while preserving the original item in the stream.
/// </summary>
public sealed class TapStage<TInput> : ItemPipelineStage<TInput, TInput>
{
    private readonly Func<TInput, PipelineContext, CancellationToken, ValueTask> _action;

    /// <summary>
    /// Initializes the stage from a synchronous side-effect action.
    /// </summary>
    public TapStage(
        Action<TInput> action,
        PipelineStageOptions? options = null)
        : this((input, _, _) =>
        {
            action(input);
            return ValueTask.CompletedTask;
        }, options)
    {
        ArgumentNullException.ThrowIfNull(action);
    }

    /// <summary>
    /// Initializes the stage from an asynchronous side-effect action.
    /// </summary>
    public TapStage(
        Func<TInput, PipelineContext, CancellationToken, ValueTask> action,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions { Category = PipelineStageCategories.Metrics })
    {
        _action = action ?? throw new ArgumentNullException(nameof(action));
    }

    protected override async ValueTask<TInput> ProcessAsync(
        TInput input,
        PipelineContext context,
        CancellationToken token)
    {
        await _action(input, context, token);
        return input;
    }
}
