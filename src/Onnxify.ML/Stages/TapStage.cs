namespace Onnxify.ML.Stages;

public sealed class TapStage<TInput> : ItemPipelineStage<TInput, TInput>
{
    private readonly Func<TInput, PipelineContext, CancellationToken, ValueTask> _action;

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
