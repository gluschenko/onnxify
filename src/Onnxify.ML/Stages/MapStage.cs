namespace Onnxify.ML.Stages;

/// <summary>
/// Projects each input item into a new output item.
/// </summary>
public sealed class MapStage<TInput, TOutput> : ItemPipelineStage<TInput, TOutput>
{
    private readonly Func<TInput, PipelineContext, CancellationToken, ValueTask<TOutput>> _transform;

    /// <summary>
    /// Initializes the stage from a synchronous transform.
    /// </summary>
    public MapStage(
        Func<TInput, TOutput> transform,
        PipelineStageOptions? options = null)
        : this((input, _, _) => ValueTask.FromResult(transform(input)), options)
    {
        ArgumentNullException.ThrowIfNull(transform);
    }

    /// <summary>
    /// Initializes the stage from an asynchronous transform.
    /// </summary>
    public MapStage(
        Func<TInput, PipelineContext, CancellationToken, ValueTask<TOutput>> transform,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions { Category = PipelineStageCategories.DataPreparation })
    {
        _transform = transform ?? throw new ArgumentNullException(nameof(transform));
    }

    protected override ValueTask<TOutput> ProcessAsync(
        TInput input,
        PipelineContext context,
        CancellationToken token)
    {
        return _transform(input, context, token);
    }
}
