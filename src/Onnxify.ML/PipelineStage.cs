namespace Onnxify.ML;

/// <summary>
/// Receives raw stage progress updates during pipeline execution.
/// </summary>
public delegate ValueTask ProgressChangeEvent(PipelineStage stage, int current, int total);

/// <summary>
/// Base non-generic metadata container for pipeline stages.
/// </summary>
public abstract class PipelineStage
{
    /// <summary>
    /// Initializes a new stage with the provided options.
    /// </summary>
    protected PipelineStage(PipelineStageOptions? options = null)
    {
        options ??= new PipelineStageOptions();

        if (options.ProgressWeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "ProgressWeight must be greater than zero.");
        }

        Name = string.IsNullOrWhiteSpace(options.Name)
            ? GetType().Name
            : options.Name;
        Category = string.IsNullOrWhiteSpace(options.Category)
            ? PipelineStageCategories.Custom
            : options.Category;
        ProgressWeight = options.ProgressWeight;
    }

    /// <summary>
    /// Gets the stage name shown in progress and diagnostics output.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the logical stage category.
    /// </summary>
    public string Category { get; }

    /// <summary>
    /// Gets the relative contribution of this stage to aggregate pipeline progress.
    /// </summary>
    public double ProgressWeight { get; }

    internal virtual IReadOnlyList<PipelineStage> GetChildren()
    {
        return [];
    }
}

/// <summary>
/// Base class for typed pipeline stages that transform a stream of input items into a stream of output items.
/// </summary>
public abstract class PipelineStage<TInput, TOutput> : PipelineStage
{
    /// <summary>
    /// Initializes a new typed stage with the provided options.
    /// </summary>
    protected PipelineStage(PipelineStageOptions? options = null)
        : base(options)
    {
    }

    /// <summary>
    /// Executes the stage for the supplied input stream.
    /// </summary>
    public abstract IAsyncEnumerable<TOutput> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token);

    /// <summary>
    /// Chains the current stage into the supplied next stage.
    /// </summary>
    public CompositeStage<TInput, TOutput, TOutputNext> Then<TOutputNext>(PipelineStage<TOutput, TOutputNext> stage)
    {
        ArgumentNullException.ThrowIfNull(stage);

        return new CompositeStage<TInput, TOutput, TOutputNext>(this, stage);
    }

    /// <summary>
    /// Reports progress for the current stage through the active execution context.
    /// </summary>
    protected ValueTask ReportProgressAsync(PipelineContext context, int current, int total)
    {
        ArgumentNullException.ThrowIfNull(context);
        return context.ReportProgressAsync(this, current, total);
    }

    /// <summary>
    /// Builds a reusable <see cref="Pipeline{TInput, TOutput}"/> rooted at this stage.
    /// </summary>
    public virtual Pipeline<TInput, TOutput> Build()
    {
        return new Pipeline<TInput, TOutput>(this);
    }
}
