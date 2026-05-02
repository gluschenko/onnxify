namespace Onnxify.ML;

public delegate ValueTask ProgressChangeEvent(PipelineStage stage, int current, int total);

public abstract class PipelineStage
{
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

    public string Name { get; }

    public string Category { get; }

    public double ProgressWeight { get; }

    internal virtual IReadOnlyList<PipelineStage> GetChildren()
    {
        return [];
    }
}

public abstract class PipelineStage<TInput, TOutput> : PipelineStage
{
    protected PipelineStage(PipelineStageOptions? options = null)
        : base(options)
    {
    }

    public abstract IAsyncEnumerable<TOutput> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        CancellationToken token);

    public CompositeStage<TInput, TOutput, TOutputNext> Then<TOutputNext>(PipelineStage<TOutput, TOutputNext> stage)
    {
        ArgumentNullException.ThrowIfNull(stage);

        return new CompositeStage<TInput, TOutput, TOutputNext>(this, stage);
    }

    protected ValueTask ReportProgressAsync(PipelineContext context, int current, int total)
    {
        ArgumentNullException.ThrowIfNull(context);
        return context.ReportProgressAsync(this, current, total);
    }

    public virtual Pipeline<TInput, TOutput> Build()
    {
        return new Pipeline<TInput, TOutput>(this);
    }
}
