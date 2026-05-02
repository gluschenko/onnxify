namespace Onnxify.ML;

/// <summary>
/// Convenience helpers for invoking single-item stage executions.
/// </summary>
public static class PipelineStageExecutionExtensions
{
    /// <summary>
    /// Executes a stage for a single input item and returns its asynchronous output stream.
    /// </summary>
    public static IAsyncEnumerable<TOutput> ExecuteSingleAsync<TInput, TOutput>(
        this PipelineStage<TInput, TOutput> stage,
        TInput input,
        PipelineContext context,
        CancellationToken token = default)
    {
        ArgumentNullException.ThrowIfNull(stage);
        ArgumentNullException.ThrowIfNull(context);

        return stage.ExecuteAsync(PipelineAsyncEnumerable.FromSingle(input), context, token);
    }
}
