namespace Onnxify.ML;

public static class PipelineStageExecutionExtensions
{
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
