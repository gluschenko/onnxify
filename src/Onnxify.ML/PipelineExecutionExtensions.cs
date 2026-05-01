namespace Onnxify.ML;

public static class PipelineExecutionExtensions
{
    public static async Task<IReadOnlyList<TOutput>> ToListAsync<TOutput>(
        this IAsyncEnumerable<TOutput> source,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);

        var result = new List<TOutput>();
        await foreach (var item in source.WithCancellation(cancellationToken))
        {
            result.Add(item);
        }

        return result;
    }

    public static Task<IReadOnlyList<TOutput>> RunToListAsync<TInput, TOutput>(
        this Pipeline<TInput, TOutput> pipeline,
        IEnumerable<TInput> input,
        PipelineContext? context = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        return pipeline.ExecuteAsync(input, context, cancellationToken).ToListAsync(cancellationToken);
    }
}
