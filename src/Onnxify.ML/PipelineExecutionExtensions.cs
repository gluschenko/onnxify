namespace Onnxify.ML;

/// <summary>
/// Helper methods for executing pipelines and materializing asynchronous results.
/// </summary>
public static class PipelineExecutionExtensions
{
    /// <summary>
    /// Wraps a synchronous enumerable as an asynchronous stream.
    /// </summary>
    public static IAsyncEnumerable<TOutput> ToAsyncEnumerable<TOutput>(
        this IEnumerable<TOutput> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return PipelineAsyncEnumerable.FromEnumerable(source);
    }

    /// <summary>
    /// Materializes an asynchronous stream into a list.
    /// </summary>
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

    /// <summary>
    /// Executes the pipeline for a synchronous input source and materializes the results into a list.
    /// </summary>
    public static Task<IReadOnlyList<TOutput>> RunToListAsync<TInput, TOutput>(
        this Pipeline<TInput, TOutput> pipeline,
        IEnumerable<TInput> input,
        PipelineContext? context = null,
        ProgressChangeEvent? progressChangeEvent = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        return pipeline.ExecuteAsync(input, context, progressChangeEvent, cancellationToken).ToListAsync(cancellationToken);
    }

    /// <summary>
    /// Executes the pipeline for an asynchronous input source and materializes the results into a list.
    /// </summary>
    public static Task<IReadOnlyList<TOutput>> RunToListAsync<TInput, TOutput>(
        this Pipeline<TInput, TOutput> pipeline,
        IAsyncEnumerable<TInput> input,
        PipelineContext? context = null,
        ProgressChangeEvent? progressChangeEvent = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        return pipeline.ExecuteAsync(input, context, progressChangeEvent, cancellationToken).ToListAsync(cancellationToken);
    }
}
