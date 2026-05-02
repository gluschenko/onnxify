namespace Onnxify.ML.TorchSharp;

/// <summary>
/// Represents the output of a single inference step.
/// </summary>
public sealed class TorchInferenceResult<TBatch, TResult> : IDisposable
{
    /// <summary>
    /// Initializes a new inference result.
    /// </summary>
    public TorchInferenceResult(TBatch batch, TResult result, int inferenceIndex)
    {
        Batch = batch;
        Result = result;
        InferenceIndex = inferenceIndex;
    }

    /// <summary>
    /// Gets the source batch that produced the result.
    /// </summary>
    public TBatch Batch { get; }

    /// <summary>
    /// Gets the projected inference payload.
    /// </summary>
    public TResult Result { get; }

    /// <summary>
    /// Gets the zero-based inference index for the owning stage instance within the current execution.
    /// </summary>
    public int InferenceIndex { get; }

    /// <summary>
    /// Disposes the batch and result when they implement <see cref="IDisposable"/>.
    /// </summary>
    public void Dispose()
    {
        DisposeIfNeeded(Batch);
        DisposeIfNeeded(Result);
    }

    private static void DisposeIfNeeded<T>(T value)
    {
        if (value is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
}
