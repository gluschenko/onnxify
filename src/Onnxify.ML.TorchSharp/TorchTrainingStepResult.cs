namespace Onnxify.ML.TorchSharp;

/// <summary>
/// Represents the output of a single training step.
/// </summary>
public sealed class TorchTrainingStepResult<TBatch, TResult> : IDisposable
{
    /// <summary>
    /// Initializes a new training step result.
    /// </summary>
    public TorchTrainingStepResult(
        TBatch batch,
        TResult result,
        int stepIndex,
        float loss)
    {
        Batch = batch;
        Result = result;
        StepIndex = stepIndex;
        Loss = loss;
    }

    /// <summary>
    /// Gets the source batch that produced the result.
    /// </summary>
    public TBatch Batch { get; }

    /// <summary>
    /// Gets the projected result payload for the training step.
    /// </summary>
    public TResult Result { get; }

    /// <summary>
    /// Gets the zero-based training step index for the owning stage instance within the current execution.
    /// </summary>
    public int StepIndex { get; }

    /// <summary>
    /// Gets the scalar loss value after the optimization step.
    /// </summary>
    public float Loss { get; }

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
