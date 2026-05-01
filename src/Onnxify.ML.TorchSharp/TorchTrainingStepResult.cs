namespace Onnxify.ML.TorchSharp;

public sealed class TorchTrainingStepResult<TBatch, TResult> : IDisposable
{
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

    public TBatch Batch { get; }

    public TResult Result { get; }

    public int StepIndex { get; }

    public float Loss { get; }

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
