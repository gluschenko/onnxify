namespace Onnxify.ML.TorchSharp;

public sealed class TorchInferenceResult<TBatch, TResult> : IDisposable
{
    public TorchInferenceResult(TBatch batch, TResult result, int inferenceIndex)
    {
        Batch = batch;
        Result = result;
        InferenceIndex = inferenceIndex;
    }

    public TBatch Batch { get; }

    public TResult Result { get; }

    public int InferenceIndex { get; }

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
