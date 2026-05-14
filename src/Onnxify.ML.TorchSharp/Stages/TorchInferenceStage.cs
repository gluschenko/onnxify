using Onnxify.ML.Stages;

namespace Onnxify.ML.TorchSharp.Stages;

/// <summary>
/// Executes a Torch forward pass and projects the resulting inference payload.
/// </summary>
public sealed class TorchInferenceStage<TBatch, TModelOutput, TResult>
    : ItemPipelineStage<TBatch, TorchInferenceResult<TBatch, TResult>>
{
    private readonly Func<TBatch, PipelineContext, CancellationToken, ValueTask<TModelOutput>> _forward;
    private readonly Func<TBatch, TModelOutput, PipelineContext, CancellationToken, ValueTask<TResult>> _resultSelector;
    private readonly bool _disposeModelOutput;

    /// <summary>
    /// Initializes a Torch inference stage.
    /// </summary>
    public TorchInferenceStage(
        Func<TBatch, PipelineContext, CancellationToken, ValueTask<TModelOutput>> forward,
        Func<TBatch, TModelOutput, PipelineContext, CancellationToken, ValueTask<TResult>> resultSelector,
        bool disposeModelOutput = true,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions
        {
            Category = PipelineStageCategories.INFERENCE,
            Name = "torch-inference"
        })
    {
        _forward = forward ?? throw new ArgumentNullException(nameof(forward));
        _resultSelector = resultSelector ?? throw new ArgumentNullException(nameof(resultSelector));
        _disposeModelOutput = disposeModelOutput;
    }

    protected override async ValueTask<TorchInferenceResult<TBatch, TResult>> ProcessAsync(
        TBatch input,
        PipelineContext context,
        CancellationToken token)
    {
        var output = await _forward(input, context, token);

        try
        {
            var result = await _resultSelector(input, output, context, token);
            return new TorchInferenceResult<TBatch, TResult>(
                input,
                result,
                context.NextSequenceNumber(this));
        }
        finally
        {
            if (_disposeModelOutput && output is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }
}
