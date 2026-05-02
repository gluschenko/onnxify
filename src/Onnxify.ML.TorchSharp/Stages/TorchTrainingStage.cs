using Onnxify.ML.Stages;
using TorchSharp;
using Tensor = global::TorchSharp.torch.Tensor;
using TorchOptimizer = global::TorchSharp.torch.optim.Optimizer;

namespace Onnxify.ML.TorchSharp.Stages;

/// <summary>
/// Executes a Torch training step including forward pass, loss computation, backward pass, and optimizer update.
/// </summary>
public sealed class TorchTrainingStage<TBatch, TModelOutput, TResult>
    : ItemPipelineStage<TBatch, TorchTrainingStepResult<TBatch, TResult>>
{
    private readonly TorchOptimizer _optimizer;
    private readonly Func<TBatch, PipelineContext, CancellationToken, ValueTask<TModelOutput>> _forward;
    private readonly Func<TBatch, TModelOutput, PipelineContext, CancellationToken, ValueTask<Tensor>> _lossSelector;
    private readonly Func<TBatch, TModelOutput, Tensor, PipelineContext, CancellationToken, ValueTask<TResult>> _resultSelector;
    private readonly bool _disposeModelOutput;
    private readonly bool _zeroGradBeforeStep;

    /// <summary>
    /// Initializes a Torch training stage.
    /// </summary>
    public TorchTrainingStage(
        TorchOptimizer optimizer,
        Func<TBatch, PipelineContext, CancellationToken, ValueTask<TModelOutput>> forward,
        Func<TBatch, TModelOutput, PipelineContext, CancellationToken, ValueTask<Tensor>> lossSelector,
        Func<TBatch, TModelOutput, Tensor, PipelineContext, CancellationToken, ValueTask<TResult>> resultSelector,
        bool zeroGradBeforeStep = true,
        bool disposeModelOutput = true,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions
        {
            Category = PipelineStageCategories.Optimization,
            Name = "torch-train-step"
        })
    {
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        _forward = forward ?? throw new ArgumentNullException(nameof(forward));
        _lossSelector = lossSelector ?? throw new ArgumentNullException(nameof(lossSelector));
        _resultSelector = resultSelector ?? throw new ArgumentNullException(nameof(resultSelector));
        _zeroGradBeforeStep = zeroGradBeforeStep;
        _disposeModelOutput = disposeModelOutput;
    }

    protected override async ValueTask<TorchTrainingStepResult<TBatch, TResult>> ProcessAsync(
        TBatch input,
        PipelineContext context,
        CancellationToken token)
    {
        if (_zeroGradBeforeStep)
        {
            _optimizer.zero_grad();
        }

        var output = await _forward(input, context, token);

        try
        {
            using var loss = await _lossSelector(input, output, context, token);
            var summary = await _resultSelector(input, output, loss, context, token);
            var lossValue = loss.ToSingle();

            loss.backward();
            _optimizer.step();

            return new TorchTrainingStepResult<TBatch, TResult>(
                input,
                summary,
                context.NextSequenceNumber(this),
                lossValue);
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
