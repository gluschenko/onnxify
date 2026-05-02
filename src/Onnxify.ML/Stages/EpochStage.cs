using System.Runtime.CompilerServices;

namespace Onnxify.ML.Stages;

public sealed class EpochStage<TInput> : BatchPipelineStage<TInput, EpochItem<TInput>>
{
    public EpochStage(
        int epochs,
        PipelineStageOptions? options = null)
        : base(options ?? new PipelineStageOptions { Category = PipelineStageCategories.Orchestration })
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        Epochs = epochs;
    }

    public int Epochs { get; }

    protected override async IAsyncEnumerable<EpochItem<TInput>> ExecuteBatchAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        [EnumeratorCancellation] CancellationToken token)
    {
        var items = await MaterializeAsync(input, token);
        var total = items.Count * Epochs;
        var current = 0;

        await ReportProgressAsync(context, current, total);

        for (var epochIndex = 0; epochIndex < Epochs; epochIndex++)
        {
            for (var position = 0; position < items.Count; position++)
            {
                token.ThrowIfCancellationRequested();

                current++;
                
                yield return new EpochItem<TInput>
                {
                    Value = items[position],
                    EpochIndex = epochIndex,
                    EpochNumber = epochIndex + 1,
                    Position = position,
                    IsLastInEpoch = position == items.Count - 1,
                };

                await ReportProgressAsync(context, current, total);
            }
        }
    }

    protected override int? GetKnownOutputCount(IAsyncEnumerable<TInput> input)
    {
        return PipelineAsyncEnumerable.TryGetKnownCount(input, out var count)
            ? checked(count * Epochs)
            : null;
    }
}
