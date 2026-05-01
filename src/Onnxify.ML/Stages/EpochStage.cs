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
        IReadOnlyList<TInput> input,
        PipelineContext context,
        [EnumeratorCancellation] CancellationToken token)
    {
        var total = input.Count * Epochs;
        var current = 0;

        await ReportProgressAsync(current, total);

        for (var epochIndex = 0; epochIndex < Epochs; epochIndex++)
        {
            for (var position = 0; position < input.Count; position++)
            {
                token.ThrowIfCancellationRequested();

                current++;
                
                yield return new EpochItem<TInput>
                {
                    Value = input[position],
                    EpochIndex = epochIndex,
                    EpochNumber = epochIndex + 1,
                    Position = position,
                    IsLastInEpoch = position == input.Count - 1,
                };

                await ReportProgressAsync(current, total);
            }
        }
    }
}
