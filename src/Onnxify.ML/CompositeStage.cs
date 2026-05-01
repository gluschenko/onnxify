using System.Runtime.CompilerServices;

namespace Onnxify.ML;

public sealed class CompositeStage<TInput, TMiddle, TOutput> : PipelineStage<TInput, TOutput>
{
    private readonly PipelineStage<TInput, TMiddle> _first;
    private readonly PipelineStage<TMiddle, TOutput> _second;
    private ProgressChangeEvent? _progressChangeEvent;

    public CompositeStage(
        PipelineStage<TInput, TMiddle> first,
        PipelineStage<TMiddle, TOutput> second,
        PipelineStageOptions? options = null
    ) : base(options)
    {
        _first = first ?? throw new ArgumentNullException(nameof(first));
        _second = second ?? throw new ArgumentNullException(nameof(second));
    }

    public override async IAsyncEnumerable<TOutput> ExecuteAsync(
        IEnumerable<TInput> input,
        PipelineContext context,
        [EnumeratorCancellation] CancellationToken token)
    {
        var middle = new List<TMiddle>();

        var inputCurrent = 0;
        var inputTotal = TryGetTotal(input);

        if (_progressChangeEvent is not null && IsLeaf(_first))
        {
            await _progressChangeEvent.Invoke(_first, inputCurrent, inputTotal);
        }

        await foreach (var item in _first.ExecuteAsync(input, context, token))
        {
            inputCurrent++;

            if (_progressChangeEvent is not null && IsLeaf(_first))
            {
                await _progressChangeEvent.Invoke(_first, inputCurrent, inputTotal);
            }

            middle.Add(item);
        }

        var outputCurrent = 0;
        var outputTotal = middle.Count;

        if (_progressChangeEvent is not null && IsLeaf(_second))
        {
            await _progressChangeEvent.Invoke(_second, outputCurrent, outputTotal);
        }

        await foreach (var item in _second.ExecuteAsync(middle, context, token))
        {
            outputCurrent++;

            if (_progressChangeEvent is not null && IsLeaf(_second))
            {
                await _progressChangeEvent.Invoke(_second, outputCurrent, outputTotal);
            }

            yield return item;
        }
    }

    public override CompositeStage<TInput, TMiddle, TOutput> WithProgress(ProgressChangeEvent progressChangeEvent)
    {
        ArgumentNullException.ThrowIfNull(progressChangeEvent);

        _progressChangeEvent = progressChangeEvent;
        _first.WithProgress(progressChangeEvent);
        _second.WithProgress(progressChangeEvent);
        return this;
    }

    internal override IReadOnlyList<PipelineStage> GetChildren()
    {
        return [_first, _second];
    }

    private static bool IsLeaf(PipelineStage stage)
    {
        return stage.GetChildren().Count == 0;
    }

    private static int TryGetTotal<T>(IEnumerable<T> source)
    {
        return source.TryGetNonEnumeratedCount(out var total)
            ? total
            : -1;
    }
}
