using System.Runtime.CompilerServices;

namespace Onnxify.ML;

/// <summary>
/// Composes two stages into a single streaming stage.
/// </summary>
public sealed class CompositeStage<TInput, TMiddle, TOutput> : PipelineStage<TInput, TOutput>
{
    private readonly PipelineStage<TInput, TMiddle> _first;
    private readonly PipelineStage<TMiddle, TOutput> _second;

    /// <summary>
    /// Initializes a composite stage from two connected child stages.
    /// </summary>
    public CompositeStage(
        PipelineStage<TInput, TMiddle> first,
        PipelineStage<TMiddle, TOutput> second,
        PipelineStageOptions? options = null
    ) : base(options)
    {
        _first = first ?? throw new ArgumentNullException(nameof(first));
        _second = second ?? throw new ArgumentNullException(nameof(second));
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<TOutput> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext context,
        [EnumeratorCancellation] CancellationToken token)
    {
        await foreach (var item in _second.ExecuteAsync(_first.ExecuteAsync(input, context, token), context, token))
        {
            yield return item;
        }
    }

    internal override IReadOnlyList<PipelineStage> GetChildren()
    {
        return [_first, _second];
    }
}
