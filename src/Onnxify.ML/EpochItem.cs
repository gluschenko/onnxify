namespace Onnxify.ML;

/// <summary>
/// Wraps an item emitted by <see cref="Stages.EpochStage{TInput}"/> together with epoch metadata.
/// </summary>
public sealed class EpochItem<TItem>
{
    /// <summary>
    /// Gets the original item being replayed for the current epoch.
    /// </summary>
    public required TItem Value { get; init; }

    /// <summary>
    /// Gets the zero-based epoch index.
    /// </summary>
    public required int EpochIndex { get; init; }

    /// <summary>
    /// Gets the one-based epoch number.
    /// </summary>
    public required int EpochNumber { get; init; }

    /// <summary>
    /// Gets the zero-based position of the item inside the epoch.
    /// </summary>
    public required int Position { get; init; }

    /// <summary>
    /// Gets a value indicating whether this item is the last item in the current epoch.
    /// </summary>
    public required bool IsLastInEpoch { get; init; }
}
