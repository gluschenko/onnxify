namespace Onnxify.ML;

/// <summary>
/// Represents a batch of items produced by a batching stage.
/// </summary>
public sealed class MiniBatch<TItem>
{
    /// <summary>
    /// Gets the items contained in the batch.
    /// </summary>
    public required IReadOnlyList<TItem> Items { get; init; }

    /// <summary>
    /// Gets the zero-based batch index.
    /// </summary>
    public required int BatchIndex { get; init; }

    /// <summary>
    /// Gets a value indicating whether the batch was flushed before reaching the nominal batch size.
    /// </summary>
    public required bool IsPartialBatch { get; init; }

    /// <summary>
    /// Gets the number of items in the batch.
    /// </summary>
    public int Count => Items.Count;
}
