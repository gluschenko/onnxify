namespace Onnxify.ML;

public sealed class MiniBatch<TItem>
{
    public required IReadOnlyList<TItem> Items { get; init; }
    public required int BatchIndex { get; init; }
    public required bool IsPartialBatch { get; init; }
    public int Count => Items.Count;
}
