namespace Onnxify.ML;

public sealed class EpochItem<TItem>
{
    public required TItem Value { get; init; }
    public required int EpochIndex { get; init; }
    public required int EpochNumber { get; init; }
    public required int Position { get; init; }
    public required bool IsLastInEpoch { get; init; }
}
