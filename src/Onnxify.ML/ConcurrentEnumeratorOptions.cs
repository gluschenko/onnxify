namespace Onnxify.ML;

public sealed class ConcurrentEnumeratorOptions
{
    private int? _maxDegreeOfParallelism;

    public int MaxDegreeOfParallelism
    {
        get => _maxDegreeOfParallelism ?? Math.Min(Environment.ProcessorCount, 8);
        init
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
            _maxDegreeOfParallelism = value;
        }
    }

    public bool PreserveOrder { get; init; } = true;

    public int? BoundedCapacity { get; init; }
}
