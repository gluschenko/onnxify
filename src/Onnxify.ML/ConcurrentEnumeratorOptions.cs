namespace Onnxify.ML;

/// <summary>
/// Configures concurrency behavior for <see cref="ConcurrentEnumerator{TInput, TOutput}"/>.
/// </summary>
public sealed class ConcurrentEnumeratorOptions
{
    private int? _maxDegreeOfParallelism;

    /// <summary>
    /// Gets the maximum number of concurrent workers used to process items.
    /// </summary>
    public int MaxDegreeOfParallelism
    {
        get => _maxDegreeOfParallelism ?? Math.Min(Environment.ProcessorCount, 8);
        init
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
            _maxDegreeOfParallelism = value;
        }
    }

    /// <summary>
    /// Gets a value indicating whether emitted results should preserve input order.
    /// </summary>
    public bool PreserveOrder { get; init; } = true;

    /// <summary>
    /// Gets the bounded channel capacity used for pending work and produced results.
    /// </summary>
    public int? BoundedCapacity { get; init; }
}
