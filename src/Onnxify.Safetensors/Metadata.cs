namespace Onnxify.Safetensors;

/// <summary>
/// Represents the validated safetensors header model used to resolve tensor names, offsets, and optional archive metadata.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>Metadata</c>.
/// </remarks>
public sealed class Metadata
{
    private readonly Dictionary<string, int> _indexMap;
    private readonly List<string> _namesByIndex;
    private readonly List<TensorInfo> _tensors;

    /// <summary>
    /// Gets the optional top-level <c>__metadata__</c> entries stored in the safetensors header.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Metadata.metadata</c>.
    /// </remarks>
    public IReadOnlyDictionary<string, string>? MetadataEntries { get; }

    /// <summary>
    /// Initializes a validated metadata object from the header metadata dictionary and the ordered tensor table.
    /// </summary>
    /// <param name="metadataEntries">Optional top-level metadata key-value pairs.</param>
    /// <param name="tensors">The tensor entries to index by name and validate for contiguous offsets.</param>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Metadata::new</c>.
    /// </remarks>
    public Metadata(
        IReadOnlyDictionary<string, string>? metadataEntries,
        IEnumerable<KeyValuePair<string, TensorInfo>> tensors
    )
    {
        ArgumentNullException.ThrowIfNull(tensors);

        _indexMap = new Dictionary<string, int>(StringComparer.Ordinal);
        _namesByIndex = [];
        _tensors = [];

        foreach (var pair in tensors)
        {
            _indexMap[pair.Key] = _tensors.Count;
            _namesByIndex.Add(pair.Key);
            _tensors.Add(pair.Value);
        }

        MetadataEntries = metadataEntries is null
            ? null
            : new Dictionary<string, string>(metadataEntries, StringComparer.Ordinal);

        Validate();
    }

    /// <summary>
    /// Looks up tensor metadata by name and returns <see langword="null"/> when the tensor does not exist.
    /// </summary>
    /// <param name="name">The tensor name as stored in the safetensors header.</param>
    /// <returns>The matching tensor metadata, or <see langword="null"/> if no tensor uses that name.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Metadata.index_map</c> lookup used by <c>SafeTensors::tensor</c>.
    /// </remarks>
    public TensorInfo? Info(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return _indexMap.TryGetValue(name, out var index) ? _tensors[index] : null;
    }

    /// <summary>
    /// Materializes the current tensor table as a name-to-info dictionary.
    /// </summary>
    /// <returns>A snapshot of the indexed tensor metadata keyed by tensor name.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Metadata.tensors</c> plus <c>Metadata.index_map</c>.
    /// </remarks>
    public IReadOnlyDictionary<string, TensorInfo> Tensors()
    {
        return _indexMap.ToDictionary(pair => pair.Key, pair => _tensors[pair.Value], StringComparer.Ordinal);
    }

    /// <summary>
    /// Returns tensor names in the stored order used by the managed metadata index.
    /// </summary>
    /// <returns>An ordered list of tensor names.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>SafeTensors::names</c>.
    /// </remarks>
    public IReadOnlyList<string> OffsetKeys()
    {
        return _namesByIndex.ToArray();
    }

    /// <summary>
    /// Computes the total byte length covered by the tensor payload described by this metadata object.
    /// </summary>
    /// <returns>The exclusive end offset of the final tensor, or zero when there are no tensors.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Metadata::validate</c> return value.
    /// </remarks>
    public ulong DataLength()
    {
        return _tensors.Count == 0 ? 0UL : _tensors[^1].DataOffsets.End;
    }

    /// <summary>
    /// Validates that tensor offsets are contiguous and that every entry's shape, data type, and byte range agree.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Metadata::validate</c>.
    /// </remarks>
    internal void Validate()
    {
        ulong start = 0;

        for (var i = 0; i < _tensors.Count; i++)
        {
            var info = _tensors[i];
            var (s, e) = info.DataOffsets;

            if (s != start || e < s)
            {
                throw SafeTensorException.InvalidOffset(_namesByIndex[i]);
            }

            start = e;

            var size = SafeTensorMath.ComputeSizeInBytes(info.DataType, info.Shape, allowMisaligned: false);
            if (e - s != size)
            {
                throw SafeTensorException.TensorInvalidInfo();
            }
        }
    }
}
