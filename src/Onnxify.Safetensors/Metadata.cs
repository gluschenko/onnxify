namespace Onnxify.Safetensors;

public sealed class Metadata
{
    private readonly Dictionary<string, int> _indexMap;
    private readonly List<string> _namesByIndex;
    private readonly List<TensorInfo> _tensors;

    public IReadOnlyDictionary<string, string>? MetadataEntries { get; }

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

    public TensorInfo? Info(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return _indexMap.TryGetValue(name, out var index) ? _tensors[index] : null;
    }

    public IReadOnlyDictionary<string, TensorInfo> Tensors()
    {
        return _indexMap.ToDictionary(pair => pair.Key, pair => _tensors[pair.Value], StringComparer.Ordinal);
    }

    public IReadOnlyList<string> OffsetKeys()
    {
        return _namesByIndex.ToArray();
    }

    public ulong DataLength()
    {
        return _tensors.Count == 0 ? 0UL : _tensors[^1].DataOffsets.End;
    }

    internal void Validate()
    {
        ulong start = 0;

        for (var i = 0; i < _tensors.Count; i++)
        {
            var info = _tensors[i];
            var (s, e) = info.DataOffsets;

            if (s != start || e < s)
            {
                throw SafetensorException.InvalidOffset(_namesByIndex[i]);
            }

            start = e;

            var size = SafetensorMath.ComputeSizeInBytes(info.DataType, info.Shape, allowMisaligned: false);
            if (e - s != size)
            {
                throw SafetensorException.TensorInvalidInfo();
            }
        }
    }
}
