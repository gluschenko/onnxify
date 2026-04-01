namespace Onnxify.TorchSharp;

public sealed class TorchModuleExportState
{
    private readonly Dictionary<string, int> _counters = new(StringComparer.Ordinal);

    public string Next(string prefix)
    {
        if (!_counters.TryGetValue(prefix, out var index))
        {
            _counters[prefix] = 1;
            return $"{prefix}0";
        }

        _counters[prefix] = index + 1;
        return $"{prefix}{index}";
    }
}

