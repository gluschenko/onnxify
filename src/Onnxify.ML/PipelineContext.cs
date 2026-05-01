namespace Onnxify.ML;

/// <summary>
/// Shared ambient state for a single pipeline execution.
/// </summary>
public sealed class PipelineContext
{
    private readonly Dictionary<string, object?> _properties = new(StringComparer.Ordinal);
    private readonly Dictionary<Type, object> _typedState = new();

    public PipelineContext(IServiceProvider? services = null)
    {
        Services = services;
    }

    public static PipelineContext Empty { get; } = new();

    public IServiceProvider? Services { get; }

    public IReadOnlyDictionary<string, object?> Properties => _properties;

    public PipelineContext Set<TState>(TState value)
        where TState : notnull
    {
        _typedState[typeof(TState)] = value;
        return this;
    }

    public PipelineContext Set(string key, object? value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        _properties[key] = value;
        return this;
    }

    public bool TryGet<TState>(out TState? value)
    {
        if (_typedState.TryGetValue(typeof(TState), out var raw) && raw is TState typed)
        {
            value = typed;
            return true;
        }

        value = default;
        return false;
    }

    public TState GetRequired<TState>()
        where TState : notnull
    {
        if (TryGet<TState>(out var value) && value is not null)
        {
            return value;
        }

        throw new KeyNotFoundException($"Pipeline context does not contain state of type '{typeof(TState).FullName}'.");
    }

    public bool TryGet<TValue>(string key, out TValue? value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);

        if (_properties.TryGetValue(key, out var raw) && raw is TValue typed)
        {
            value = typed;
            return true;
        }

        value = default;
        return false;
    }

    public TValue? Get<TValue>(string key)
    {
        return TryGet<TValue>(key, out var value) ? value : default;
    }

    public TService? GetService<TService>()
        where TService : class
    {
        return Services?.GetService(typeof(TService)) as TService;
    }

    public TService GetRequiredService<TService>()
        where TService : class
    {
        return GetService<TService>()
            ?? throw new InvalidOperationException($"Pipeline service '{typeof(TService).FullName}' is not available.");
    }
}
