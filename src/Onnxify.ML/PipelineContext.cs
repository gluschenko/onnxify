using System.Collections.Concurrent;

namespace Onnxify.ML;

/// <summary>
/// Shared ambient state for a single pipeline execution.
/// </summary>
public sealed class PipelineContext
{
    private readonly Dictionary<string, object?> _properties = new(StringComparer.Ordinal);
    private readonly Dictionary<Type, object> _typedState = new();
    private readonly ConcurrentDictionary<string, int> _sequenceCounters = new(StringComparer.Ordinal);
    private ProgressChangeEvent? _progressChangeEvent;

    /// <summary>
    /// Initializes a new execution context with optional service-provider access.
    /// </summary>
    public PipelineContext(IServiceProvider? services = null)
    {
        Services = services;
    }

    /// <summary>
    /// Gets an empty reusable context template.
    /// </summary>
    public static PipelineContext Empty { get; } = new();

    /// <summary>
    /// Gets the ambient service provider available to stages.
    /// </summary>
    public IServiceProvider? Services { get; }

    /// <summary>
    /// Gets the named property bag stored in the context.
    /// </summary>
    public IReadOnlyDictionary<string, object?> Properties => _properties;

    /// <summary>
    /// Stores typed state in the context.
    /// </summary>
    public PipelineContext Set<TState>(TState value)
        where TState : notnull
    {
        _typedState[typeof(TState)] = value;
        return this;
    }

    /// <summary>
    /// Stores a named value in the context.
    /// </summary>
    public PipelineContext Set(string key, object? value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        _properties[key] = value;
        return this;
    }

    /// <summary>
    /// Attempts to retrieve typed state from the context.
    /// </summary>
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

    /// <summary>
    /// Retrieves typed state or throws when it is missing.
    /// </summary>
    public TState GetRequired<TState>()
        where TState : notnull
    {
        if (TryGet<TState>(out var value) && value is not null)
        {
            return value;
        }

        throw new KeyNotFoundException($"Pipeline context does not contain state of type '{typeof(TState).FullName}'.");
    }

    /// <summary>
    /// Attempts to retrieve a named value from the context.
    /// </summary>
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

    /// <summary>
    /// Retrieves a named value or returns <see langword="default"/> when it is missing.
    /// </summary>
    public TValue? Get<TValue>(string key)
    {
        return TryGet<TValue>(key, out var value) ? value : default;
    }

    /// <summary>
    /// Resolves a service from the ambient service provider.
    /// </summary>
    public TService? GetService<TService>()
        where TService : class
    {
        return Services?.GetService(typeof(TService)) as TService;
    }

    /// <summary>
    /// Resolves a service or throws when it is unavailable.
    /// </summary>
    public TService GetRequiredService<TService>()
        where TService : class
    {
        return GetService<TService>()
            ?? throw new InvalidOperationException($"Pipeline service '{typeof(TService).FullName}' is not available.");
    }

    internal PipelineContext CreateExecutionContext(ProgressChangeEvent? progressChangeEvent)
    {
        var clone = new PipelineContext(Services);

        foreach (var pair in _properties)
        {
            clone._properties.Add(pair.Key, pair.Value);
        }

        foreach (var pair in _typedState)
        {
            clone._typedState.Add(pair.Key, pair.Value);
        }

        clone._progressChangeEvent = progressChangeEvent;
        return clone;
    }

    internal ValueTask ReportProgressAsync(PipelineStage stage, int current, int total)
    {
        return _progressChangeEvent is null
            ? ValueTask.CompletedTask
            : _progressChangeEvent.Invoke(stage, current, total);
    }

    /// <summary>
    /// Returns the next monotonically increasing sequence number for the specified stage.
    /// </summary>
    public int NextSequenceNumber(PipelineStage stage, string? suffix = null)
    {
        ArgumentNullException.ThrowIfNull(stage);

        var key = suffix is null
            ? $"{stage.GetType().FullName}:{stage.Name}"
            : $"{stage.GetType().FullName}:{stage.Name}:{suffix}";

        return _sequenceCounters.AddOrUpdate(key, 0, static (_, current) => checked(current + 1));
    }
}
