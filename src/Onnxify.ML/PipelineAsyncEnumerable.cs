using System.Collections;
using System.Runtime.CompilerServices;

namespace Onnxify.ML;

internal interface ICountAwareAsyncEnumerable
{
    int? KnownCount { get; }
}

internal static class PipelineAsyncEnumerable
{
    public static IAsyncEnumerable<T> FromEnumerable<T>(IEnumerable<T> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return WithKnownCount(Enumerate(source), TryGetEnumerableCount(source));
    }

    public static IAsyncEnumerable<T> FromSingle<T>(T item)
    {
        return WithKnownCount(Single(item), 1);
    }

    public static IAsyncEnumerable<T> WithKnownCount<T>(IAsyncEnumerable<T> source, int? knownCount)
    {
        ArgumentNullException.ThrowIfNull(source);

        return source is CountAwareAsyncEnumerable<T> existing && existing.KnownCount == knownCount
            ? existing
            : new CountAwareAsyncEnumerable<T>(source, knownCount);
    }

    public static bool TryGetKnownCount<T>(IAsyncEnumerable<T> source, out int count)
    {
        ArgumentNullException.ThrowIfNull(source);

        if (source is ICountAwareAsyncEnumerable countAware && countAware.KnownCount is { } knownCount)
        {
            count = knownCount;
            return true;
        }

        count = 0;
        return false;
    }

    private static int? TryGetEnumerableCount<T>(IEnumerable<T> source)
    {
        return source switch
        {
            IReadOnlyCollection<T> readOnlyCollection => readOnlyCollection.Count,
            ICollection<T> collection => collection.Count,
            ICollection nonGenericCollection => nonGenericCollection.Count,
            _ when Enumerable.TryGetNonEnumeratedCount(source, out var count) => count,
            _ => null
        };
    }

    private static async IAsyncEnumerable<T> Enumerate<T>(
        IEnumerable<T> source,
        [EnumeratorCancellation] CancellationToken token = default)
    {
        foreach (var item in source)
        {
            token.ThrowIfCancellationRequested();
            yield return item;
            await Task.Yield();
        }
    }

    private static async IAsyncEnumerable<T> Single<T>(
        T item,
        [EnumeratorCancellation] CancellationToken token = default)
    {
        token.ThrowIfCancellationRequested();
        yield return item;
        await Task.CompletedTask;
    }

    private sealed class CountAwareAsyncEnumerable<T> : IAsyncEnumerable<T>, ICountAwareAsyncEnumerable
    {
        private readonly IAsyncEnumerable<T> _source;

        public CountAwareAsyncEnumerable(IAsyncEnumerable<T> source, int? knownCount)
        {
            _source = source;
            KnownCount = knownCount;
        }

        public int? KnownCount { get; }

        public IAsyncEnumerator<T> GetAsyncEnumerator(CancellationToken cancellationToken = default)
        {
            return _source.GetAsyncEnumerator(cancellationToken);
        }
    }
}
