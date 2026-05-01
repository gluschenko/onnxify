using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace Onnxify.ML;

/// <summary>
/// Executes asynchronous item work concurrently while optionally preserving source order.
/// </summary>
public sealed class ConcurrentEnumerator<TInput, TOutput>
{
    private readonly IReadOnlyList<TInput> _input;
    private readonly Func<TInput, CancellationToken, Task<TOutput>> _action;
    private readonly ConcurrentEnumeratorOptions _options;

    public ConcurrentEnumerator(
        IEnumerable<TInput> input,
        Func<TInput, CancellationToken, Task<TOutput>> action,
        ConcurrentEnumeratorOptions? options = null
    )
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(action);

        _input = input as IReadOnlyList<TInput> ?? input.ToArray();
        _action = action;
        _options = options ?? new ConcurrentEnumeratorOptions();

        if (_options.BoundedCapacity is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "BoundedCapacity must be greater than zero when provided.");
        }
    }

    public async IAsyncEnumerable<TOutput> ExecuteAsync([EnumeratorCancellation] CancellationToken token = default)
    {
        if (_input.Count == 0)
        {
            yield break;
        }

        var maxDegree = Math.Min(_options.MaxDegreeOfParallelism, _input.Count);
        var capacity = _options.BoundedCapacity ?? maxDegree;
        using var semaphore = new SemaphoreSlim(maxDegree);

        var channel = Channel.CreateBounded<(int Index, TOutput Value)>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false
        });

        var workers = _input.Select((item, index) => ProcessAsync(item, index, semaphore, channel.Writer, token)).ToArray();

        var producer = Task.Run(
            async () =>
            {
                try
                {
                    await Task.WhenAll(workers);
                    channel.Writer.TryComplete();
                }
                catch (Exception ex)
                {
                    channel.Writer.TryComplete(ex);
                }
            }, 
            token
        );

        if (!_options.PreserveOrder)
        {
            await foreach (var (_, result) in channel.Reader.ReadAllAsync(token))
            {
                yield return result;
            }

            await producer;
            yield break;
        }

        var buffer = new SortedDictionary<int, TOutput>();
        var nextIndex = 0;

        await foreach (var (index, result) in channel.Reader.ReadAllAsync(token))
        {
            buffer[index] = result;

            while (buffer.TryGetValue(nextIndex, out var ready))
            {
                buffer.Remove(nextIndex);
                yield return ready;
                nextIndex++;
            }
        }

        await producer;
    }

    private async Task ProcessAsync(
        TInput item,
        int index,
        SemaphoreSlim semaphore,
        ChannelWriter<(int Index, TOutput Value)> writer,
        CancellationToken token)
    {
        await semaphore.WaitAsync(token);

        try
        {
            var result = await _action(item, token);
            await writer.WriteAsync((index, result), token);
        }
        finally
        {
            semaphore.Release();
        }
    }
}
