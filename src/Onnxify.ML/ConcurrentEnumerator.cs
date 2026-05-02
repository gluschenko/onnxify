using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace Onnxify.ML;

/// <summary>
/// Executes asynchronous item work concurrently while optionally preserving source order.
/// </summary>
public sealed class ConcurrentEnumerator<TInput, TOutput>
{
    private readonly IAsyncEnumerable<TInput> _input;
    private readonly Func<TInput, CancellationToken, ValueTask<TOutput>> _action;
    private readonly ConcurrentEnumeratorOptions _options;

    public ConcurrentEnumerator(
        IAsyncEnumerable<TInput> input,
        Func<TInput, CancellationToken, ValueTask<TOutput>> action,
        ConcurrentEnumeratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(action);

        _input = input;
        _action = action;
        _options = options ?? new ConcurrentEnumeratorOptions();

        if (_options.BoundedCapacity is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "BoundedCapacity must be greater than zero when provided.");
        }
    }

    public ConcurrentEnumerator(
        IAsyncEnumerable<TInput> input,
        Func<TInput, CancellationToken, Task<TOutput>> action,
        ConcurrentEnumeratorOptions? options = null)
        : this(
            input,
            (Func<TInput, CancellationToken, ValueTask<TOutput>>)((item, token) => new ValueTask<TOutput>(action(item, token))),
            options)
    {
        ArgumentNullException.ThrowIfNull(action);
    }

    public ConcurrentEnumerator(
        IEnumerable<TInput> input,
        Func<TInput, CancellationToken, Task<TOutput>> action,
        ConcurrentEnumeratorOptions? options = null)
        : this(
            PipelineAsyncEnumerable.FromEnumerable(input),
            (Func<TInput, CancellationToken, ValueTask<TOutput>>)((item, token) => new ValueTask<TOutput>(action(item, token))),
            options)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(action);
    }

    public async IAsyncEnumerable<TOutput> ExecuteAsync([EnumeratorCancellation] CancellationToken token = default)
    {
        var maxDegree = _options.MaxDegreeOfParallelism;
        var capacity = _options.BoundedCapacity ?? maxDegree;

        var inputChannel = Channel.CreateBounded<(int Index, TInput Item)>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = false,
            SingleWriter = true
        });

        var outputChannel = Channel.CreateBounded<(int Index, TOutput Value)>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false
        });

        var producer = Task.Run(async () =>
        {
            var index = 0;

            try
            {
                await foreach (var item in _input.WithCancellation(token))
                {
                    await inputChannel.Writer.WriteAsync((index, item), token);
                    index++;
                }

                inputChannel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                inputChannel.Writer.TryComplete(ex);
                throw;
            }
        }, token);

        var workers = Enumerable.Range(0, maxDegree)
            .Select(_ => Task.Run(async () =>
            {
                await foreach (var (index, item) in inputChannel.Reader.ReadAllAsync(token))
                {
                    var result = await _action(item, token);
                    await outputChannel.Writer.WriteAsync((index, result), token);
                }
            }, token))
            .ToArray();

        var completion = Task.Run(async () =>
        {
            try
            {
                await producer;
                await Task.WhenAll(workers);
                outputChannel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                outputChannel.Writer.TryComplete(ex);
                throw;
            }
        }, token);

        if (!_options.PreserveOrder)
        {
            await foreach (var (_, result) in outputChannel.Reader.ReadAllAsync(token))
            {
                yield return result;
            }

            await completion;
            yield break;
        }

        var buffer = new SortedDictionary<int, TOutput>();
        var nextIndex = 0;

        await foreach (var (index, result) in outputChannel.Reader.ReadAllAsync(token))
        {
            buffer[index] = result;

            while (buffer.TryGetValue(nextIndex, out var ready))
            {
                buffer.Remove(nextIndex);
                yield return ready;
                nextIndex++;
            }
        }

        await completion;
    }
}
