using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace Onnxify.Examples;

internal sealed class DataReader
{
    private readonly string _root;
    private readonly int _width;
    private readonly int _height;
    private readonly int _channels;
    private readonly int _count;
    private readonly string[] _labelNames;
    private readonly Dictionary<string, int> _labelToIndex;
    private readonly SampleInfo[] _samples;
    private readonly string _cacheDirectory;
    private readonly string _datasetCacheDirectory;
    private readonly int _sampleElementCount;

    private long _sampleCount;

    public DataReader(
        string root,
        int width,
        int height,
        int channels,
        int count
    )
    {
        if (!Directory.Exists(root))
        {
            throw new DirectoryNotFoundException(root);
        }

        _root = root;
        _width = width;
        _height = height;
        _channels = channels;
        _count = count;

        _labelNames = Directory
            .GetDirectories(_root)
            .Select(x => Path.GetFileName(x) ?? "")
            .OrderBy(x => x, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        _labelToIndex = _labelNames
            .Select((label, index) => new KeyValuePair<string, int>(label, index))
            .ToDictionary(x => x.Key, x => x.Value, StringComparer.OrdinalIgnoreCase);

        _cacheDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ".temp");
        _datasetCacheDirectory = Path.Combine(_cacheDirectory, MD5(_root));
        _sampleElementCount = _channels * _height * _width;
        _samples = BuildSampleIndex();
        _sampleCount = _samples.Length;
    }

    public IReadOnlyList<string> LabelNames => _labelNames;

    public long ClassCount => _labelNames.Length;
    public long SampleCount => _sampleCount;

    public async IAsyncEnumerable<LoadingProgress> Convert()
    {
        if (!Directory.Exists(_root))
        {
            throw new DirectoryNotFoundException(_root);
        }

        Directory.CreateDirectory(_cacheDirectory);
        Directory.CreateDirectory(_datasetCacheDirectory);

        var current = 0L;
        var failed = 0L;
        var total = _samples.Length;

        foreach (var sampleChunk in _samples.Chunk(Environment.ProcessorCount * 2))
        {
            var tasks = sampleChunk.Select(async sample =>
            {
                Interlocked.Increment(ref current);

                if (File.Exists(sample.CachePath))
                {
                    return;
                }

                try
                {
                    using var stream = File.OpenRead(sample.SourcePath);
                    using var bitmap = SKBitmap.Decode(stream);

                    if (bitmap == null)
                    {
                        Interlocked.Increment(ref failed);
                        return;
                    }

                    using var resized = bitmap.Resize(
                        new SKImageInfo(_width, _height),
                        SKFilterQuality.Medium
                    );

                    if (resized == null)
                    {
                        Interlocked.Increment(ref failed);
                        return;
                    }

                    var imageData = ImageData.FromBitmap(resized, _channels);

                    Directory.CreateDirectory(Path.GetDirectoryName(sample.CachePath) ?? "");
                    await File.WriteAllBytesAsync(sample.CachePath, imageData.Data);
                }
                catch (Exception ex)
                {
                    Interlocked.Increment(ref failed);
                    Console.WriteLine($"Failed to convert '{sample.SourcePath}': {ex}");
                }
            });

            await Task.WhenAll(tasks);

            yield return new LoadingProgress(current, failed, total);
        }
    }

    public async IAsyncEnumerable<Batch> BatchAsync(
        int batchSize,
        bool shuffle = false,
        int shuffleSeed = 42,
        int prefetchBatches = 2,
        [EnumeratorCancellation] CancellationToken cancellationToken = default
    )
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(prefetchBatches);

        if (!Directory.Exists(_root))
        {
            throw new DirectoryNotFoundException(_root);
        }

        var sampleIndices = Enumerable.Range(0, _samples.Length).ToArray();
        if (shuffle)
        {
            Shuffle(sampleIndices, shuffleSeed);
        }

        var totalBatches = (sampleIndices.Length + batchSize - 1) / batchSize;
        var queue = new Queue<Task<Batch>>(prefetchBatches);

        Task<Batch> StartBatchLoad(int batchIndex)
        {
            return LoadBatchAsync(batchIndex, batchSize, sampleIndices, cancellationToken);
        }

        var nextBatchIndex = 0;

        while (nextBatchIndex < totalBatches && queue.Count < prefetchBatches)
        {
            queue.Enqueue(StartBatchLoad(nextBatchIndex));
            nextBatchIndex++;
        }

        while (queue.Count > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var currentTask = queue.Dequeue();

            if (nextBatchIndex < totalBatches)
            {
                queue.Enqueue(StartBatchLoad(nextBatchIndex));
                nextBatchIndex++;
            }

            var batch = await currentTask;
            yield return batch;
        }
    }

    private async Task<Batch> LoadBatchAsync(
        int batchIndex,
        int batchSize,
        int[] sampleIndices,
        CancellationToken cancellationToken
    )
    {
        var batchStart = batchIndex * batchSize;
        var currentBatchSize = Math.Min(batchSize, sampleIndices.Length - batchStart);

        var totalElements = (long)currentBatchSize * _sampleElementCount;
        var batchData = new float[checked((int)totalElements)];
        var labels = new int[currentBatchSize];

        await Parallel.ForAsync(
            fromInclusive: 0,
            toExclusive: currentBatchSize,
            parallelOptions: new ParallelOptions
            {
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = Environment.ProcessorCount * 2,
            },
            body: async (batchOffset, ct) =>
            {
                var sample = _samples[sampleIndices[batchStart + batchOffset]];

                if (!File.Exists(sample.CachePath))
                {
                    // throw new FileNotFoundException($"Cache file not found: {sample.CachePath}");
                    return;
                }

                var bytes = await File.ReadAllBytesAsync(sample.CachePath, ct).ConfigureAwait(false);

                if (bytes.Length != _sampleElementCount)
                {
                    throw new InvalidOperationException(
                        $"Unexpected cached sample size {bytes.Length} for '{sample.SourcePath}'. Expected {_sampleElementCount}."
                    );
                }

                const float SCALE = 1f / 255f;
                var destinationOffset = (long)batchOffset * (long)_sampleElementCount;

                for (var i = 0; i < bytes.Length; i++)
                {
                    var offset = destinationOffset + i;

                    if (offset < batchData.Length)
                    {
                        batchData[offset] = bytes[i] * SCALE;
                    }
                }

                labels[batchOffset] = sample.LabelIndex;
            }
        );

        return new Batch(batchData, labels, _channels, _height, _width);
    }

    private SampleInfo[] BuildSampleIndex()
    {
        var random = new Random(42);
        return Directory
            .EnumerateFiles(_root, "*.*", SearchOption.AllDirectories)
            .Where(static path =>
                path.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                path.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                path.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
            )
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .Select(path => (Path: path, Key: random.NextDouble()))
            .OrderBy(x => x.Key)
            .Take(_count)
            .Select(x => x.Path)
            .Select(path =>
            {
                var label = GetTopLevelLabel(path);
                if (!_labelToIndex.TryGetValue(label, out var labelIndex))
                {
                    throw new InvalidOperationException($"No label mapping for '{label}' from '{path}'.");
                }

                return new SampleInfo(
                    SourcePath: path,
                    CachePath: GetCachePath(path),
                    LabelIndex: labelIndex
                );
            })
            .ToArray();
    }

    private string GetTopLevelLabel(string path)
    {
        var relativePath = Path.GetRelativePath(_root, path);
        var firstSeparator = relativePath.IndexOfAny([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar]);
        if (firstSeparator <= 0)
        {
            throw new InvalidOperationException($"Could not resolve label folder for '{path}'.");
        }

        return relativePath[..firstSeparator];
    }

    private string GetCachePath(string sourcePath)
    {
        var hash = MD5(sourcePath);
        var prefix = hash.Substring(0, 3);

        return Path.Combine(_datasetCacheDirectory, prefix, $"{hash}_{_width}x{_height}x{_channels}");
    }

    private static void Shuffle<T>(T[] values, int seed)
    {
        var random = new Random(seed);
        for (var i = values.Length - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    internal sealed class Batch
    {
        private readonly float[] _data;
        private readonly int[] _labels;
        private readonly long[] _shape;
        private readonly int[] _denseShape;

        public IReadOnlyList<int> LabelIndices => _labels;

        public Batch(
            float[] data,
            int[] labels,
            int channels,
            int height,
            int width
        )
        {
            _data = data;
            _labels = labels;
            _shape = [labels.Length, channels, height, width];
            _denseShape = [labels.Length, channels, height, width];
        }

        public int Size => _labels.Length;

        public Tensor GetDataTensor(Device device)
        {
            return torch.tensor(_data, _shape, device: device);
        }

        public Tensor GetLabelTensor(Device device)
        {
            return torch.tensor(_labels, dtype: ScalarType.Int64, device: device);
        }

        public DenseTensor<float> GetDenseTensor()
        {
            return new DenseTensor<float>(_data, _denseShape);
        }
    }

    private sealed record SampleInfo(
        string SourcePath,
        string CachePath,
        int LabelIndex
    );

    public static string MD5(string input)
    {
        var inputBytes = Encoding.UTF8.GetBytes(input);
        var hashBytes = System.Security.Cryptography.MD5.HashData(inputBytes);
        return System.Convert.ToHexString(hashBytes);
    }
}

public class ImageData
{
    public required int Width { get; set; }
    public required int Height { get; set; }
    public required int Channels { get; set; }
    public required byte[] Data { get; set; }

    public static ImageData FromBitmap(SKBitmap bitmap, int channels)
    {
        var width = bitmap.Width;
        var height = bitmap.Height;
        var data = new byte[channels * height * width];

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var color = bitmap.GetPixel(x, y);
                var index = y * width + x;

                if (channels == 3)
                {
                    data[index] = color.Red;
                    data[height * width + index] = color.Green;
                    data[2 * height * width + index] = color.Blue;
                }
                else if (channels == 1)
                {
                    data[index] = (byte)(((int)color.Red + (int)color.Green + (int)color.Blue) / 3);
                }
                else
                {
                    throw new NotImplementedException($"Not implemented for {nameof(channels)} = {channels}");
                }
            }
        }

        return new ImageData
        {
            Width = width,
            Height = height,
            Channels = channels,
            Data = data,
        };
    }
}

public record LoadingProgress(long Current, long Failed, long All);
