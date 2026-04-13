using System.Text;
using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Onnxify.Examples;

internal sealed class DataReader
{
    private readonly string _root;
    private readonly int _width;
    private readonly int _height;
    private readonly int _channels;
    private readonly int _count;
    private readonly string[] _labelNames;

    private long _sampleCount;
    private string _cacheDirectory;

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

        _cacheDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ".temp");
    }

    public IReadOnlyList<string> LabelNames => _labelNames;

    public long ClassCount => _labelNames.Length;
    public long SampleCount => _sampleCount;

    private IEnumerable<string> GetFiles(string dir)
    {
        return Directory
            .EnumerateFiles(dir, "*.*", SearchOption.AllDirectories)
            .Where(f =>
            {
                return 
                    f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                    f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                    f.EndsWith(".png", StringComparison.OrdinalIgnoreCase);
            })
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .Take(_count);
    }

    public async IAsyncEnumerable<LoadingProgress> Convert()
    {
        if (!Directory.Exists(_root))
        {
            throw new DirectoryNotFoundException(_root);
        }

        Directory.CreateDirectory(_cacheDirectory);

        var total = 0L;
        foreach (var x in GetFiles(_root))
        {
            total++;
        }

        _sampleCount = total;

        var current = 0L;
        var failed = 0L;
        foreach (var x in GetFiles(_root))
        {
            current++;

            var cachePath = GetCachePath(x);
            if (File.Exists(cachePath))
            {
                yield return new LoadingProgress(current, failed, total);
                continue;
            }

            var cacheDir = Path.GetDirectoryName(cachePath);
            if (cacheDir is not null)
            {
                Directory.CreateDirectory(cacheDir);
            }

            try
            {
                using var stream = File.OpenRead(x);
                using var bitmap = SKBitmap.Decode(stream);

                if (bitmap == null)
                {
                    continue;
                }

                using var resized = bitmap.Resize(
                    new SKImageInfo(_width, _height),
                    SKFilterQuality.Medium
                );

                if (resized == null)
                {
                    continue;
                }

                var imageData = ImageData.FromBitmap(resized, _channels);
                await File.WriteAllBytesAsync(cachePath, imageData.Data);
            }
            catch
            {
                failed++;
            }

            yield return new LoadingProgress(current, failed, total);
        }
    }

    private string GetCachePath(string x)
    {
        return Path.Combine(_cacheDirectory, MD5(_root), $"{MD5(x)}_{_width}x{_height}x{_channels}");
    }

    public async IAsyncEnumerable<Batch> BatchAsync(
        int batchSize,
        bool shuffle = false,
        int? shuffleSeed = null
    )
    {
        if (!Directory.Exists(_root))
        {
            throw new DirectoryNotFoundException(_root);
        }

        var imagePaths = GetFiles(_root).ToArray();
        if (shuffle)
        {
            Shuffle(imagePaths, shuffleSeed ?? 42);
        }

        foreach (var batchImagePaths in imagePaths.Chunk(batchSize))
        {
            var images = new List<ImageData>();
            var labels = new List<int>();

            foreach (var imagePath in batchImagePaths)
            {
                var label = Path.GetFileName(Path.GetDirectoryName(imagePath) ?? "") ?? "";
                var cachePath = GetCachePath(imagePath);

                if (!File.Exists(cachePath))
                {
                    throw new Exception($"No file in cache: {cachePath}");
                }

                if (!_labelNames.Contains(label))
                {
                    throw new Exception($"No label: {label}");
                }

                var data = File.ReadAllBytes(cachePath);

                images.Add(new ImageData
                {
                    Width = _width,
                    Height = _height,
                    Channels = _channels,
                    Data = data,
                });

                labels.Add(_labelNames.IndexOf(label));
            }

            yield return new Batch(images, labels);
        }
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
        private readonly IReadOnlyList<ImageData> _samples;
        private readonly IReadOnlyList<int> _labels;

        public IReadOnlyList<int> LabelIndices => _labels;

        public Batch(IReadOnlyList<ImageData> samples, IReadOnlyList<int> labels)
        {
            _samples = samples;
            _labels = labels;
        }

        public int Size => _samples.Count;

        public Tensor GetDataTensor(Device device)
        {
            var tensors = _samples.Select(x => x.AsTensor()).ToArray();
            var x = torch.stack(tensors).to(device);

            foreach (var y in tensors)
            {
                y.Dispose();
            }

            return x;
        }

        public Tensor GetLabelTensor(Device device)
        {
            var labelTensor = torch.tensor(_labels.ToArray(), dtype: ScalarType.Int64);
            return labelTensor.to(device);
        }
    }

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

    public Tensor AsTensor()
    {
        var data = Data.Select(x => x / 256f).ToArray();
        return torch.tensor(data, [Channels, Height, Width]);
    }
}

public record LoadingProgress(long Current, long Failed, long All);
