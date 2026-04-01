using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Onnxify.Examples
{
    internal sealed class DataReader : IDisposable
    {
        private readonly string _root;
        private readonly IList<torchvision.ITransform> _transforms;

        private readonly List<DatasetSample> _baseSamples = [];
        private readonly List<DatasetSample> _augmentedTrainSamples = [];
        private readonly List<DatasetSample> _trainSamples = [];
        private readonly List<DatasetSample> _testSamples = [];
        private readonly List<string> _labelNames = [];

        public DataReader(
            string root,
            IList<torchvision.ITransform>? transforms = null
        )
        {
            _root = root;
            _transforms = transforms ?? [];
        }

        public IReadOnlyList<string> LabelNames => _labelNames;

        public int ClassCount => _labelNames.Count;

        public int TrainSampleCount => _trainSamples.Count;

        public int TestSampleCount => _testSamples.Count;

        public void Load(
            int width,
            int height,
            int channels,
            int count
        )
        {
            if (!Directory.Exists(_root))
            {
                throw new DirectoryNotFoundException(_root);
            }

            DisposeGeneratedSamples();
            _baseSamples.ForEach(static sample => sample.Dispose());
            _baseSamples.Clear();
            _labelNames.Clear();

            var labelDirs = Directory
                .GetDirectories(_root)
                .OrderBy(d => d, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            _labelNames.AddRange(labelDirs.Select(Path.GetFileName).Where(static x => !string.IsNullOrWhiteSpace(x))!);

            for (var labelIndex = 0; labelIndex < labelDirs.Length; labelIndex++)
            {
                var dir = labelDirs[labelIndex];
                var files = Directory
                    .EnumerateFiles(dir, "*.*", SearchOption.TopDirectoryOnly)
                    .Where(static f =>
                        f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                        f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                        f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                    .Take(count)
                    .ToArray();

                foreach (var file in files)
                {
                    try
                    {
                        using var stream = File.OpenRead(file);
                        using var bitmap = SKBitmap.Decode(stream);

                        if (bitmap == null)
                        {
                            continue;
                        }

                        using var resized = bitmap.Resize(
                            new SKImageInfo(width, height),
                            SKFilterQuality.Medium
                        );

                        if (resized == null)
                        {
                            continue;
                        }

                        var tensor = ImageToTensor(resized, channels);
                        var labelTensor = torch.tensor(labelIndex, dtype: ScalarType.Int64);
                        _baseSamples.Add(new DatasetSample(tensor, labelTensor, labelIndex));
                    }
                    catch
                    {
                        continue;
                    }
                }
            }
        }

        public void Split(
            float testFraction = 0.2f,
            int seed = 42
        )
        {
            if (_baseSamples.Count == 0)
            {
                return;
            }

            DisposeGeneratedSamples();

            var rng = new Random(seed);

            foreach (var group in _baseSamples
                .Select((sample, index) => new { sample, index })
                .GroupBy(x => x.sample.LabelIndex)
                .OrderBy(x => x.Key))
            {
                var indices = group
                    .Select(x => x.index)
                    .OrderBy(_ => rng.Next())
                    .ToArray();

                var requestedTestCount = (int)Math.Round(indices.Length * testFraction, MidpointRounding.AwayFromZero);
                var testCount = indices.Length <= 1
                    ? 0
                    : Math.Clamp(requestedTestCount, 1, indices.Length - 1);

                foreach (var index in indices.Take(testCount))
                {
                    _testSamples.Add(_baseSamples[index]);
                }

                foreach (var index in indices.Skip(testCount))
                {
                    var baseSample = _baseSamples[index];
                    _trainSamples.Add(baseSample);

                    foreach (var transform in _transforms)
                    {
                        var transformedTensor = transform.call(baseSample.Data);
                        var labelTensor = torch.tensor(baseSample.LabelIndex, dtype: ScalarType.Int64);
                        var augmentedSample = new DatasetSample(
                            transformedTensor,
                            labelTensor,
                            baseSample.LabelIndex
                        );

                        _augmentedTrainSamples.Add(augmentedSample);
                        _trainSamples.Add(augmentedSample);
                    }
                }
            }
        }

        public IEnumerable<Batch> GetTrainingBatches(
            int batchSize,
            bool shuffle = true,
            int? seed = null
        )
        {
            return GetBatches(_trainSamples, batchSize, shuffle, seed);
        }

        public IEnumerable<Batch> GetTestBatches(
            int batchSize
        )
        {
            return GetBatches(_testSamples, batchSize, shuffle: false, seed: null);
        }

        public void Dispose()
        {
            DisposeGeneratedSamples();
            _baseSamples.ForEach(static sample => sample.Dispose());
            _baseSamples.Clear();
        }

        private IEnumerable<Batch> GetBatches(
            IReadOnlyList<DatasetSample> samples,
            int batchSize,
            bool shuffle,
            int? seed
        )
        {
            if (samples.Count == 0)
            {
                yield break;
            }

            var ordered = samples.ToArray();
            if (shuffle)
            {
                var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
                for (var i = ordered.Length - 1; i > 0; i--)
                {
                    var swapIndex = rng.Next(i + 1);
                    (ordered[i], ordered[swapIndex]) = (ordered[swapIndex], ordered[i]);
                }
            }

            for (var i = 0; i < ordered.Length; i += batchSize)
            {
                var size = Math.Min(batchSize, ordered.Length - i);
                var batchData = new List<Tensor>(size);
                var batchLabels = new List<Tensor>(size);
                var labelIndices = new int[size];

                for (var j = 0; j < size; j++)
                {
                    var sample = ordered[i + j];
                    batchData.Add(sample.Data);
                    batchLabels.Add(sample.LabelTensor);
                    labelIndices[j] = sample.LabelIndex;
                }

                yield return new Batch(batchData, batchLabels, labelIndices);
            }
        }

        private void DisposeGeneratedSamples()
        {
            _augmentedTrainSamples.ForEach(static sample => sample.Dispose());
            _augmentedTrainSamples.Clear();
            _trainSamples.Clear();
            _testSamples.Clear();
        }

        private static Tensor ImageToTensor(SKBitmap bitmap, int channels)
        {
            var width = bitmap.Width;
            var height = bitmap.Height;
            var data = new float[channels * height * width];

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var color = bitmap.GetPixel(x, y);
                    var index = y * width + x;

                    if (channels == 3)
                    {
                        data[index] = color.Red / 255f;
                        data[height * width + index] = color.Green / 255f;
                        data[2 * height * width + index] = color.Blue / 255f;
                    }
                    else if (channels == 1)
                    {
                        data[index] = (color.Red + color.Green + color.Blue) / 3f / 255f;
                    }
                    else
                    {
                        throw new NotImplementedException($"Not implemented for {nameof(channels)} = {channels}");
                    }
                }
            }

            return torch.tensor(data, [channels, height, width]);
        }

        internal sealed class Batch
        {
            public Batch(
                List<Tensor> data,
                List<Tensor> labels,
                int[] labelIndices
            )
            {
                Data = data;
                Labels = labels;
                LabelIndices = labelIndices;
            }

            public List<Tensor> Data { get; }

            public List<Tensor> Labels { get; }

            public int[] LabelIndices { get; }
        }

        private sealed class DatasetSample : IDisposable
        {
            public DatasetSample(
                Tensor data,
                Tensor labelTensor,
                int labelIndex
            )
            {
                Data = data;
                LabelTensor = labelTensor;
                LabelIndex = labelIndex;
            }

            public Tensor Data { get; }

            public Tensor LabelTensor { get; }

            public int LabelIndex { get; }

            public void Dispose()
            {
                Data.Dispose();
                LabelTensor.Dispose();
            }
        }
    }
}
