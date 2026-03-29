using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Onnxify.Examples
{
    internal class DataReader : IDisposable
    {
        private readonly string _root;

        private readonly List<Tensor> _data = [];
        private readonly List<Tensor> _labels = [];

        private readonly IList<torchvision.ITransform> _transforms;

        public DataReader(
            string root,
            IList<torchvision.ITransform>? transforms = null
        )
        {
            _root = root;
            _transforms = transforms ?? [];
        }

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

            var labelDirs = Directory
                .GetDirectories(_root)
                .OrderBy(d => d)
                .ToArray();

            var labelMap = labelDirs
                .Select((dir, idx) => new { dir, idx })
                .ToDictionary(x => x.dir, x => x.idx);

            foreach (var dir in labelDirs)
            {
                var labelIndex = labelMap[dir];

                var files = Directory
                    .EnumerateFiles(dir, "*.*", SearchOption.TopDirectoryOnly)
                    .Take(count)
                    .Where(f =>
                    {
                        return 
                            f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase);
                    })
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

                        // Resize
                        using var resized = bitmap.Resize(
                            new SKImageInfo(width, height),
                            SKFilterQuality.Medium);

                        if (resized == null)
                            continue;

                        // Convert to tensor
                        var tensor = ImageToTensor(resized, channels);

                        // Label tensor (long)
                        var labelTensor = torch.tensor(labelIndex, dtype: ScalarType.Int64);

                        _data.Add(tensor);
                        _labels.Add(labelTensor);
                    }
                    catch
                    {
                        // dataset часто содержит битые изображения
                        continue;
                    }
                }
            }
        }

        public IEnumerable<(Tensor, Tensor)> Data()
        {
            for (var i = 0; i < _data.Count; i++)
            {
                yield return (_data[i], _labels[i]);

                foreach (var tfrm in _transforms)
                {
                    yield return (tfrm.call(_data[i]), _labels[i]);
                }
            }
        }

        public void Dispose()
        {
            _data.ForEach(d => d.Dispose());
            _labels.ForEach(d => d.Dispose());
        }

        private static Tensor ImageToTensor(SKBitmap bitmap, int channels)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;

            // CHW layout
            var data = new float[channels * height * width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var color = bitmap.GetPixel(x, y);

                    int idx = y * width + x;

                    if (channels >= 3)
                    {
                        data[0 * height * width + idx] = color.Red / 255f;
                        data[1 * height * width + idx] = color.Green / 255f;
                        data[2 * height * width + idx] = color.Blue / 255f;
                    }

                    if (channels == 1)
                    {
                        var gray = (color.Red + color.Green + color.Blue) / 3f / 255f;
                        data[idx] = gray;
                    }
                }
            }

            return torch.tensor(data, new long[] { channels, height, width });
        }
    }
}
