using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Onnxify.Examples.Models;

internal sealed class LanguageLstmTrainingPipeline
{
    private readonly LanguageTextSample[] _trainSamples;
    private readonly LanguageTextSample[] _validationSamples;
    private readonly int _maxSequenceLength;

    public LanguageLstmTrainingPipeline(
        string datasetPath,
        int maxSequenceLength,
        int trainSamplesPerLanguage,
        int validationSamplesPerLanguage,
        int samplingSeed = 1234
    )
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(trainSamplesPerLanguage);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(validationSamplesPerLanguage);

        DatasetPath = datasetPath;
        _maxSequenceLength = maxSequenceLength;

        var corpus = LoadCorpus(
            datasetPath,
            trainSamplesPerLanguage,
            validationSamplesPerLanguage,
            samplingSeed
        );

        _trainSamples = corpus.TrainSamples;
        _validationSamples = corpus.ValidationSamples;
        TotalCorpusSamples = corpus.TotalSamples;
        CharToIdx = BuildCharacterVocabulary(corpus.Characters);
        LangToIdx = BuildLanguageVocabulary(_trainSamples.Concat(_validationSamples));
    }

    public string DatasetPath { get; }
    public long TotalCorpusSamples { get; }
    public Dictionary<string, int> CharToIdx { get; }
    public Dictionary<string, int> LangToIdx { get; }

    public async Task<LanguageLstmTrainingResult> TrainAsync(
        int embeddingDim,
        int hiddenDim,
        int layers,
        int epochs,
        int batchSize,
        float learningRate,
        Device device,
        int shuffleSeed = 1234
    )
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        var trainDataset = new LanguageLstmDataset(
            _trainSamples,
            CharToIdx,
            LangToIdx,
            _maxSequenceLength
        );

        var validationDataset = new LanguageLstmDataset(
            _validationSamples.Length == 0 ? _trainSamples : _validationSamples,
            CharToIdx,
            LangToIdx,
            _maxSequenceLength
        );

        var model = new LSTMLIDModel(
            new Dictionary<string, int>(CharToIdx),
            new Dictionary<string, int>(LangToIdx),
            LangToIdx.Count,
            embeddingDim,
            hiddenDim,
            layers
        );

        var trainer = new LanguageLstmTrainer(model, trainDataset, validationDataset);
        var finalEvaluation = await trainer.TrainAsync(
            epochs,
            batchSize,
            learningRate,
            device,
            shuffleSeed
        );

        return new LanguageLstmTrainingResult(
            model,
            trainDataset,
            validationDataset,
            finalEvaluation
        );
    }

    private static CorpusLoadResult LoadCorpus(
        string datasetPath,
        int trainSamplesPerLanguage,
        int validationSamplesPerLanguage,
        int samplingSeed
    )
    {
        if (!Directory.Exists(datasetPath))
        {
            throw new DirectoryNotFoundException(datasetPath);
        }

        var trainSamples = new List<LanguageTextSample>();
        var validationSamples = new List<LanguageTextSample>();
        var characters = new HashSet<string>(StringComparer.Ordinal);
        var totalSamples = 0L;
        var reservoirCapacity = trainSamplesPerLanguage + validationSamplesPerLanguage;

        var files = Directory
            .EnumerateFiles(datasetPath)
            .OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        foreach (var file in files)
        {
            var language = Path.GetFileName(file);
            if (string.IsNullOrWhiteSpace(language))
            {
                continue;
            }

            var reservoir = new List<string>(reservoirCapacity);
            var random = new Random(unchecked(samplingSeed + StableHash(language)));
            var seenForLanguage = 0;

            using var reader = new StreamReader(
                file,
                Encoding.UTF8,
                detectEncodingFromByteOrderMarks: true
            );

            while (reader.ReadLine() is { } line)
            {
                var normalized = Normalize(line);
                if (normalized.Length == 0)
                {
                    continue;
                }

                totalSamples++;
                seenForLanguage++;

                foreach (var ch in normalized)
                {
                    characters.Add(ch.ToString());
                }

                if (reservoir.Count < reservoirCapacity)
                {
                    reservoir.Add(normalized);
                    continue;
                }

                var replacementIndex = random.Next(seenForLanguage);
                if (replacementIndex < reservoirCapacity)
                {
                    reservoir[replacementIndex] = normalized;
                }
            }

            if (reservoir.Count == 0)
            {
                continue;
            }

            Shuffle(reservoir, random);

            var validationCount = reservoir.Count == 1
                ? 0
                : Math.Min(validationSamplesPerLanguage, reservoir.Count - 1);

            var trainCount = Math.Min(trainSamplesPerLanguage, reservoir.Count - validationCount);

            foreach (var text in reservoir.Take(trainCount))
            {
                trainSamples.Add(new LanguageTextSample(text, language));
            }

            foreach (var text in reservoir.Skip(trainCount).Take(validationCount))
            {
                validationSamples.Add(new LanguageTextSample(text, language));
            }
        }

        if (trainSamples.Count == 0)
        {
            throw new InvalidOperationException($"No training samples were read from '{datasetPath}'.");
        }

        return new CorpusLoadResult(
            trainSamples.ToArray(),
            validationSamples.ToArray(),
            characters,
            totalSamples
        );
    }

    private static Dictionary<string, int> BuildCharacterVocabulary(IEnumerable<string> characters)
    {
        var charToIdx = new Dictionary<string, int>
        {
            ["PAD"] = 0,
            ["UNK"] = 1,
        };

        foreach (var token in characters.OrderBy(ch => ch, StringComparer.Ordinal))
        {
            if (!charToIdx.ContainsKey(token))
            {
                charToIdx.Add(token, charToIdx.Count);
            }
        }

        return charToIdx;
    }

    private static Dictionary<string, int> BuildLanguageVocabulary(IEnumerable<LanguageTextSample> samples)
    {
        return samples
            .Select(sample => sample.Language)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(lang => lang, StringComparer.OrdinalIgnoreCase)
            .Select((lang, index) => new KeyValuePair<string, int>(lang, index))
            .ToDictionary(x => x.Key, x => x.Value, StringComparer.OrdinalIgnoreCase);
    }

    private static void Shuffle<T>(IList<T> values, Random random)
    {
        for (var i = values.Count - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    private static int StableHash(string value)
    {
        unchecked
        {
            var hash = 17;
            foreach (var ch in value.ToUpperInvariant())
            {
                hash = (hash * 31) + ch;
            }

            return hash;
        }
    }

    internal static string Normalize(string text)
    {
        return text.Trim().Normalize(NormalizationForm.FormC).ToLowerInvariant();
    }

    private sealed record CorpusLoadResult(
        LanguageTextSample[] TrainSamples,
        LanguageTextSample[] ValidationSamples,
        HashSet<string> Characters,
        long TotalSamples
    );
}

internal sealed class LanguageLstmTrainer
{
    private readonly LSTMLIDModel _model;
    private readonly LanguageLstmDataset _trainDataset;
    private readonly LanguageLstmDataset _validationDataset;

    public LanguageLstmTrainer(
        LSTMLIDModel model,
        LanguageLstmDataset trainDataset,
        LanguageLstmDataset validationDataset
    )
    {
        _model = model;
        _trainDataset = trainDataset;
        _validationDataset = validationDataset;
    }

    public async Task<LanguageLstmEvaluation> TrainAsync(
        int epochs,
        int batchSize,
        float learningRate,
        Device device,
        int shuffleSeed
    )
    {
        var stopwatch = Stopwatch.StartNew();

        _model.to(device);

        var optimizer = optim.Adam(_model.parameters(), learningRate);
        var criterion = CrossEntropyLoss();
        var finalEvaluation = new LanguageLstmEvaluation(0f, 0f, 0);

        for (var epoch = 1; epoch <= epochs; epoch++)
        {
            _model.train();

            var batchIndex = 0;
            var processedSamples = 0;
            var correctPredictions = 0;
            var lossSum = 0f;

            foreach (var batch in _trainDataset.Batches(
                batchSize,
                shuffle: true,
                shuffleSeed: shuffleSeed + epoch - 1
            ))
            {
                TrainBatch(
                    batch,
                    optimizer,
                    criterion,
                    device,
                    ref batchIndex,
                    ref processedSamples,
                    ref correctPredictions,
                    ref lossSum
                );

                Console.Write(
                    $"\r[T+{Math.Round(stopwatch.Elapsed.TotalSeconds)}s] " +
                    $"LanguageLSTM epoch {epoch}/{epochs} | " +
                    $"train loss {lossSum / Math.Max(1, processedSamples):0.000000} | " +
                    $"train acc {((float)correctPredictions / Math.Max(1, processedSamples)):0.000000} | " +
                    $"val loss {finalEvaluation.Loss:0.000000} | " +
                    $"val acc {finalEvaluation.Accuracy:0.000000}"
                );
            }

            Console.WriteLine();

            finalEvaluation = Evaluate(batchSize, device);
        }

        _model.eval();
        await Task.CompletedTask;
        return finalEvaluation;
    }

    private void TrainBatch(
        LanguageLstmDataset.Batch batch,
        optim.Optimizer optimizer,
        Loss<Tensor, Tensor, Tensor> criterion,
        Device device,
        ref int batchIndex,
        ref int processedSamples,
        ref int correctPredictions,
        ref float lossSum
    )
    {
        using var d = torch.NewDisposeScope();
        using var input = batch.GetInputTensor(device);
        using var labels = batch.GetLabelTensor(device);

        optimizer.zero_grad();

        using var output = _model.forward(input);
        using var loss = criterion.call(output, labels);

        loss.backward();
        optimizer.step();

        using var predicted = output.argmax(1);
        using var correct = predicted.eq(labels);

        processedSamples += batch.Size;
        correctPredictions += correct.sum().ToInt32();
        lossSum += loss.ToSingle() * batch.Size;
        batchIndex++;
    }

    private LanguageLstmEvaluation Evaluate(int batchSize, Device device)
    {
        _model.eval();

        var criterion = CrossEntropyLoss();
        var processedSamples = 0;
        var correctPredictions = 0;
        var lossSum = 0f;

        foreach (var batch in _validationDataset.Batches(batchSize))
        {
            using var d = torch.NewDisposeScope();
            using var input = batch.GetInputTensor(device);
            using var labels = batch.GetLabelTensor(device);
            using var output = _model.forward(input);
            using var loss = criterion.call(output, labels);
            using var predicted = output.argmax(1);
            using var correct = predicted.eq(labels);

            processedSamples += batch.Size;
            correctPredictions += correct.sum().ToInt32();
            lossSum += loss.ToSingle() * batch.Size;
        }

        return new LanguageLstmEvaluation(
            lossSum / Math.Max(1, processedSamples),
            (float)correctPredictions / Math.Max(1, processedSamples),
            processedSamples
        );
    }
}

internal sealed class LanguageLstmDataset
{
    private readonly LanguageTextSample[] _samples;
    private readonly Dictionary<string, int> _charToIdx;
    private readonly Dictionary<string, int> _langToIdx;
    private readonly int _maxSequenceLength;

    public LanguageLstmDataset(
        IEnumerable<LanguageTextSample> samples,
        Dictionary<string, int> charToIdx,
        Dictionary<string, int> langToIdx,
        int maxSequenceLength
    )
    {
        _samples = samples.ToArray();
        if (_samples.Length == 0)
        {
            throw new ArgumentException("Dataset samples cannot be empty.", nameof(samples));
        }

        _charToIdx = charToIdx;
        _langToIdx = langToIdx;
        _maxSequenceLength = maxSequenceLength;
    }

    public int Count => _samples.Length;

    public Batch FirstBatch(int batchSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        return CreateBatch(_samples.Take(batchSize).ToArray());
    }

    public IEnumerable<Batch> Batches(
        int batchSize,
        bool shuffle = false,
        int shuffleSeed = 1234
    )
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        var indices = Enumerable.Range(0, _samples.Length).ToArray();
        if (shuffle)
        {
            Shuffle(indices, shuffleSeed);
        }

        for (var offset = 0; offset < indices.Length; offset += batchSize)
        {
            var batchSamples = indices
                .Skip(offset)
                .Take(batchSize)
                .Select(index => _samples[index])
                .ToArray();

            yield return CreateBatch(batchSamples);
        }
    }

    private Batch CreateBatch(LanguageTextSample[] samples)
    {
        var tokens = new long[samples.Length * _maxSequenceLength];
        var labels = new long[samples.Length];

        for (var sampleIndex = 0; sampleIndex < samples.Length; sampleIndex++)
        {
            var sample = samples[sampleIndex];
            var normalized = LanguageLstmTrainingPipeline.Normalize(sample.Text);
            var tokenOffset = sampleIndex * _maxSequenceLength;

            for (var charIndex = 0; charIndex < _maxSequenceLength; charIndex++)
            {
                var token = charIndex < normalized.Length
                    ? normalized[charIndex].ToString()
                    : "PAD";

                tokens[tokenOffset + charIndex] = _charToIdx.TryGetValue(token, out var tokenIndex)
                    ? tokenIndex
                    : _charToIdx["UNK"];
            }

            labels[sampleIndex] = _langToIdx[sample.Language];
        }

        return new Batch(tokens, labels, _maxSequenceLength);
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
        private readonly long[] _tokens;
        private readonly long[] _labels;
        private readonly long[] _shape;

        public Batch(
            long[] tokens,
            long[] labels,
            int maxSequenceLength
        )
        {
            _tokens = tokens;
            _labels = labels;
            _shape = [labels.Length, maxSequenceLength];
        }

        public int Size => _labels.Length;

        public Tensor GetInputTensor(Device device)
        {
            return torch.tensor(_tokens, _shape, dtype: ScalarType.Int64, device: device);
        }

        public Tensor GetLabelTensor(Device device)
        {
            return torch.tensor(_labels, dtype: ScalarType.Int64, device: device);
        }
    }
}

internal sealed record LanguageTextSample(string Text, string Language);

internal sealed record LanguageLstmTrainingResult(
    LSTMLIDModel Model,
    LanguageLstmDataset TrainDataset,
    LanguageLstmDataset ValidationDataset,
    LanguageLstmEvaluation FinalEvaluation
);

internal readonly record struct LanguageLstmEvaluation(float Loss, float Accuracy, int Samples);
