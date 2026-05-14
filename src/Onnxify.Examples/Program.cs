using System.Diagnostics;
using System.Globalization;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Examples.Models;
using Onnxify.Safetensors;
using Onnxify.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace Onnxify.Examples;

internal class Program
{
    private static readonly Sample[] _samples =
    [
        new AlexNetSample(),
        new MobileNetV1LikeSample(),
        new TinyYoloLikeSample(),
        new LSTMSample(),
        new MiniGpt2LikeSample(),
        new TorchSharpExportShowcaseSample(),
    ];

    static async Task Main(string[] args)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        Console.Title = nameof(Onnxify);
        Console.InputEncoding = Encoding.Unicode;
        Console.OutputEncoding = Encoding.Unicode;

        var selectedItem = SelectExample(_samples, args);
        await selectedItem.RunAsync();

        if (args.Length == 0)
        {
            Console.WriteLine("Press any key to pay respect...");
            Console.ReadKey();
        }
    }

    static Sample SelectExample(
        Sample[] items,
        string[] args
    )
    {
        var selector = args.FirstOrDefault()?.Trim();
        if (!string.IsNullOrWhiteSpace(selector))
        {
            var item = FindExample(items, selector);
            if (item is not null)
            {
                return item;
            }
        }

        while (true)
        {
            Console.WriteLine("Select example:");
            for (var i = 0; i < items.Length; i++)
            {
                Console.WriteLine($"{i + 1}. {items[i].Description} ({items[i].Name})");
            }

            Console.Write("Enter number or key: ");
            var input = Console.ReadLine()?.Trim();
            var item = FindExample(items, input);
            if (item is not null)
            {
                Console.WriteLine();
                return item;
            }

            Console.WriteLine("Unknown selection. Try again.");
            Console.WriteLine();
        }
    }

    static Sample? FindExample(
        IReadOnlyList<Sample> items,
        string? selector
    )
    {
        if (string.IsNullOrWhiteSpace(selector))
        {
            return null;
        }

        if (int.TryParse(selector, out var index) && index >= 1 && index <= items.Count)
        {
            return items[index - 1];
        }

        return items.FirstOrDefault(item =>
            string.Equals(item.Name, selector, StringComparison.OrdinalIgnoreCase)
        );
    }
}

internal class TorchSharpExportShowcaseSample : Sample
{
    public override string Name => "showcase";
    public override string Description => "TorchSharp export showcase";

    public override async Task RunAsync()
    {
        torch.random.manual_seed(1234);

        var outputDirectory = Utils.EnsureAssetsDirectory();
        var outputPath = Path.Combine(outputDirectory, "torchsharp-export-showcase.onnx");

        var model = new TorchSharpExportShowcase();
        model.eval();

        using var input = torch.randn([2, 3, 16, 16], device: torch.CPU);
        using var torchOutput = model.forward(input).cpu();

        var onnxModel = model.Export();
        onnxModel.Save(outputPath, true);

        var weightOutputPath = Path.Combine(outputDirectory, "torchsharp-export-showcase.safetensors");
        model.SaveStateAsSafetensors(weightOutputPath);

        using var session = new InferenceSession(outputPath);
        var onnxOutput = Utils.RunOnnx(session, input);

        Console.WriteLine("TorchSharp export showcase");
        Console.WriteLine($"Saved ONNX model: {outputPath}");
        Console.WriteLine($"Output shape: [{string.Join(", ", torchOutput.shape)}]");
        Console.WriteLine($"Max abs diff Torch vs ONNX: {Utils.ComputeMaxAbsDiff(torchOutput, onnxOutput):G9}");
        Console.WriteLine();
        Console.WriteLine("Operators exercised by this example:");
        Console.WriteLine(
            "ReflectionPad2d, Conv2d, GELU, AvgPool2d, Mish, MaxPool2d, PixelUnshuffle, PReLU, SiLU, PixelShuffle, LayerNorm, AdaptiveAvgPool2d, Flatten, Linear, SELU, Softplus, LogSoftmax"
        );

        await Task.CompletedTask;
    }
}

internal abstract class Sample
{
    public abstract string Name { get; }
    public abstract string Description { get; }
    public abstract Task RunAsync();
}

public static class Utils
{
    public static string EnsureAssetsDirectory()
    {
        var outputDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets");
        Directory.CreateDirectory(outputDirectory);
        return outputDirectory;
    }

    public static float[] RunOnnx(InferenceSession session, Tensor input)
    {
        var inputData = input.cpu().data<float>().ToArray();
        var inputTensor = new DenseTensor<float>(
            inputData,
            [(int)input.shape[0], (int)input.shape[1], (int)input.shape[2], (int)input.shape[3]]
        );

        var inputValue = NamedOnnxValue.CreateFromTensor("input", inputTensor);
        using var results = session.Run([inputValue]);
        return results.Single().AsTensor<float>().ToArray();
    }

    public static float[] RunOnnxInt64(InferenceSession session, Tensor input)
    {
        var inputData = input.cpu().data<long>().ToArray();
        var inputTensor = new DenseTensor<long>(
            inputData,
            [(int)input.shape[0], (int)input.shape[1]]
        );

        var inputValue = NamedOnnxValue.CreateFromTensor("input", inputTensor);
        using var results = session.Run([inputValue]);
        return results.Single().AsTensor<float>().ToArray();
    }

    public static float ComputeMaxAbsDiff(Tensor torchOutput, IReadOnlyList<float> onnxOutput)
    {
        var torchData = torchOutput.data<float>().ToArray();
        if (torchData.Length != onnxOutput.Count)
        {
            throw new InvalidOperationException(
                $"Torch output length {torchData.Length} does not match ONNX output length {onnxOutput.Count}."
            );
        }

        var maxAbsDiff = 0f;
        for (var i = 0; i < torchData.Length; i++)
        {
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(torchData[i] - onnxOutput[i]));
        }

        return maxAbsDiff;
    }
}

internal class MiniGpt2LikeSample : Sample
{
    public override string Name => "gpt2";
    public override string Description => "Mini GPT-2-like decoder export";

    public override async Task RunAsync()
    {
        torch.random.manual_seed(1234);

        var outputDirectory = Utils.EnsureAssetsDirectory();
        var outputPath = Path.Combine(outputDirectory, "mini-gpt2-like.onnx");

        var model = new MiniGpt2LikeModel();
        model.eval();

        using var tokenIds = torch
            .tensor(
                new long[]
                {
                    1, 5, 7, 9, 11, 13, 15, 17,
                    2, 4, 6, 8, 10, 12, 14, 16,
                },
                dtype: ScalarType.Int64,
                device: torch.CPU
            )
            .reshape(2, model.MaxSequenceLength);

        using var torchOutput = model.forward(tokenIds).cpu();

        var onnxModel = model.Export();
        onnxModel.Save(outputPath, true);

        var weightOutputPath = Path.Combine(outputDirectory, "mini-gpt2-like.safetensors");
        model.SaveStateAsSafetensors(weightOutputPath);

        using var session = new InferenceSession(outputPath);
        var onnxOutput = Utils.RunOnnxInt64(session, tokenIds);

        Console.WriteLine("Mini GPT-2-like decoder export");
        Console.WriteLine($"Saved ONNX model: {outputPath}");
        Console.WriteLine($"Input shape: [{string.Join(", ", tokenIds.shape)}]");
        Console.WriteLine($"Output shape: [{string.Join(", ", torchOutput.shape)}]");
        Console.WriteLine($"Max abs diff Torch vs ONNX: {Utils.ComputeMaxAbsDiff(torchOutput, onnxOutput):G9}");
        Console.WriteLine();
        Console.WriteLine("Architecture:");
        Console.WriteLine("Token embedding, position embedding, 2 GPT-2-style transformer blocks, fused QKV causal self-attention, GELU MLP, tied output projection");

        await Task.CompletedTask;
    }
}

internal class LSTMSample : Sample
{
    private const string DATASET_PATH = @"D:\Backups\ML\language-detector\dataset\export_lstm_46\languages";

    public override string Name => "lstm";
    public override string Description => "Language LSTM training and export";

    public override async Task RunAsync()
    {
        torch.random.manual_seed(1234);

        var outputDirectory = Utils.EnsureAssetsDirectory();
        var device = cuda.is_available() ? CUDA : CPU;
        var outputPath = Path.Combine(outputDirectory, "lang-lstm.onnx");
        var weightOutputPath = Path.Combine(outputDirectory, "lang-lstm.safetensors");
        var torchModelPath = Path.Combine(outputDirectory, "lang-lstm.pt");

        var pipeline = new LanguageLstmTrainingPipeline(
            DATASET_PATH,
            maxSequenceLength: 32,
            trainSamplesPerLanguage: 1000000,
            validationSamplesPerLanguage: 64
        );

        var embeddingDim = 128;
        var hiddenDim = 128;
        var layers = 2;

        var training = await pipeline.TrainAsync(
            embeddingDim,
            hiddenDim,
            layers,
            epochs: 2,
            batchSize: 2048,
            learningRate: 1e-1f,
            device: device
        );

        var model = training.Model;
        model.SaveModel(torchModelPath);
        model.SaveStateAsSafetensors(weightOutputPath);

        var onnxModel = model.Export();
        onnxModel.Save(outputPath, true);

        using var smokeBatch = training.ValidationDataset.FirstBatch(batchSize: 3).GetInputTensor(device);
        using var torchOutput = model.forward(smokeBatch).cpu();
        using var session = new InferenceSession(outputPath);
        var onnxOutput = Utils.RunOnnxInt64(session, smokeBatch.cpu());
        var languageNames = pipeline.LangToIdx.OrderBy(x => x.Value).Select(x => x.Key).ToArray();
        var languagePreview = string.Join(", ", languageNames.Take(24));
        if (languageNames.Length > 24)
        {
            languagePreview += $", ... +{languageNames.Length - 24}";
        }

        Console.WriteLine("Language LSTM training and export");
        Console.WriteLine($"Dataset: {pipeline.DatasetPath}");
        Console.WriteLine($"Device: {device.type}");
        Console.WriteLine($"Corpus samples: {pipeline.TotalCorpusSamples}");
        Console.WriteLine($"Training samples: {training.TrainDataset.Count}");
        Console.WriteLine($"Validation samples: {training.ValidationDataset.Count}");
        Console.WriteLine($"Vocabulary size: {pipeline.CharToIdx.Count}");
        Console.WriteLine($"Languages: {languageNames.Length} ({languagePreview})");
        Console.WriteLine($"Final validation loss: {training.FinalEvaluation.Loss:0.000000}");
        Console.WriteLine($"Final validation accuracy: {training.FinalEvaluation.Accuracy:0.000000}");
        Console.WriteLine($"Saved Torch model: {torchModelPath}");
        Console.WriteLine($"Saved weights: {weightOutputPath}");
        Console.WriteLine($"Saved ONNX model: {outputPath}");
        Console.WriteLine($"Max abs diff Torch vs ONNX: {Utils.ComputeMaxAbsDiff(torchOutput, onnxOutput):G9}");

        await Task.CompletedTask;
    }
}

internal class TinyYoloLikeSample : Sample
{
    public override string Name => "yolo";
    public override string Description => "Tiny YOLO-like detector export";

    public override async Task RunAsync()
    {
        torch.random.manual_seed(1234);

        var outputDirectory = Utils.EnsureAssetsDirectory();
        var outputPath = Path.Combine(outputDirectory, "tiny-yolo-like.onnx");

        var model = new TinyYoloLikeDetector();
        model.eval();

        using var input = torch.randn([2, 3, 64, 64], device: torch.CPU);
        using var torchOutput = model.forward(input).cpu();

        var onnxModel = model.Export();
        onnxModel.Save(outputPath, true);

        var weightOutputPath = Path.Combine(outputDirectory, "tiny-yolo-like.safetensors");
        model.SaveStateAsSafetensors(weightOutputPath);

        using var session = new InferenceSession(outputPath);
        var onnxOutput = Utils.RunOnnx(session, input);

        Console.WriteLine("Tiny YOLO-like detector export");
        Console.WriteLine($"Saved ONNX model: {outputPath}");
        Console.WriteLine($"Input shape: [{string.Join(", ", input.shape)}]");
        Console.WriteLine($"Output shape: [{string.Join(", ", torchOutput.shape)}]");
        Console.WriteLine($"Max abs diff Torch vs ONNX: {Utils.ComputeMaxAbsDiff(torchOutput, onnxOutput):G9}");
        Console.WriteLine();
        Console.WriteLine("Detection layout:");
        Console.WriteLine($"[batch, {model.PredictionCount}, {model.AttributeCount}] where attrs = (x, y, w, h, objectness, classes...)");

        await Task.CompletedTask;
    }
}

internal class MobileNetV1LikeSample : Sample
{
    public override string Name => "mobilenet";
    public override string Description => "MobileNetV1-like classifier export";

    public override async Task RunAsync()
    {
        torch.random.manual_seed(1234);

        var outputDirectory = Utils.EnsureAssetsDirectory();
        var outputPath = Path.Combine(outputDirectory, "mobilenet-v1-like.onnx");

        var model = new MobileNetV1LikeClassifier();
        model.eval();

        using var input = torch.randn([2, 3, 96, 96], device: torch.CPU);
        using var torchOutput = model.forward(input).cpu();

        var onnxModel = model.Export();
        onnxModel.Save(outputPath, true);

        var weightOutputPath = Path.Combine(outputDirectory, "mobilenet-v1-like.safetensors");
        model.SaveStateAsSafetensors(weightOutputPath);

        using var session = new InferenceSession(outputPath);
        var onnxOutput = Utils.RunOnnx(session, input);

        Console.WriteLine("MobileNetV1-like classifier export");
        Console.WriteLine($"Saved ONNX model: {outputPath}");
        Console.WriteLine($"Input shape: [{string.Join(", ", input.shape)}]");
        Console.WriteLine($"Output shape: [{string.Join(", ", torchOutput.shape)}]");
        Console.WriteLine($"Max abs diff Torch vs ONNX: {Utils.ComputeMaxAbsDiff(torchOutput, onnxOutput):G9}");
        Console.WriteLine();
        Console.WriteLine("Architecture:");
        Console.WriteLine("Conv-BN-ReLU6 stem, depthwise-separable MobileNet blocks, adaptive average pooling, linear classifier");

        await Task.CompletedTask;
    }
}

internal class AlexNetSample : Sample
{
    public override string Name => "alexnet";
    public override string Description => "AlexNet training and export";

    public override async Task RunAsync()
    {
        var trainDatasetDirectory = @"D:\Backups\ML\.image-classification\nsfw_dataset_v1\train";
        var testDatasetDirectory = @"D:\Backups\ML\.image-classification\nsfw_dataset_v1\test";

        var outputDirectory = Utils.EnsureAssetsDirectory();
        var device = cuda.is_available() ? CUDA : CPU;

        var trainDataset = new DataReader(
            trainDatasetDirectory,
            width: 227,
            height: 227,
            channels: 3,
            count: 200000
        );

        var testDataset = new DataReader(
            testDatasetDirectory,
            width: 227,
            height: 227,
            channels: 3,
            count: 50000
        );

        var weightOutputPath = Path.Combine(outputDirectory, "alexnet.safetensors");
        var outputPath = Path.Combine(outputDirectory, "alexnet.onnx");

        var model = new AlexNet("alexnet", trainDataset.LabelNames.Count, device);

        if (File.Exists(weightOutputPath))
        {
            model.LoadStateFromSafetensors(weightOutputPath);
        }

        var stopwatch = Stopwatch.StartNew();

        await foreach (var x in trainDataset.Convert())
        {
            Console.Write(
                $"\r[T+{Math.Round(stopwatch.Elapsed.TotalSeconds)}s] " +
                $"[Train dataset] " +
                $"{x.Current} / {x.All} | classes: {trainDataset.ClassCount} | failed: {x.Failed}"
            );
        }

        Console.WriteLine();

        await foreach (var x in testDataset.Convert())
        {
            Console.Write(
                $"\r[T+{Math.Round(stopwatch.Elapsed.TotalSeconds)}s] " +
                $"[Test dataset] " +
                $"{x.Current} / {x.All} | classes: {testDataset.ClassCount} | failed: {x.Failed}"
            );
        }

        Console.WriteLine();

        var trainer = new AlexNetTrainer(model, trainDataset);
        await trainer.TrainAsync(
            epochs: 75,
            batchSize: 256 + 128,
            learningRate: 1e-4f,
            schedulerStepSize: 10,
            schedulerGamma: 0.5f,
            minLearningRate: 1e-6f,
            device: device
        );

        model.SaveStateAsSafetensors(weightOutputPath);

        var onnxModel = model.Export();
        onnxModel.Save(outputPath, true);

        var torchEvaluation = await ModelEvaluator.EvaluateTorch(model, testDataset, batchSize: 256, device);
        var onnxEvaluation = await ModelEvaluator.EvaluateOnnx(outputPath, testDataset, batchSize: 256);

        ModelEvaluator.PrintConfusionMatrix(
            "Torch Confusion Matrix",
            torchEvaluation,
            trainDataset.LabelNames
        );

        ModelEvaluator.PrintConfusionMatrix(
            "ONNX Confusion Matrix",
            onnxEvaluation,
            trainDataset.LabelNames
        );
    }
}
