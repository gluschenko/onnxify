using System.Diagnostics;
using Onnxify.ML;
using Onnxify.ML.Stages;
using Onnxify.ML.TorchSharp.Stages;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Onnxify.Examples.Models;

internal sealed class AlexNetTrainer
{
    private readonly AlexNet _model;
    private readonly DataReader _reader;

    public AlexNetTrainer(AlexNet model, DataReader reader)
    {
        _model = model;
        _reader = reader;
    }

    public async Task TrainAsync(
        int epochs = 5,
        int batchSize = 32,
        float learningRate = 1e-3f,
        int schedulerStepSize = 5,
        float schedulerGamma = 0.5f,
        float minLearningRate = 1e-5f,
        int shuffleSeed = 42,
        Device? device = null
    )
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        device ??= torch.cuda.is_available() ? CUDA : CPU;

        _model.to(device);
        _model.train();

        using var optimizer = optim.Adam(_model.parameters(), learningRate);
        using var criterion = CrossEntropyLoss();

        var scheduler = new StepLearningRateScheduler(
            learningRate,
            schedulerStepSize,
            schedulerGamma,
            minLearningRate
        );

        var run = new AlexNetTrainingRun(
            _model,
            _reader,
            optimizer,
            criterion,
            scheduler,
            epochs,
            batchSize,
            shuffleSeed,
            device,
            Stopwatch.StartNew()
        );

        var pipeline = Pipeline.Begin<AlexNetTrainingRun>()
            .Then(new EpochStage<AlexNetTrainingRun>(
                epochs,
                new PipelineStageOptions
                {
                    Name = "alexnet-epochs",
                    Category = PipelineStageCategories.Orchestration
                }
            ))
            .Then(new AlexNetEpochTrainingStage())
            .Then(new TapStage<AlexNetEpochSummary>(
                summary =>
                {

                },
                new PipelineStageOptions
                {
                    Name = "epoch-summary",
                    Category = PipelineStageCategories.Metrics
                }
            ))
            .WithProgress(async (stage, current, total) =>
            {
                Console.WriteLine(
                    $"Progress: epoch {current}/{total} | " +
                    $"stage {stage.Name} ({stage.Category})"
                );
            })
            .Build();

        var context = new PipelineContext().Set(run);

        await foreach (var summary in pipeline.ExecuteAsync([run], context))
        {
            Console.WriteLine();
            Console.WriteLine(
                $"Epoch {summary.EpochNumber}/{summary.TotalEpochs} summary | " +
                $"samples {summary.ProcessedSamples} | " +
                $"avg loss {summary.AverageLoss:0.000000} | " +
                $"acc {summary.Accuracy:0.000000} | " +
                $"lr {FormatLearningRate(summary.LearningRate)}"
            );
        }
    }

    private static string FormatLearningRate(float learningRate)
    {
        return learningRate.ToString("0.######E+0");
    }

    private sealed class AlexNetEpochTrainingStage
        : ItemPipelineStage<EpochItem<AlexNetTrainingRun>, AlexNetEpochSummary>
    {
        private readonly AlexNetBatchTensorStage _tensorStage = new();

        public AlexNetEpochTrainingStage()
            : base(new PipelineStageOptions
            {
                Name = "alexnet-train-epoch",
                Category = PipelineStageCategories.Optimization
            })
        {
        }

        protected override async ValueTask<AlexNetEpochSummary> ProcessAsync(
            EpochItem<AlexNetTrainingRun> input,
            PipelineContext context,
            CancellationToken token)
        {
            var run = input.Value;
            var learningRate = run.Scheduler.GetLearningRate(input.EpochNumber);
            run.Scheduler.Apply(run.Optimizer, input.EpochNumber);

            Console.WriteLine(
                $"[T+{Math.Round(run.Stopwatch.Elapsed.TotalSeconds)}s] " +
                $"Epoch {input.EpochNumber}/{run.Epochs} | " +
                $"lr {FormatLearningRate(learningRate)}");

            var trainingStage = new TorchTrainingStage<AlexNetTorchBatch, Tensor, AlexNetBatchMetrics>(
                run.Optimizer,
                forward: (batch, _, _) => ValueTask.FromResult(run.Model.forward(batch.Inputs)),
                lossSelector: (batch, output, _, _) => ValueTask.FromResult(run.Criterion.call(output, batch.Targets)),
                resultSelector: (batch, output, loss, _, _) =>
                {
                    using var predicted = output.argmax(1);
                    using var correct = predicted.eq(batch.Targets);

                    return ValueTask.FromResult(new AlexNetBatchMetrics(
                        BatchSize: batch.Source.Size,
                        CorrectPredictions: correct.sum().ToInt32()));
                },
                options: new PipelineStageOptions
                {
                    Name = "alexnet-train-step",
                    Category = PipelineStageCategories.Optimization
                });

            var batchIndex = 0;
            var processedSamples = 0;
            var correctPredictions = 0;
            var weightedLoss = 0f;

            await foreach (var batch in run.Reader.BatchAsync(
                run.BatchSize,
                shuffle: true,
                shuffleSeed: run.ShuffleSeed + input.EpochIndex,
                cancellationToken: token))
            {
                using var torchBatch = await _tensorStage.ExecuteSingleAsync(
                    batch,
                    context,
                    token);

                using var step = await trainingStage.ExecuteSingleAsync(
                    torchBatch,
                    context,
                    token);

                batchIndex++;
                processedSamples += step.Result.BatchSize;
                correctPredictions += step.Result.CorrectPredictions;
                weightedLoss += step.Loss * step.Result.BatchSize;

                var accuracy = processedSamples == 0
                    ? 0f
                    : (float)correctPredictions / processedSamples;

                Console.Write(
                    $"\rTrain: epoch {input.EpochNumber}/{run.Epochs} | " +
                    $"batch {batchIndex} | " +
                    $"samples {processedSamples} | " +
                    $"loss {step.Loss:0.000000} | " +
                    $"acc {accuracy:0.000000} | " +
                    $"lr {FormatLearningRate(learningRate)}");
            }

            return new AlexNetEpochSummary(
                EpochNumber: input.EpochNumber,
                TotalEpochs: run.Epochs,
                ProcessedSamples: processedSamples,
                CorrectPredictions: correctPredictions,
                AverageLoss: processedSamples == 0 ? 0f : weightedLoss / processedSamples,
                LearningRate: learningRate);
        }
    }

    private sealed class AlexNetBatchTensorStage : ItemPipelineStage<DataReader.Batch, AlexNetTorchBatch>
    {
        public AlexNetBatchTensorStage()
            : base(new PipelineStageOptions
            {
                Name = "alexnet-materialize-tensors",
                Category = PipelineStageCategories.DevicePlacement
            })
        {
        }

        protected override ValueTask<AlexNetTorchBatch> ProcessAsync(
            DataReader.Batch input,
            PipelineContext context,
            CancellationToken token)
        {
            var device = context.GetRequired<AlexNetTrainingRun>().Device;
            var x = input.GetDataTensor(device);
            var y = input.GetLabelTensor(device).view(-1);
            return ValueTask.FromResult(new AlexNetTorchBatch(input, x, y));
        }
    }

    private sealed class StepLearningRateScheduler
    {
        private readonly float _initialLearningRate;
        private readonly int _stepSize;
        private readonly float _gamma;
        private readonly float _minLearningRate;

        public StepLearningRateScheduler(
            float initialLearningRate,
            int stepSize,
            float gamma,
            float minLearningRate)
        {
            _initialLearningRate = initialLearningRate;
            _stepSize = Math.Max(1, stepSize);
            _gamma = gamma;
            _minLearningRate = MathF.Max(0f, minLearningRate);
        }

        public float GetLearningRate(int epoch)
        {
            var decaySteps = Math.Max(0, (epoch - 1) / _stepSize);
            var learningRate = _initialLearningRate * MathF.Pow(_gamma, decaySteps);
            return MathF.Max(_minLearningRate, learningRate);
        }

        public void Apply(optim.Optimizer optimizer, int epoch)
        {
            var learningRate = GetLearningRate(epoch);
            foreach (var group in optimizer.ParamGroups)
            {
                group.LearningRate = learningRate;
            }
        }
    }

    private sealed record AlexNetTrainingRun(
        AlexNet Model,
        DataReader Reader,
        optim.Optimizer Optimizer,
        Loss<Tensor, Tensor, Tensor> Criterion,
        StepLearningRateScheduler Scheduler,
        int Epochs,
        int BatchSize,
        int ShuffleSeed,
        Device Device,
        Stopwatch Stopwatch);

    private sealed record AlexNetBatchMetrics(
        int BatchSize,
        int CorrectPredictions);

    private sealed record AlexNetEpochSummary(
        int EpochNumber,
        int TotalEpochs,
        int ProcessedSamples,
        int CorrectPredictions,
        float AverageLoss,
        float LearningRate)
    {
        public float Accuracy => ProcessedSamples == 0
            ? 0f
            : (float)CorrectPredictions / ProcessedSamples;
    }

    private sealed class AlexNetTorchBatch : IDisposable
    {
        public AlexNetTorchBatch(
            DataReader.Batch source,
            Tensor inputs,
            Tensor targets)
        {
            Source = source;
            Inputs = inputs;
            Targets = targets;
        }

        public DataReader.Batch Source { get; }

        public Tensor Inputs { get; }

        public Tensor Targets { get; }

        public void Dispose()
        {
            Inputs.Dispose();
            Targets.Dispose();
        }
    }
}
