using System.Diagnostics;
using System.Runtime.CompilerServices;
using Onnxify.ML;
using Onnxify.ML.Stages;
using Onnxify.ML.TorchSharp;
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

        var batchesPerEpoch = checked((int)((_reader.SampleCount + batchSize - 1) / batchSize));
        var run = new AlexNetTrainingRun
        {
            Model = _model,
            Reader = _reader,
            Optimizer = optimizer,
            Criterion = criterion,
            Scheduler = scheduler,
            Epochs = epochs,
            BatchSize = batchSize,
            ShuffleSeed = shuffleSeed,
            Device = device,
            BatchesPerEpoch = batchesPerEpoch,
            TotalBatches = checked(epochs * batchesPerEpoch),
            Stopwatch = Stopwatch.StartNew()
        };

        var pipeline = Pipeline.Begin<AlexNetTrainingRun>()
            .Then(new EpochStage<AlexNetTrainingRun>(
                epochs,
                new PipelineStageOptions
                {
                    Name = "alexnet-epochs",
                    Category = PipelineStageCategories.Orchestration
                }
            ))
            .Then(new AlexNetBatchSourceStage())
            .Then(new AlexNetBatchTensorStage())
            .Then(new TorchTrainingStage<AlexNetTorchBatch, Tensor, AlexNetBatchMetrics>(
                optimizer,
                forward: (batch, _, _) => ValueTask.FromResult(batch.Run.Model.forward(batch.Inputs)),
                lossSelector: (batch, output, _, _) => ValueTask.FromResult(batch.Run.Criterion.call(output, batch.Targets)),
                resultSelector: (batch, output, loss, _, _) =>
                {
                    using var predicted = output.argmax(1);
                    using var correct = predicted.eq(batch.Targets);

                    return ValueTask.FromResult(new AlexNetBatchMetrics
                    {
                        BatchSize = batch.Source.Size,
                        CorrectPredictions = correct.sum().ToInt32()
                    });
                },
                options: new PipelineStageOptions
                {
                    Name = "alexnet-train-step",
                    Category = PipelineStageCategories.Optimization
                }))
            .Then(new AlexNetBatchProgressStage())
            .Build();

        var context = new PipelineContext()
            .Set(run)
            .Set(new AlexNetTrainingState());

        await foreach (var progress in pipeline.ExecuteAsync([run], context))
        {
            Console.WriteLine(
                $"[T+{Math.Round(progress.Elapsed.TotalSeconds)}s] " +
                $"Progress: batch {progress.GlobalBatchNumber}/{progress.TotalBatches} | " +
                $"epoch {progress.EpochNumber}/{progress.TotalEpochs} | " +
                $"epoch batch {progress.BatchNumber}/{progress.BatchesInEpoch} | " +
                $"loss {progress.BatchLoss:0.000000} | " +
                $"avg loss {progress.RunningAverageLoss:0.000000} | " +
                $"acc {progress.RunningAccuracy:0.000000} | " +
                $"lr {FormatLearningRate(progress.LearningRate)}");

            if (progress.IsLastBatchInEpoch)
            {
                Console.WriteLine(
                    $"Epoch {progress.EpochNumber}/{progress.TotalEpochs} complete | " +
                    $"samples {progress.ProcessedSamples} | " +
                    $"avg loss {progress.RunningAverageLoss:0.000000} | " +
                    $"acc {progress.RunningAccuracy:0.000000}");
            }
        }
    }

    private static string FormatLearningRate(float learningRate)
    {
        return learningRate.ToString("0.######E+0");
    }

    private sealed class AlexNetBatchSourceStage
        : PipelineStage<EpochItem<AlexNetTrainingRun>, AlexNetBatchWorkItem>
    {
        public AlexNetBatchSourceStage()
            : base(new PipelineStageOptions
            {
                Name = "alexnet-batch-source",
                Category = PipelineStageCategories.Orchestration
            })
        {
        }

        public override async IAsyncEnumerable<AlexNetBatchWorkItem> ExecuteAsync(
            IAsyncEnumerable<EpochItem<AlexNetTrainingRun>> input,
            PipelineContext context,
            [EnumeratorCancellation] CancellationToken token)
        {
            await foreach (var epoch in input.WithCancellation(token))
            {
                token.ThrowIfCancellationRequested();

                var run = epoch.Value;
                var learningRate = run.Scheduler.GetLearningRate(epoch.EpochNumber);
                run.Scheduler.Apply(run.Optimizer, epoch.EpochNumber);

                var batchIndex = 0;

                await foreach (var batch in run.Reader.BatchAsync(
                    run.BatchSize,
                    shuffle: true,
                    shuffleSeed: run.ShuffleSeed + epoch.EpochIndex,
                    cancellationToken: token))
                {
                    batchIndex++;

                    yield return new AlexNetBatchWorkItem
                    {
                        Run = run,
                        EpochIndex = epoch.EpochIndex,
                        EpochNumber = epoch.EpochNumber,
                        TotalEpochs = run.Epochs,
                        BatchIndex = batchIndex - 1,
                        BatchNumber = batchIndex,
                        BatchesInEpoch = run.BatchesPerEpoch,
                        GlobalBatchIndex = epoch.EpochIndex * run.BatchesPerEpoch + (batchIndex - 1),
                        GlobalBatchNumber = epoch.EpochIndex * run.BatchesPerEpoch + batchIndex,
                        TotalBatches = run.TotalBatches,
                        LearningRate = learningRate,
                        Source = batch
                    };
                }
            }
        }
    }

    private sealed class AlexNetBatchTensorStage : ItemPipelineStage<AlexNetBatchWorkItem, AlexNetTorchBatch>
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
            AlexNetBatchWorkItem input,
            PipelineContext context,
            CancellationToken token)
        {
            var x = input.Source.GetDataTensor(input.Run.Device);
            var y = input.Source.GetLabelTensor(input.Run.Device).view(-1);
            return ValueTask.FromResult(new AlexNetTorchBatch(input, x, y));
        }
    }

    private sealed class AlexNetBatchProgressStage
        : ItemPipelineStage<TorchTrainingStepResult<AlexNetTorchBatch, AlexNetBatchMetrics>, AlexNetBatchProgress>
    {
        public AlexNetBatchProgressStage()
            : base(new PipelineStageOptions
            {
                Name = "alexnet-batch-progress",
                Category = PipelineStageCategories.Metrics
            })
        {
        }

        protected override ValueTask<AlexNetBatchProgress> ProcessAsync(
            TorchTrainingStepResult<AlexNetTorchBatch, AlexNetBatchMetrics> input,
            PipelineContext context,
            CancellationToken token)
        {
            try
            {
                var batch = input.Batch;
                var state = context.GetRequired<AlexNetTrainingState>();
                var metrics = input.Result;

                if (state.CurrentEpochIndex != batch.WorkItem.EpochIndex)
                {
                    state.CurrentEpochIndex = batch.WorkItem.EpochIndex;
                    state.ProcessedSamples = 0;
                    state.CorrectPredictions = 0;
                    state.WeightedLoss = 0f;
                }

                state.ProcessedSamples += metrics.BatchSize;
                state.CorrectPredictions += metrics.CorrectPredictions;
                state.WeightedLoss += input.Loss * metrics.BatchSize;

                var runningAverageLoss = state.ProcessedSamples == 0
                    ? 0f
                    : state.WeightedLoss / state.ProcessedSamples;
                var runningAccuracy = state.ProcessedSamples == 0
                    ? 0f
                    : (float)state.CorrectPredictions / state.ProcessedSamples;

                return ValueTask.FromResult(new AlexNetBatchProgress
                {
                    EpochNumber = batch.WorkItem.EpochNumber,
                    TotalEpochs = batch.WorkItem.TotalEpochs,
                    BatchNumber = batch.WorkItem.BatchNumber,
                    BatchesInEpoch = batch.WorkItem.BatchesInEpoch,
                    GlobalBatchNumber = batch.WorkItem.GlobalBatchNumber,
                    TotalBatches = batch.WorkItem.TotalBatches,
                    ProcessedSamples = state.ProcessedSamples,
                    BatchLoss = input.Loss,
                    RunningAverageLoss = runningAverageLoss,
                    RunningAccuracy = runningAccuracy,
                    LearningRate = batch.WorkItem.LearningRate,
                    Elapsed = batch.WorkItem.Run.Stopwatch.Elapsed,
                    IsLastBatchInEpoch = batch.WorkItem.BatchNumber == batch.WorkItem.BatchesInEpoch
                });
            }
            finally
            {
                input.Dispose();
            }
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

    private sealed class AlexNetTrainingRun
    {
        public required AlexNet Model { get; init; }

        public required DataReader Reader { get; init; }

        public required optim.Optimizer Optimizer { get; init; }

        public required Loss<Tensor, Tensor, Tensor> Criterion { get; init; }

        public required StepLearningRateScheduler Scheduler { get; init; }

        public required int Epochs { get; init; }

        public required int BatchSize { get; init; }

        public required int ShuffleSeed { get; init; }

        public required Device Device { get; init; }

        public required int BatchesPerEpoch { get; init; }

        public required int TotalBatches { get; init; }

        public required Stopwatch Stopwatch { get; init; }
    }

    private sealed class AlexNetBatchWorkItem
    {
        public required AlexNetTrainingRun Run { get; init; }

        public required int EpochIndex { get; init; }

        public required int EpochNumber { get; init; }

        public required int TotalEpochs { get; init; }

        public required int BatchIndex { get; init; }

        public required int BatchNumber { get; init; }

        public required int BatchesInEpoch { get; init; }

        public required int GlobalBatchIndex { get; init; }

        public required int GlobalBatchNumber { get; init; }

        public required int TotalBatches { get; init; }

        public required float LearningRate { get; init; }

        public required DataReader.Batch Source { get; init; }
    }

    private sealed class AlexNetBatchMetrics
    {
        public required int BatchSize { get; init; }

        public required int CorrectPredictions { get; init; }
    }

    private sealed class AlexNetBatchProgress
    {
        public required int EpochNumber { get; init; }

        public required int TotalEpochs { get; init; }

        public required int BatchNumber { get; init; }

        public required int BatchesInEpoch { get; init; }

        public required int GlobalBatchNumber { get; init; }

        public required int TotalBatches { get; init; }

        public required int ProcessedSamples { get; init; }

        public required float BatchLoss { get; init; }

        public required float RunningAverageLoss { get; init; }

        public required float RunningAccuracy { get; init; }

        public required float LearningRate { get; init; }

        public required TimeSpan Elapsed { get; init; }

        public required bool IsLastBatchInEpoch { get; init; }
    }

    private sealed class AlexNetTrainingState
    {
        public int CurrentEpochIndex { get; set; } = -1;

        public int ProcessedSamples { get; set; }

        public int CorrectPredictions { get; set; }

        public float WeightedLoss { get; set; }
    }

    private sealed class AlexNetTorchBatch : IDisposable
    {
        public AlexNetTorchBatch(
            AlexNetBatchWorkItem workItem,
            Tensor inputs,
            Tensor targets)
        {
            WorkItem = workItem;
            Inputs = inputs;
            Targets = targets;
        }

        public AlexNetBatchWorkItem WorkItem { get; }

        public AlexNetTrainingRun Run => WorkItem.Run;

        public DataReader.Batch Source => WorkItem.Source;

        public Tensor Inputs { get; }

        public Tensor Targets { get; }

        public void Dispose()
        {
            Inputs.Dispose();
            Targets.Dispose();
        }
    }
}
