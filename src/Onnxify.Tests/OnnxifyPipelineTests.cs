using Onnxify.ML;
using Onnxify.ML.Stages;
using Onnxify.ML.TorchSharp;
using Onnxify.ML.TorchSharp.Stages;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Onnxify.Tests;

public sealed class OnnxifyPipelineTests
{
    [Fact]
    public async Task Pipeline_ComposesTypedStages_AndReportsWeightedProgress()
    {
        var observed = new List<int>();
        var context = new PipelineContext()
            .Set("multiplier", 3)
            .Set(observed);

        var root = new MapStage<int, int>(
                (value, ctx, _) => ValueTask.FromResult(value * ctx.Get<int>("multiplier")),
                new PipelineStageOptions
                {
                    Name = "scale",
                    Category = PipelineStageCategories.DataPreparation,
                    ProgressWeight = 1
                })
            .Then(new FilterStage<int>(
                value => value % 2 == 0,
                new PipelineStageOptions
                {
                    Name = "keep-even",
                    Category = PipelineStageCategories.DataPreparation,
                    ProgressWeight = 1
                })
            )
            .Then(new TapStage<int>(
                (value, ctx, _) =>
                {
                    ctx.GetRequired<List<int>>().Add(value);
                    return ValueTask.CompletedTask;
                },
                new PipelineStageOptions
                {
                    Name = "tap",
                    Category = PipelineStageCategories.Metrics,
                    ProgressWeight = 1
                })
            )
            .Then(new MiniBatchStage<int>(
                batchSize: 2,
                options: new PipelineStageOptions
                {
                    Name = "mini-batch",
                    Category = PipelineStageCategories.Batching,
                    ProgressWeight = 3
                })
            )
            .Then(new MapStage<MiniBatch<int>, int[]>(
                batch => batch.Items.ToArray(),
                new PipelineStageOptions
                {
                    Name = "materialize",
                    Category = PipelineStageCategories.PostProcessing,
                    ProgressWeight = 1
                })
            );

        var pipeline = new Pipeline<int, int[]>(root);
        var progressUpdates = new List<PipelineProgress>();

        root.WithProgress((stage, current, total) =>
        {
            progressUpdates.Add(pipeline.CalculateProgress(stage, current, total));
            return ValueTask.CompletedTask;
        });

        var result = await pipeline.RunToListAsync([1, 2, 3, 4, 5, 6], context);

        Assert.Equal(2, result.Count);
        Assert.Equal([6, 12, 18], observed);
        Assert.Equal([6, 12], result[0]);
        Assert.Equal([18], result[1]);
        Assert.Equal("materialize", progressUpdates[^1].StageName);
        Assert.True(progressUpdates[^1].Percent >= 99.9);
    }

    [Fact]
    public async Task CustomBatchingStage_CanImplementTokenBudgetGrouping()
    {
        var stage = new TokenBudgetBatchStage(tokenBudget: 3);
        var pipeline = new Pipeline<string, MiniBatch<string>>(stage);

        var result = await pipeline.RunToListAsync(["a", "bb", "ccc", "d"]);

        Assert.Equal(3, result.Count);
        Assert.Equal(["a", "bb"], result[0].Items);
        Assert.Equal(["ccc"], result[1].Items);
        Assert.Equal(["d"], result[2].Items);
        Assert.False(result[0].IsPartialBatch);
        Assert.False(result[1].IsPartialBatch);
        Assert.True(result[2].IsPartialBatch);
    }

    [Fact]
    public async Task ConcurrentEnumerator_PreservesSourceOrder_WhileRunningConcurrently()
    {
        var enumerator = new ConcurrentEnumerator<int, int>(
            [1, 2, 3, 4],
            async (value, token) =>
            {
                await Task.Delay((5 - value) * 20, token);
                return value * 10;
            },
            new ConcurrentEnumeratorOptions
            {
                MaxDegreeOfParallelism = 4,
                PreserveOrder = true
            });

        var result = await PipelineExecutionExtensions.ToListAsync(enumerator.ExecuteAsync());

        Assert.Equal([10, 20, 30, 40], result);
    }

    [Fact]
    public async Task ParallelMapStage_UsesConcurrentEnumerator_AndKeepsPipelineFriendlyOrder()
    {
        var stage = new ParallelMapStage<int, int>(
            async (value, _, token) =>
            {
                await Task.Delay((5 - value) * 20, token);
                return value * value;
            },
            options: new ConcurrentEnumeratorOptions
            {
                MaxDegreeOfParallelism = 4,
                PreserveOrder = true
            });

        var pipeline = new Pipeline<int, int>(stage);
        var result = await pipeline.RunToListAsync([1, 2, 3, 4]);

        Assert.Equal([1, 4, 9, 16], result);
    }

    [Fact]
    public async Task TorchInferenceStage_CanProjectPredictions()
    {
        var root = new TorchTensorBatchStage<float[]>(
                batchSize: 2,
                collate: (samples, token) => ValueTask.FromResult(CreateFeatureBatch(samples)))
            .Then(new TorchInferenceStage<TorchMiniBatch<float[]>, Tensor, long[]>(
                forward: (batch, _, _) => ValueTask.FromResult(batch.Inputs.argmax(1)),
                resultSelector: (batch, output, _, _) => ValueTask.FromResult(output.cpu().data<long>().ToArray())));

        var pipeline = new Pipeline<float[], TorchInferenceResult<TorchMiniBatch<float[]>, long[]>>(root);
        var results = await pipeline.RunToListAsync(
            [
                [1f, 3f],
                [5f, 1f],
                [2f, 9f]
            ]
        );

        try
        {
            Assert.Equal(2, results.Count);
            Assert.Equal([1L, 0L], results[0].Result);
            Assert.Equal([1L], results[1].Result);
        }
        finally
        {
            foreach (var item in results)
            {
                item.Dispose();
            }
        }
    }

    [Fact]
    public async Task TorchTrainingStage_PerformsOptimizerStep()
    {
        using var model = Linear(1, 1);
        using var optimizer = optim.SGD(model.parameters(), 0.1);

        var before = model.weight.detach().cpu().data<float>().ToArray()[0];

        var root = new TorchTensorBatchStage<(float X, float Y)>(
                batchSize: 2,
                collate: (samples, token) => ValueTask.FromResult(CreateRegressionBatch(samples)))
            .Then(new TorchTrainingStage<TorchMiniBatch<(float X, float Y)>, Tensor, float>(
                optimizer,
                forward: (batch, _, _) => ValueTask.FromResult(model.forward(batch.Inputs)),
                lossSelector: (batch, output, _, _) => ValueTask.FromResult((output - batch.Targets!).pow(2).mean()),
                resultSelector: (batch, output, loss, _, _) => ValueTask.FromResult(output.detach().cpu().mean().ToSingle())));

        var pipeline = new Pipeline<(float X, float Y), TorchTrainingStepResult<TorchMiniBatch<(float X, float Y)>, float>>(root);
        var results = await pipeline.RunToListAsync(
            [
                (1f, 2f),
                (2f, 4f),
                (3f, 6f),
                (4f, 8f)
            ]
        );

        try
        {
            var after = model.weight.detach().cpu().data<float>().ToArray()[0];

            Assert.Equal(2, results.Count);
            Assert.NotEqual(before, after);
            Assert.All(results, result => Assert.True(result.Loss >= 0));
        }
        finally
        {
            foreach (var item in results)
            {
                item.Dispose();
            }
        }
    }

    private static TorchBatchTensors CreateFeatureBatch(IReadOnlyList<float[]> samples)
    {
        var features = samples.SelectMany(static sample => sample).ToArray();
        var flat = tensor(features, dtype: ScalarType.Float32);
        return new TorchBatchTensors(flat.reshape(samples.Count, samples[0].Length));
    }

    private static TorchBatchTensors CreateRegressionBatch(IReadOnlyList<(float X, float Y)> samples)
    {
        var inputs = samples.Select(static sample => sample.X).ToArray();
        var targets = samples.Select(static sample => sample.Y).ToArray();

        var x = tensor(inputs, dtype: ScalarType.Float32);
        var y = tensor(targets, dtype: ScalarType.Float32);

        return new TorchBatchTensors(
            x.reshape(samples.Count, 1),
            y.reshape(samples.Count, 1));
    }

    private sealed class TokenBudgetBatchStage : BatchingStage<string, MiniBatch<string>>
    {
        private readonly int _tokenBudget;

        public TokenBudgetBatchStage(int tokenBudget)
            : base(
                batchSize: int.MaxValue,
                includeIncompleteBatch: true,
                options: new PipelineStageOptions
                {
                    Name = "token-budget",
                    Category = PipelineStageCategories.Batching
                })
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokenBudget);
            _tokenBudget = tokenBudget;
        }

        protected override bool ShouldFlushBatch(IReadOnlyList<string> buffer, int batchIndex)
        {
            return buffer.Sum(static value => value.Length) >= _tokenBudget;
        }

        protected override ValueTask<MiniBatch<string>> CreateBatchAsync(
            IReadOnlyList<string> batchItems,
            int batchIndex,
            bool isPartialBatch,
            PipelineContext context,
            CancellationToken token)
        {
            var batch = new MiniBatch<string>
            {
                Items = batchItems,
                BatchIndex = batchIndex,
                IsPartialBatch = isPartialBatch
            };

            return ValueTask.FromResult(batch);
        }
    }
}
