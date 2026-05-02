namespace Onnxify.ML;

public sealed class Pipeline<TInput, TOutput> : Pipeline
{
    private readonly PipelineStage<TInput, TOutput> _stage;
    private readonly Lazy<IReadOnlyList<PipelineStage>> _leafStages;

    public Pipeline(PipelineStage<TInput, TOutput> stage)
    {
        _stage = stage ?? throw new ArgumentNullException(nameof(stage));
        _leafStages = new Lazy<IReadOnlyList<PipelineStage>>(() => FlattenLeafStages(_stage));
    }

    public IAsyncEnumerable<TOutput> ExecuteAsync(
        IEnumerable<TInput> input,
        PipelineContext? context = null,
        ProgressChangeEvent? progressChangeEvent = null,
        CancellationToken token = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        return ExecuteAsync(PipelineAsyncEnumerable.FromEnumerable(input), context, progressChangeEvent, token);
    }

    public IAsyncEnumerable<TOutput> ExecuteAsync(
        IAsyncEnumerable<TInput> input,
        PipelineContext? context = null,
        ProgressChangeEvent? progressChangeEvent = null,
        CancellationToken token = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        var executionContext = (context ?? PipelineContext.Empty).CreateExecutionContext(progressChangeEvent);
        return _stage.ExecuteAsync(input, executionContext, token);
    }

    public PipelineProgress CalculateProgress(PipelineStage stage, int current, int total)
    {
        ArgumentNullException.ThrowIfNull(stage);

        var leafStages = _leafStages.Value;

        if (leafStages.Count == 0)
        {
            return new PipelineProgress
            {
                StageIndex = 0,
                StageCount = 0,
                StageName = string.Empty,
                StageCategory = string.Empty,
                StageWeight = 0,
                CompletedWeight = 0,
                TotalWeight = 0,
                StageProgress = 0,
                Value = 0
            };
        }

        var stageIndex = -1;
        var completedWeight = 0.0;
        var totalWeight = 0.0;

        for (var i = 0; i < leafStages.Count; i++)
        {
            var leaf = leafStages[i];
            totalWeight += leaf.ProgressWeight;

            if (ReferenceEquals(leaf, stage))
            {
                stageIndex = i;
                break;
            }

            completedWeight += leaf.ProgressWeight;
        }

        if (stageIndex < 0)
        {
            throw new ArgumentException("Stage does not belong to this pipeline.", nameof(stage));
        }

        var stageWeight = leafStages[stageIndex].ProgressWeight;
        var stageProgress = total > 0
            ? Math.Clamp((double)current / total, 0.0, 1.0)
            : 0.0;
        var value = totalWeight > 0
            ? (completedWeight + (stageWeight * stageProgress)) / totalWeight
            : 0.0;

        return new PipelineProgress
        {
            StageIndex = stageIndex,
            StageCount = leafStages.Count,
            StageName = stage.Name,
            StageCategory = stage.Category,
            StageWeight = stageWeight,
            CompletedWeight = completedWeight,
            TotalWeight = totalWeight,
            StageProgress = stageProgress,
            Value = value
        };
    }

    private static IReadOnlyList<PipelineStage> FlattenLeafStages(PipelineStage root)
    {
        var result = new List<PipelineStage>();
        Visit(root, result);
        return result;
    }

    private static void Visit(PipelineStage stage, List<PipelineStage> result)
    {
        var children = stage.GetChildren();

        if (children.Count == 0)
        {
            result.Add(stage);
            return;
        }

        foreach (var child in children)
        {
            Visit(child, result);
        }
    }
}

public abstract class Pipeline
{
    public static PipelineOrigin<TInput> Begin<TInput>()
    {
        return new PipelineOrigin<TInput>();
    }
}

public sealed class PipelineOrigin<TInput>
{
    public PipelineStage<TInput, TOutputNext> Then<TOutputNext>(PipelineStage<TInput, TOutputNext> stage)
    {
        ArgumentNullException.ThrowIfNull(stage);
        return stage;
    }
}
