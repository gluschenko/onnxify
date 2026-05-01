namespace Onnxify.ML;

public sealed class PipelineStageOptions
{
    public string? Name { get; init; }

    public string Category { get; init; } = PipelineStageCategories.Custom;

    public double ProgressWeight { get; init; } = 1.0;
}
