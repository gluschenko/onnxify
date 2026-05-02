namespace Onnxify.ML;

/// <summary>
/// Describes user-facing metadata for a pipeline stage.
/// </summary>
public sealed class PipelineStageOptions
{
    /// <summary>
    /// Gets the display name used for progress reporting and diagnostics.
    /// </summary>
    public string? Name { get; init; }

    /// <summary>
    /// Gets the logical category used to group similar stages.
    /// </summary>
    public string Category { get; init; } = PipelineStageCategories.Custom;

    /// <summary>
    /// Gets the relative weight used when aggregating stage progress into pipeline progress.
    /// </summary>
    public double ProgressWeight { get; init; } = 1.0;
}
