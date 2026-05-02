namespace Onnxify.ML;

/// <summary>
/// Represents aggregate pipeline progress derived from a single stage progress event.
/// </summary>
public readonly struct PipelineProgress
{
    /// <summary>
    /// Gets the zero-based index of the active leaf stage.
    /// </summary>
    public required int StageIndex { get; init; }

    /// <summary>
    /// Gets the total number of leaf stages in the pipeline.
    /// </summary>
    public required int StageCount { get; init; }

    /// <summary>
    /// Gets the active stage name.
    /// </summary>
    public required string StageName { get; init; }

    /// <summary>
    /// Gets the active stage category.
    /// </summary>
    public required string StageCategory { get; init; }

    /// <summary>
    /// Gets the weight assigned to the active stage.
    /// </summary>
    public required double StageWeight { get; init; }

    /// <summary>
    /// Gets the cumulative weight of all previously completed leaf stages.
    /// </summary>
    public required double CompletedWeight { get; init; }

    /// <summary>
    /// Gets the total weight across all leaf stages.
    /// </summary>
    public required double TotalWeight { get; init; }

    /// <summary>
    /// Gets the normalized progress of the active stage in the range [0, 1].
    /// </summary>
    public required double StageProgress { get; init; }

    /// <summary>
    /// Gets the normalized aggregate pipeline progress in the range [0, 1].
    /// </summary>
    public required double Value { get; init; }

    /// <summary>
    /// Gets the one-based stage number.
    /// </summary>
    public int StageNumber => StageIndex + 1;

    /// <summary>
    /// Gets the aggregate progress as a percentage in the range [0, 100].
    /// </summary>
    public double Percent => Value * 100.0;
}
