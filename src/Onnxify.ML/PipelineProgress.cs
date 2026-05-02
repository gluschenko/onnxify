namespace Onnxify.ML;

public readonly struct PipelineProgress
{
    public required int StageIndex { get; init; }

    public required int StageCount { get; init; }

    public required string StageName { get; init; }

    public required string StageCategory { get; init; }

    public required double StageWeight { get; init; }

    public required double CompletedWeight { get; init; }

    public required double TotalWeight { get; init; }

    public required double StageProgress { get; init; }

    public required double Value { get; init; }

    public int StageNumber => StageIndex + 1;

    public double Percent => Value * 100.0;
}
