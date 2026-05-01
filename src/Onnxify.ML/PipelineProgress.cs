namespace Onnxify.ML;

public readonly record struct PipelineProgress(
    int StageIndex,
    int StageCount,
    string StageName,
    string StageCategory,
    double StageWeight,
    double CompletedWeight,
    double TotalWeight,
    double StageProgress,
    double Value)
{
    public int StageNumber => StageIndex + 1;
    public double Percent => Value * 100.0;
}
