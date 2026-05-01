namespace Onnxify.ML;

/// <summary>
/// Suggested stage families for ML pipelines.
/// </summary>
public static class PipelineStageCategories
{
    public const string DataSource = "data/source";
    public const string DataPreparation = "data/preparation";
    public const string Augmentation = "data/augmentation";
    public const string Batching = "data/batching";
    public const string DevicePlacement = "runtime/device";
    public const string Orchestration = "runtime/orchestration";
    public const string Inference = "model/inference";
    public const string Loss = "training/loss";
    public const string Optimization = "training/optimization";
    public const string Metrics = "evaluation/metrics";
    public const string PostProcessing = "postprocess";
    public const string Checkpointing = "training/checkpointing";
    public const string Export = "output/export";
    public const string Custom = "custom";
}
