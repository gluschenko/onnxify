namespace Onnxify.ML;

/// <summary>
/// Suggested stage families for ML pipelines.
/// </summary>
public static class PipelineStageCategories
{
    /// <summary>Stage category for data loading sources.</summary>
    public const string DataSource = "data/source";

    /// <summary>Stage category for preprocessing and feature shaping.</summary>
    public const string DataPreparation = "data/preparation";

    /// <summary>Stage category for augmentation and sampling transforms.</summary>
    public const string Augmentation = "data/augmentation";

    /// <summary>Stage category for mini-batching and collation.</summary>
    public const string Batching = "data/batching";

    /// <summary>Stage category for device transfers and runtime placement.</summary>
    public const string DevicePlacement = "runtime/device";

    /// <summary>Stage category for orchestration, control flow, and repetition.</summary>
    public const string Orchestration = "runtime/orchestration";

    /// <summary>Stage category for forward-only inference work.</summary>
    public const string Inference = "model/inference";

    /// <summary>Stage category for loss computation.</summary>
    public const string Loss = "training/loss";

    /// <summary>Stage category for optimizer and gradient update work.</summary>
    public const string Optimization = "training/optimization";

    /// <summary>Stage category for evaluation and metric reporting.</summary>
    public const string Metrics = "evaluation/metrics";

    /// <summary>Stage category for output decoding and postprocessing.</summary>
    public const string PostProcessing = "postprocess";

    /// <summary>Stage category for checkpoint persistence.</summary>
    public const string Checkpointing = "training/checkpointing";

    /// <summary>Stage category for model export and serialization.</summary>
    public const string Export = "output/export";

    /// <summary>Fallback category for custom stages.</summary>
    public const string Custom = "custom";
}
