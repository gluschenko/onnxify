namespace Onnxify;

/// <summary>
/// Controls metadata and compatibility defaults for a newly authored ONNX model.
/// </summary>
/// <remarks>
/// These values define the initial model header. Add or adjust domain-specific opsets on <see cref="OnnxModel.OpsetImport"/> when the graph uses operators outside the default ONNX domain or requires a different schema version.
/// </remarks>
public class OnnxModelCreationOptions : OnnxModelBaseOptions
{
    /// <summary>
    /// Gets the initial opset version for the standard ONNX domain.
    /// </summary>
    /// <remarks>
    /// The value is written for the empty ONNX domain string. Choose a version supported by the runtime that will execute the model and by the operators you plan to emit.
    /// </remarks>
    public int Opset { get; init; } = 25;

    /// <summary>
    /// Gets the ONNX IR version written to the model header.
    /// </summary>
    /// <remarks>
    /// IR version controls the container format, not individual operator schemas.
    /// </remarks>
    public long IrVersion { get; init; } = 11;

    /// <summary>
    /// Gets the producer name written into model metadata for newly created models.
    /// </summary>
    public string ProducerName { get; init; } = "onnxify";
}
