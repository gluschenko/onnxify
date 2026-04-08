namespace Onnxify;

public class OnnxModelCreationOptions : OnnxModelBaseOptions
{
    /// <summary>
    /// Default ONNX opset version (ai.onnx domain).
    /// 18 - current stable baseline across runtimes.
    /// </summary>
    public int Opset { get; init; } = 18;

    /// <summary>
    /// ONNX IR version.
    /// 9 - modern format used by current exporters.
    /// </summary>
    public long IrVersion { get; init; } = 9;

    /// <summary>
    /// Producer name written into the model metadata.
    /// </summary>
    public string ProducerName { get; init; } = "onnxify";
}
