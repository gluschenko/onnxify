namespace Onnxify;

public class OnnxModelCreationOptions : OnnxModelBaseOptions
{
    public int Opset { get; init; } = 13;
    public long IrVersion { get; init; } = 8;
    public string ProducerName { get; init; } = "onnxify";
}
