namespace Onnxify;

public class OnnxModelCreationOptions
{
    public int Opset { get; set; } = 13;
    public long IrVersion { get; set; } = 8;
    public string ProducerName { get; set; } = "onnxify";
}
