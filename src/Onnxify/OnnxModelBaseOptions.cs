using Onnxify.Data;

namespace Onnxify;

public class OnnxModelBaseOptions
{
    public string? DataLocation { get; init; } = null;
    public ExternalDataProvider DataReader { get; init; } = OnnxExternalDataProvider.Instance;
    public ExternalDataProvider DataWriter { get; init; } = OnnxExternalDataProvider.Instance;
}
