using Onnx;

namespace Onnxify;

public class OnnxValue : IOnnxGraphEdge
{
    public string Name { get; init; }

    private readonly ValueInfoProto _valueInfo;

    internal OnnxValue(ValueInfoProto valueInfo)
    {
        _valueInfo = valueInfo;

        Name = valueInfo.Name;
    }

    internal ValueInfoProto ToProto()
    {
        var newValue = _valueInfo.Clone();
        newValue.Name = Name;

        return newValue;
    }
}
