using Onnx;
using Onnxify.Data;

namespace Onnxify;

/// <summary>Describes quantization parameter tensors associated with a graph tensor.</summary>
public sealed class OnnxQuantizationAnnotation
{
    private readonly LazyDictionary<string, KeyValuePair<string, string>> _parameters = new(x => x.Key, StringComparer.Ordinal);

    public string TensorName { get; }

    public IReadOnlyList<KeyValuePair<string, string>> QuantParameterTensorNames => _parameters.ToList();

    public OnnxQuantizationAnnotation(string tensorName)
    {
        ArgumentNullException.ThrowIfNull(tensorName);
        TensorName = tensorName;
    }

    internal OnnxQuantizationAnnotation(TensorAnnotation annotation)
        : this(annotation.TensorName)
    {
        foreach (var parameter in annotation.QuantParameterTensorNames)
        {
            _parameters.Add(new KeyValuePair<string, string>(parameter.Key, parameter.Value));
        }
    }

    public void SetQuantParameterTensorName(string parameterName, string tensorName)
    {
        ArgumentNullException.ThrowIfNull(parameterName);
        ArgumentNullException.ThrowIfNull(tensorName);
        _parameters[parameterName] = new KeyValuePair<string, string>(parameterName, tensorName);
    }

    public bool RemoveQuantParameterTensorName(string parameterName)
    {
        return _parameters.Remove(parameterName);
    }

    internal TensorAnnotation ToProto()
    {
        var annotation = new TensorAnnotation { TensorName = TensorName };
        annotation.QuantParameterTensorNames.Add(_parameters.Select(x => new StringStringEntryProto
        {
            Key = x.Key,
            Value = x.Value,
        }));
        return annotation;
    }
}
