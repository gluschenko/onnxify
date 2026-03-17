using Onnx;

namespace Onnxify;

public abstract class OnnxTensor : IOnnxGraphEdge
{
    public abstract string Name { get; }
    public abstract TensorProto.Types.DataType DataType { get; }
    internal abstract TensorProto ToProto();
}

public class OnnxTensor<T> : OnnxTensor
{
    public override string Name => _tensor.Name;
    public override TensorProto.Types.DataType DataType => (TensorProto.Types.DataType)_tensor.DataType;
    public TensorProto.Types.DataLocation DataLocation { set; get; }
    public IEnumerable<long> Shape { get; set; }
    public IEnumerable<T> Value { get; set; }

    private readonly TensorProto _tensor;

    internal OnnxTensor(TensorProto tensor)
    {
        _tensor = tensor;

        DataLocation = tensor.DataLocation;
        Shape = tensor.Dims.ToList();
        Value = OnnxHelper.GetValue<T>(tensor);
    }

    internal override TensorProto ToProto()
    {
        var newTensor = _tensor.Clone();
        newTensor.Name = Name;
        newTensor.DataType = (int)DataType;
        newTensor.DataLocation = DataLocation;

        newTensor.Dims.Clear();
        foreach (var size in Shape)
        {
            newTensor.Dims.Add(size);
        }

        var data = Value.ToArray() ?? [];
        newTensor.SetValue(data, _tensor.Dims.ToArray());

        return newTensor;
    }
}
