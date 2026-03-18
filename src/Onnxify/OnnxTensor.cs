using Onnx;

namespace Onnxify;

public abstract class OnnxTensor : IOnnxGraphEdge
{
    public abstract string Name { get; }
    public abstract TensorProto.Types.DataType DataType { get; }
    internal abstract TensorProto ToProto();

    internal static OnnxTensor<T> FromProto<T>(TensorProto tensor)
    {
        var name = tensor.Name;
        var dataType = (TensorProto.Types.DataType)tensor.DataType;
        var dataLocation = tensor.DataLocation;
        var shape = tensor.Dims.ToList();
        var value = OnnxHelper.GetValue<T>(tensor);

        return new OnnxTensor<T>(
            name: name,
            dataType: dataType,
            dataLocation: dataLocation,
            shape: shape,
            value: value,
            tensor: tensor
        );
    }
}

public class OnnxTensor<T> : OnnxTensor
{
    public override string Name { get; }
    public override TensorProto.Types.DataType DataType { get; }
    public TensorProto.Types.DataLocation DataLocation { get; private set; }
    public IEnumerable<long> Shape { get; private set; }
    public IEnumerable<T> Value { get; private set; }

    private readonly TensorProto _tensor;

    internal OnnxTensor(
        string name,
        TensorProto.Types.DataType dataType,
        TensorProto.Types.DataLocation dataLocation,
        IEnumerable<long> shape,
        IEnumerable<T> value,
        TensorProto? tensor
    )
    {
        Name = name;
        DataType = dataType;
        DataLocation = dataLocation;
        Shape = shape;
        Value = value;

        _tensor = tensor ?? new TensorProto 
        {
            Name = name,
        };
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
