using Onnx;
using Onnxify.Helpers;

namespace Onnxify;

public abstract class OnnxTensor : IOnnxGraphEdge
{
    public abstract string Name { get; }
    public abstract Type DataType { get; }
    internal abstract TensorProto ToProto();

    internal static OnnxTensor<T> FromProto<T>(TensorProto tensor, OnnxModelBaseOptions options)
    {
        var name = tensor.Name;
        var dataLocation = FromProto(tensor.DataLocation);
        var shape = tensor.Dims.ToArray();
        var value = OnnxHelper.GetValue<T>(tensor, options);

        return new OnnxTensor<T>(
            name: name,
            dataLocation: dataLocation,
            shape: shape,
            value: value,
            tensor: tensor
        );
    }

    public enum TensorDataLocation
    {
        Default = 0,
        External = 1,
    }

    internal static TensorDataLocation FromProto(TensorProto.Types.DataLocation value)
    {
        return value switch
        {
            TensorProto.Types.DataLocation.Default => TensorDataLocation.Default,
            TensorProto.Types.DataLocation.External => TensorDataLocation.External,
            _ => throw new NotImplementedException($"Not implemented for '{value}'"),
        };
    }

    internal static TensorProto.Types.DataLocation ToProto(TensorDataLocation value)
    {
        return value switch
        {
            TensorDataLocation.Default => TensorProto.Types.DataLocation.Default,
            TensorDataLocation.External => TensorProto.Types.DataLocation.External,
            _ => throw new NotImplementedException($"Not implemented for '{value}'"),
        };
    }
}

public class OnnxTensor<T> : OnnxTensor
{
    public override string Name { get; }
    public override Type DataType { get; }
    public TensorDataLocation DataLocation { get; private set; }
    public long[] Shape { get; private set; }
    public IEnumerable<T> Value { get; private set; }

    private readonly TensorProto _tensor;

    internal OnnxTensor(
        string name,
        TensorDataLocation dataLocation,
        long[] shape,
        IEnumerable<T> value,
        TensorProto? tensor
    )
    {
        Name = name;
        DataType = typeof(T);
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
        newTensor.DataType = (int)OnnxHelper.GetDataType(DataType);
        newTensor.DataLocation = ToProto(DataLocation);
        newTensor.Dims.Set(Shape);

        newTensor.DataLocation = TensorProto.Types.DataLocation.Default;
        newTensor.ExternalData.Clear();

        var data = Value.ToArray();
        newTensor.SetValue(data, Shape);

        return newTensor;
    }
}
