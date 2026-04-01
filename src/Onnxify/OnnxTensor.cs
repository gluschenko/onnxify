using System.Globalization;
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
    private const int PREVIEW_EDGE_COUNT = 3;

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

    public override string ToString()
    {
        var data = Value.ToArray();
        var shape = $"[{string.Join(", ", Shape)}]";

        return $"{Name}: {DataType.Name}{shape} = {FormatPreview(data)}";
    }

    private static string FormatPreview(T[] data)
    {
        if (data.Length == 0)
        {
            return "[]";
        }

        if (data.Length <= PREVIEW_EDGE_COUNT * 2)
        {
            return $"[{string.Join(", ", data.Select(FormatValue))}]";
        }

        var head = data.Take(PREVIEW_EDGE_COUNT).Select(FormatValue);
        var tail = data.Skip(data.Length - PREVIEW_EDGE_COUNT).Select(FormatValue);
        return $"[{string.Join(", ", head)}, ... {string.Join(", ", tail)}]";
    }

    private static string FormatValue(T value)
    {
        if (value is null)
        {
            return "null";
        }

        return value switch
        {
            string text => text,
            IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString() ?? string.Empty,
        };
    }
}
