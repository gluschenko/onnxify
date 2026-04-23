using System.Globalization;
using Onnx;
using Onnxify.Helpers;

namespace Onnxify;

/// <summary>
/// Base type for ONNX initializer tensors and tensor-valued attributes.
/// </summary>
/// <remarks>
/// Use the generic <see cref="OnnxTensor{T}"/> form when you need typed access to tensor values. The non-generic base is useful when inspecting models whose element type is only known at runtime.
/// </remarks>
public abstract class OnnxTensor : IOnnxGraphEdge
{
    /// <summary>
    /// Gets the tensor name used by graph nodes or attribute payloads.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the CLR element type that maps to the tensor's ONNX element type.
    /// </summary>
    public abstract Type DataType { get; }

    /// <summary>
    /// Gets tensor dimensions in ONNX order.
    /// </summary>
    public abstract long[] Shape { get; }
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

    /// <summary>
    /// Indicates whether the tensor payload was embedded in the model or referenced through ONNX external-data metadata when loaded.
    /// </summary>
    public enum TensorDataLocation
    {
        /// <summary>
        /// Tensor data is stored inside the ONNX protobuf representation.
        /// </summary>
        Default = 0,

        /// <summary>
        /// Tensor data was referenced from an external file by the source ONNX model.
        /// </summary>
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

/// <summary>
/// Provides typed access to an ONNX tensor's shape, element type, data-location metadata, and values.
/// </summary>
/// <typeparam name="T">CLR element type used by Onnxify for this tensor's ONNX data type.</typeparam>
/// <remarks>
/// Values are exposed as a flat sequence in ONNX row-major order. When serialized through Onnxify, tensors are currently written back as embedded tensor payloads even if they were loaded from external data.
/// </remarks>
public class OnnxTensor<T> : OnnxTensor
{
    private const int PREVIEW_EDGE_COUNT = 3;

    /// <inheritdoc />
    public override string Name { get; }

    /// <inheritdoc />
    public override Type DataType { get; }

    /// <summary>
    /// Gets the data-location state observed when the tensor was loaded.
    /// </summary>
    public TensorDataLocation DataLocation { get; private set; }

    /// <inheritdoc />
    public override long[] Shape { get; }

    /// <summary>
    /// Gets the flat tensor data in ONNX row-major order.
    /// </summary>
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

    /// <summary>
    /// Returns a compact diagnostic preview of the tensor values.
    /// </summary>
    /// <returns>A single-line preview intended for logs and graph inspection.</returns>
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
