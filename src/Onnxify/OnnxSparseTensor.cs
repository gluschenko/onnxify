using Onnx;
using Onnxify.Helpers;

namespace Onnxify;

/// <summary>
/// Base type for ONNX sparse tensors whose element type may only be known after parsing.
/// </summary>
/// <remarks>
/// The current wrapper exposes the sparse tensor's value payload. Sparse index metadata is preserved through the underlying ONNX protobuf during round-tripping.
/// </remarks>
public abstract class OnnxSparseTensor : IOnnxGraphEdge
{
    /// <summary>
    /// Gets the sparse tensor value name used by graph references.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the CLR element type of the sparse values tensor.
    /// </summary>
    public abstract Type DataType { get; }

    /// <summary>
    /// Gets the dense value tensor that stores non-zero sparse values.
    /// </summary>
    public abstract OnnxTensor Value { get; }
    internal abstract SparseTensorProto ToProto();
}

/// <summary>
/// Provides typed access to the values carried by an ONNX sparse tensor.
/// </summary>
/// <typeparam name="T">CLR element type used by the sparse tensor values.</typeparam>
public class OnnxSparseTensor<T> : OnnxSparseTensor
{
    /// <inheritdoc />
    public override string Name => _value.Name;

    /// <inheritdoc />
    public override Type DataType => _value.DataType;

    /// <inheritdoc />
    public override OnnxTensor<T> Value => _value;

    private readonly SparseTensorProto _tensor;
    private readonly OnnxTensor<T> _value;

    internal OnnxSparseTensor(SparseTensorProto tensor, OnnxModelBaseOptions options)
    {
        _tensor = tensor;
        _value = (OnnxTensor<T>)OnnxHelper.FromProto(tensor.Values, options);
    }

    internal override SparseTensorProto ToProto()
    {
        var newTensor = _tensor.Clone();
        newTensor.Values = _value.ToProto();

        return newTensor;
    }

    /// <summary>
    /// Returns a compact diagnostic preview of the sparse value tensor.
    /// </summary>
    /// <returns>A string intended for inspection.</returns>
    public override string ToString()
    {
        return _value.ToString();
    }
}
