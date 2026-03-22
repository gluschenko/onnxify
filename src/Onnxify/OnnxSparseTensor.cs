using Onnx;

namespace Onnxify;

public abstract class OnnxSparseTensorBase : IOnnxGraphEdge
{
    public abstract string Name { get; }
    public abstract Type DataType { get; }
    public abstract OnnxTensor Value { get; }
    internal abstract SparseTensorProto ToProto();
}

public class OnnxSparseTensor<T> : OnnxSparseTensorBase
{
    public override string Name => _value.Name;
    public override Type DataType => _value.DataType;
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
}
