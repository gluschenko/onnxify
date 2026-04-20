namespace Onnxify.Safetensors;

public sealed class TensorView : IEquatable<TensorView>
{
    private readonly ulong[] _shape;

    public DataType DataType { get; }
    public ReadOnlyMemory<byte> Data { get; }
    public IReadOnlyList<ulong> Shape => _shape;

    public TensorView(DataType dtype, IEnumerable<ulong> shape, ReadOnlyMemory<byte> data)
    {
        ArgumentNullException.ThrowIfNull(shape);

        var shapeArray = shape.ToArray();
        var expectedSize = SafetensorMath.ComputeSizeInBytes(dtype, shapeArray, allowMisaligned: false);

        if ((ulong)data.Length != expectedSize)
        {
            throw SafetensorException.InvalidTensorView(dtype, shapeArray, (ulong)data.Length);
        }

        DataType = dtype;
        Data = data;
        _shape = shapeArray;
    }

    public SliceIterator Slice(params TensorIndexer[] slices)
    {
        ArgumentNullException.ThrowIfNull(slices);
        return new SliceIterator(this, slices);
    }

    public bool Equals(TensorView? other)
    {
        if (other is null)
        {
            return false;
        }

        return DataType == other.DataType
            && _shape.SequenceEqual(other._shape)
            && Data.Span.SequenceEqual(other.Data.Span);
    }

    public override bool Equals(object? obj) => Equals(obj as TensorView);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(DataType);
        hash.Add(Data.Length);
        foreach (var dim in _shape)
        {
            hash.Add(dim);
        }

        return hash.ToHashCode();
    }
}
