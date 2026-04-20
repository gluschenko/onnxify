namespace Onnxify.Safetensors;

/// <summary>
/// Represents a validated zero-copy tensor payload view together with the data type and shape needed to interpret it.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>TensorView</c>.
/// </remarks>
public sealed class TensorView : IEquatable<TensorView>
{
    private readonly ulong[] _shape;

    /// <summary>
    /// Gets the element type used to interpret the underlying bytes.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>TensorView.dtype</c>.
    /// </remarks>
    public DataType DataType { get; }

    /// <summary>
    /// Gets the raw tensor payload bytes without copying them.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>TensorView.data</c>.
    /// </remarks>
    public ReadOnlyMemory<byte> Data { get; }

    /// <summary>
    /// Gets the logical tensor shape.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>TensorView.shape</c>.
    /// </remarks>
    public IReadOnlyList<ulong> Shape => _shape;

    /// <summary>
    /// Initializes a tensor view and validates that the supplied byte length exactly matches the implied size of the shape and data type.
    /// </summary>
    /// <param name="dtype">The tensor element type.</param>
    /// <param name="shape">The tensor shape.</param>
    /// <param name="data">The raw tensor payload bytes.</param>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>TensorView::new</c>.
    /// </remarks>
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

    /// <summary>
    /// Creates a slice iterator that describes how to reconstruct a logical sub-tensor from the underlying payload bytes.
    /// </summary>
    /// <param name="slices">The per-dimension slice/index operations to apply.</param>
    /// <returns>A slice iterator over the matching payload chunks.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>IndexOp::slice</c> over <c>TensorView</c>.
    /// </remarks>
    public SliceIterator Slice(params TensorIndexer[] slices)
    {
        ArgumentNullException.ThrowIfNull(slices);
        return new SliceIterator(this, slices);
    }

    /// <summary>
    /// Determines whether another tensor view has the same data type, shape, and byte contents.
    /// </summary>
    /// <param name="other">The tensor view to compare.</param>
    /// <returns><see langword="true"/> when both views describe identical tensor content; otherwise <see langword="false"/>.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: local C# equality helper added for managed tests and collection behavior.
    /// </remarks>
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

    /// <summary>
    /// Determines whether another object is an equal tensor view.
    /// </summary>
    /// <param name="obj">The object to compare.</param>
    /// <returns><see langword="true"/> when <paramref name="obj"/> is an equal tensor view; otherwise <see langword="false"/>.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: local C# equality helper added for managed tests and collection behavior.
    /// </remarks>
    public override bool Equals(object? obj) => Equals(obj as TensorView);

    /// <summary>
    /// Produces a hash code based on the data type, payload length, and tensor shape.
    /// </summary>
    /// <returns>A hash code suitable for dictionary and set usage.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: local C# equality helper added for managed tests and collection behavior.
    /// </remarks>
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
