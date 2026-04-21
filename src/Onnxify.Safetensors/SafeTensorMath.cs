namespace Onnxify.Safetensors;

/// <summary>
/// Provides overflow-checked tensor size helpers shared by metadata validation and tensor view construction.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: local C# helper extracted from size calculations inside <c>Metadata::validate</c> and <c>TensorView::new</c>.
/// </remarks>
internal static class SafeTensorMath
{
    /// <summary>
    /// Multiplies all dimensions in a shape to produce the tensor element count.
    /// </summary>
    /// <param name="shape">The tensor shape to multiply.</param>
    /// <returns>The number of logical elements described by the shape.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: shape product logic used by <c>Metadata::validate</c> and <c>TensorView::new</c>.
    /// </remarks>
    public static ulong ComputeElementCount(IReadOnlyList<ulong> shape)
    {
        ArgumentNullException.ThrowIfNull(shape);

        ulong n = 1;
        checked
        {
            foreach (var dim in shape)
            {
                n *= dim;
            }
        }

        return n;
    }

    /// <summary>
    /// Computes the byte length implied by a tensor data type and shape, optionally tolerating non-byte-aligned sub-byte shapes.
    /// </summary>
    /// <param name="dtype">The tensor element type.</param>
    /// <param name="shape">The tensor shape.</param>
    /// <param name="allowMisaligned"><see langword="true"/> to allow sub-byte shapes that do not end on a byte boundary; otherwise an exception is thrown.</param>
    /// <returns>The required byte count for the tensor payload.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: byte-size validation logic used by <c>Metadata::validate</c> and <c>TensorView::new</c>.
    /// </remarks>
    public static ulong ComputeSizeInBytes(DataType dtype, IReadOnlyList<ulong> shape, bool allowMisaligned)
    {
        try
        {
            var elementCount = ComputeElementCount(shape);
            var bitCount = checked(elementCount * (ulong)dtype.Bitsize());

            if (!allowMisaligned && bitCount % 8 != 0)
            {
                throw SafeTensorException.MisalignedSlice();
            }

            return bitCount / 8;
        }
        catch (OverflowException)
        {
            throw SafeTensorException.ValidationOverflow();
        }
    }
}
