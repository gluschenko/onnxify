namespace Onnxify.Safetensors;

/// <summary>
/// Represents the byte-range decomposition for a logical tensor slice and exposes the resulting chunks as a read-only list.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
/// Original Rust entity: <c>SliceIterator</c>.
/// </remarks>
public sealed class SliceIterator : IReadOnlyList<ReadOnlyMemory<byte>>
{
    private readonly TensorView _view;
    private readonly SliceRange[] _indices;
    private readonly ulong[] _newShape;

    /// <summary>
    /// Initializes a slice iterator by normalizing the requested indexers into byte ranges over the source tensor payload.
    /// </summary>
    /// <param name="view">The tensor view being sliced.</param>
    /// <param name="slices">The logical slice/index expressions to apply.</param>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>SliceIterator::new</c>.
    /// </remarks>
    internal SliceIterator(TensorView view, IReadOnlyList<TensorIndexer> slices)
    {
        _view = view;

        if (slices.Count > view.Shape.Count)
        {
            throw InvalidSliceException.TooManySlices();
        }

        var newShape = new List<ulong>(view.Shape.Count);
        var indices = new List<SliceRange>();
        ulong span = (ulong)view.DataType.Bitsize();

        try
        {
            for (var i = view.Shape.Count - 1; i >= 0; i--)
            {
                var dimSize = view.Shape[i];

                if (i >= slices.Count)
                {
                    newShape.Add(dimSize);
                }
                else
                {
                    var slice = slices[i];
                    var (start, stop, isSelect) = ResolveSlice(slice, dimSize, i);

                    if (!isSelect)
                    {
                        newShape.Add(stop - start);
                    }

                    if (indices.Count == 0)
                    {
                        if (start != 0 || stop != dimSize)
                        {
                            EnsureByteAligned(start, span);
                            EnsureByteAligned(stop, span);

                            var offset = checked((start * span) / 8);
                            var smallSpan = checked((stop * span) / 8 - offset);
                            indices.Add(new SliceRange(CheckedInt(offset), CheckedInt(offset + smallSpan)));
                        }
                    }
                    else
                    {
                        var newIndices = new List<SliceRange>(checked((int)((stop - start) * (ulong)indices.Count)));
                        for (var n = start; n < stop; n++)
                        {
                            EnsureByteAligned(n, span);
                            var offset = checked((n * span) / 8);

                            foreach (var range in indices)
                            {
                                newIndices.Add(new SliceRange(
                                    CheckedInt((ulong)range.Start + offset),
                                    CheckedInt((ulong)range.Stop + offset)));
                            }
                        }

                        indices = newIndices;
                    }
                }

                span = checked(span * dimSize);
            }
        }
        catch (OverflowException)
        {
            throw SafetensorException.ValidationOverflow();
        }

        if (indices.Count == 0)
        {
            indices.Add(new SliceRange(0, view.Data.Length));
        }

        newShape.Reverse();

        _indices = indices.ToArray();
        _newShape = newShape.ToArray();
    }

    /// <summary>
    /// Gets the total number of bytes still represented by the current slice result.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>SliceIterator::remaining_byte_len</c>.
    /// </remarks>
    public ulong RemainingByteLength => (ulong)_indices.Sum(x => x.Stop - x.Start);

    /// <summary>
    /// Gets the shape of the logical tensor produced by the slice.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>SliceIterator::newshape</c>.
    /// </remarks>
    public IReadOnlyList<ulong> NewShape => _newShape;

    /// <summary>
    /// Gets the number of contiguous byte chunks needed to materialize the sliced tensor.
    /// </summary>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: chunk count implied by <c>SliceIterator.indices</c>.
    /// </remarks>
    public int Count => _indices.Length;

    /// <summary>
    /// Gets the byte chunk at the specified slice-iterator position.
    /// </summary>
    /// <param name="index">The zero-based chunk index.</param>
    /// <returns>A view over the requested chunk of the original tensor payload.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>Iterator for SliceIterator::next</c>.
    /// </remarks>
    public ReadOnlyMemory<byte> this[int index]
    {
        get
        {
            var range = _indices[index];
            return _view.Data.Slice(range.Start, range.Stop - range.Start);
        }
    }

    /// <summary>
    /// Enumerates the contiguous byte chunks that make up the logical slice.
    /// </summary>
    /// <returns>An enumerator over the slice chunks.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>Iterator for SliceIterator</c>.
    /// </remarks>
    public IEnumerator<ReadOnlyMemory<byte>> GetEnumerator()
    {
        for (var i = 0; i < _indices.Length; i++)
        {
            yield return this[i];
        }
    }

    /// <summary>
    /// Enumerates the contiguous byte chunks that make up the logical slice.
    /// </summary>
    /// <returns>A non-generic enumerator over the slice chunks.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>Iterator for SliceIterator</c>.
    /// </remarks>
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>
    /// Normalizes a high-level tensor indexer into a concrete start/stop range and selection mode.
    /// </summary>
    /// <param name="slice">The user-facing tensor indexer.</param>
    /// <param name="dimSize">The size of the dimension being sliced.</param>
    /// <param name="dimIndex">The zero-based dimension index.</param>
    /// <returns>A normalized range ready for byte-range expansion.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>SliceIterator::new</c> match over <c>TensorIndexer</c>.
    /// </remarks>
    private static ResolvedSlice ResolveSlice(TensorIndexer slice, ulong dimSize, int dimIndex)
    {
        return slice switch
        {
            SelectTensorIndexer select => ResolveSelect(select, dimSize, dimIndex),
            NarrowTensorIndexer narrow => ResolveNarrow(narrow, dimSize, dimIndex),
            _ => throw new ArgumentOutOfRangeException(nameof(slice), slice, null),
        };
    }

    /// <summary>
    /// Converts a single-index selection into an executable slice range.
    /// </summary>
    /// <param name="slice">The selection indexer.</param>
    /// <param name="dimSize">The size of the addressed dimension.</param>
    /// <param name="dimIndex">The zero-based dimension index.</param>
    /// <returns>A one-element normalized slice.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>TensorIndexer::Select</c> handling in <c>SliceIterator::new</c>.
    /// </remarks>
    private static ResolvedSlice ResolveSelect(SelectTensorIndexer slice, ulong dimSize, int dimIndex)
    {
        var start = slice.Index;
        var stop = checked(slice.Index + 1);
        EnsureBounds(dimIndex, dimSize, start, stop);
        return new ResolvedSlice(start, stop, true);
    }

    /// <summary>
    /// Converts a range-based selection into an executable slice range.
    /// </summary>
    /// <param name="slice">The range indexer.</param>
    /// <param name="dimSize">The size of the addressed dimension.</param>
    /// <param name="dimIndex">The zero-based dimension index.</param>
    /// <returns>A normalized slice range.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: <c>TensorIndexer::Narrow</c> handling in <c>SliceIterator::new</c>.
    /// </remarks>
    private static ResolvedSlice ResolveNarrow(NarrowTensorIndexer slice, ulong dimSize, int dimIndex)
    {
        var start = ResolveLowerBound(slice.Start);
        var stop = ResolveUpperBound(slice.End, dimSize);
        EnsureBounds(dimIndex, dimSize, start, stop);
        return new ResolvedSlice(start, stop, false);
    }

    /// <summary>
    /// Converts a lower-bound descriptor into an inclusive start index.
    /// </summary>
    /// <param name="bound">The lower-bound descriptor.</param>
    /// <returns>The inclusive start index.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: lower-bound translation inside <c>SliceIterator::new</c>.
    /// </remarks>
    private static ulong ResolveLowerBound(TensorBounds bound)
    {
        if (bound.Value is null)
        {
            return 0;
        }

        return bound.Inclusive ? bound.Value.Value : checked(bound.Value.Value + 1);
    }

    /// <summary>
    /// Converts an upper-bound descriptor into an exclusive stop index.
    /// </summary>
    /// <param name="bound">The upper-bound descriptor.</param>
    /// <param name="dimSize">The size of the addressed dimension.</param>
    /// <returns>The exclusive stop index.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: upper-bound translation inside <c>SliceIterator::new</c>.
    /// </remarks>
    private static ulong ResolveUpperBound(TensorBounds bound, ulong dimSize)
    {
        if (bound.Value is null)
        {
            return dimSize;
        }

        return bound.Inclusive ? checked(bound.Value.Value + 1) : bound.Value.Value;
    }

    /// <summary>
    /// Ensures a normalized slice range stays within the addressed tensor dimension.
    /// </summary>
    /// <param name="dimIndex">The zero-based dimension index.</param>
    /// <param name="dimSize">The size of the dimension.</param>
    /// <param name="start">The inclusive start index.</param>
    /// <param name="stop">The exclusive stop index.</param>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: bounds checks inside <c>SliceIterator::new</c>.
    /// </remarks>
    private static void EnsureBounds(int dimIndex, ulong dimSize, ulong start, ulong stop)
    {
        if (start >= dimSize || stop > dimSize)
        {
            var asked = start >= dimSize ? start : stop == 0 ? 0 : stop - 1;
            throw InvalidSliceException.SliceOutOfRange(dimIndex, asked, dimSize);
        }
    }

    /// <summary>
    /// Ensures a logical slice position maps to a byte boundary for sub-byte element types.
    /// </summary>
    /// <param name="value">The element-space position being checked.</param>
    /// <param name="span">The current row-major span in bits.</param>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: misalignment checks inside <c>SliceIterator::new</c>.
    /// </remarks>
    private static void EnsureByteAligned(ulong value, ulong span)
    {
        if ((value * span) % 8 != 0)
        {
            throw InvalidSliceException.MisalignedSlice();
        }
    }

    /// <summary>
    /// Converts a validated unsigned byte offset into <see cref="int"/> for use with managed memory slicing APIs.
    /// </summary>
    /// <param name="value">The offset to convert.</param>
    /// <returns>The offset as a signed 32-bit integer.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/slice.rs</c>.
    /// Original Rust entity: checked <c>usize</c> usage around slice offsets.
    /// </remarks>
    private static int CheckedInt(ulong value)
    {
        if (value > int.MaxValue)
        {
            throw new OverflowException();
        }

        return (int)value;
    }
}
