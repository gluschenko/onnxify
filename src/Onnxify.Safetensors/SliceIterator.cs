namespace Onnxify.Safetensors;

public sealed class SliceIterator : IReadOnlyList<ReadOnlyMemory<byte>>
{
    private readonly TensorView _view;
    private readonly SliceRange[] _indices;
    private readonly ulong[] _newShape;

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

    public ulong RemainingByteLength => (ulong)_indices.Sum(x => x.Stop - x.Start);

    public IReadOnlyList<ulong> NewShape => _newShape;

    public int Count => _indices.Length;

    public ReadOnlyMemory<byte> this[int index]
    {
        get
        {
            var range = _indices[index];
            return _view.Data.Slice(range.Start, range.Stop - range.Start);
        }
    }

    public IEnumerator<ReadOnlyMemory<byte>> GetEnumerator()
    {
        for (var i = 0; i < _indices.Length; i++)
        {
            yield return this[i];
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    private static ResolvedSlice ResolveSlice(TensorIndexer slice, ulong dimSize, int dimIndex)
    {
        return slice switch
        {
            SelectTensorIndexer select => ResolveSelect(select, dimSize, dimIndex),
            NarrowTensorIndexer narrow => ResolveNarrow(narrow, dimSize, dimIndex),
            _ => throw new ArgumentOutOfRangeException(nameof(slice), slice, null),
        };
    }

    private static ResolvedSlice ResolveSelect(SelectTensorIndexer slice, ulong dimSize, int dimIndex)
    {
        var start = slice.Index;
        var stop = checked(slice.Index + 1);
        EnsureBounds(dimIndex, dimSize, start, stop);
        return new ResolvedSlice(start, stop, true);
    }

    private static ResolvedSlice ResolveNarrow(NarrowTensorIndexer slice, ulong dimSize, int dimIndex)
    {
        var start = ResolveLowerBound(slice.Start);
        var stop = ResolveUpperBound(slice.End, dimSize);
        EnsureBounds(dimIndex, dimSize, start, stop);
        return new ResolvedSlice(start, stop, false);
    }

    private static ulong ResolveLowerBound(TensorBounds bound)
    {
        if (bound.Value is null)
        {
            return 0;
        }

        return bound.Inclusive ? bound.Value.Value : checked(bound.Value.Value + 1);
    }

    private static ulong ResolveUpperBound(TensorBounds bound, ulong dimSize)
    {
        if (bound.Value is null)
        {
            return dimSize;
        }

        return bound.Inclusive ? checked(bound.Value.Value + 1) : bound.Value.Value;
    }

    private static void EnsureBounds(int dimIndex, ulong dimSize, ulong start, ulong stop)
    {
        if (start >= dimSize || stop > dimSize)
        {
            var asked = start >= dimSize ? start : stop == 0 ? 0 : stop - 1;
            throw InvalidSliceException.SliceOutOfRange(dimIndex, asked, dimSize);
        }
    }

    private static void EnsureByteAligned(ulong value, ulong span)
    {
        if ((value * span) % 8 != 0)
        {
            throw InvalidSliceException.MisalignedSlice();
        }
    }

    private static int CheckedInt(ulong value)
    {
        if (value > int.MaxValue)
        {
            throw new OverflowException();
        }

        return (int)value;
    }
}
