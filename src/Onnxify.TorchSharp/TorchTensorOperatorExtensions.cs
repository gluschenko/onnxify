namespace Onnxify.TorchSharp;

/// <summary>
/// Converts common Torch tensor-style operators into explicit Onnxify graph fragments.
/// </summary>
/// <remarks>
/// These helpers cover the most common arithmetic, matrix, and shape operators that
/// appear in Torch graphs but are not module-backed. They follow the same explicit-graph
/// approach as the module exporters in <see cref="TorchModuleExtensions"/>.
/// </remarks>
public static class TorchTensorOperatorExtensions
{
    [TorchOp("aten::add.Tensor")]
    public static IOnnxGraphEdge ExportAdd(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other,
        float alpha = 1f
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return graph.Add(
            name: graph.NextName("add"),
            options: new AddInputOptions
            {
                A = input,
                B = ScaleIfNeeded(graph, other, alpha, "add"),
            }
        );
    }

    [TorchOp("aten::add.Scalar")]
    public static IOnnxGraphEdge ExportAdd(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        float other,
        float alpha = 1f
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var scalar = AddScalar(graph, "add", other * alpha);
        return graph.Add(
            name: graph.NextName("add"),
            options: new AddInputOptions
            {
                A = input,
                B = scalar,
            }
        );
    }

    [TorchOp("aten::sub.Tensor")]
    public static IOnnxGraphEdge ExportSub(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other,
        float alpha = 1f
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return graph.Sub(
            name: graph.NextName("sub"),
            options: new SubInputOptions
            {
                A = input,
                B = ScaleIfNeeded(graph, other, alpha, "sub"),
            }
        );
    }

    [TorchOp("aten::sub.Scalar")]
    public static IOnnxGraphEdge ExportSub(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        float other,
        float alpha = 1f
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var scalar = AddScalar(graph, "sub", other * alpha);
        return graph.Sub(
            name: graph.NextName("sub"),
            options: new SubInputOptions
            {
                A = input,
                B = scalar,
            }
        );
    }

    [TorchOp("aten::mul")]
    [TorchOp("aten::mul.Tensor")]
    public static IOnnxGraphEdge ExportMul(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return graph.Mul(
            name: graph.NextName("mul"),
            options: new MulInputOptions
            {
                A = input,
                B = other,
            }
        );
    }

    [TorchOp("aten::div.Tensor")]
    [TorchOp("aten::divide.Tensor")]
    [TorchOp("aten::true_divide.Tensor")]
    public static IOnnxGraphEdge ExportDiv(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return graph.Div(
            name: graph.NextName("div"),
            options: new DivInputOptions
            {
                A = input,
                B = other,
            }
        );
    }

    [TorchOp("aten::div.Scalar")]
    [TorchOp("aten::divide.Scalar")]
    [TorchOp("aten::true_divide.Scalar")]
    public static IOnnxGraphEdge ExportDiv(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        float other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var scalar = AddScalar(graph, "div", other);
        return graph.Div(
            name: graph.NextName("div"),
            options: new DivInputOptions
            {
                A = input,
                B = scalar,
            }
        );
    }

    [TorchOp("aten::matmul")]
    [TorchOp("aten::mm")]
    [TorchOp("aten::bmm")]
    public static IOnnxGraphEdge ExportMatMul(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return graph.MatMul(
            name: graph.NextName("matmul"),
            options: new MatMulInputOptions
            {
                A = input,
                B = other,
            }
        );
    }

    [TorchOp("aten::reshape")]
    [TorchOp("aten::view")]
    public static IOnnxGraphEdge ExportView(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        params long[] shape
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(shape);

        var name = graph.NextName("reshape");
        var shapeTensor = graph.AddTensor(
            name: $"{name}_shape",
            shape: [shape.LongLength],
            value: shape
        );

        return graph.Reshape(
            name: name,
            options: new ReshapeInputOptions
            {
                Data = input,
                Shape = shapeTensor,
            }
        );
    }

    [TorchOp("aten::permute")]
    public static IOnnxGraphEdge ExportPermute(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        params long[] permutation
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(permutation);

        return graph.Transpose(
            name: graph.NextName("transpose"),
            options: new TransposeInputOptions
            {
                Data = input,
                Perm = permutation,
            }
        );
    }

    [TorchOp("aten::transpose.int")]
    public static IOnnxGraphEdge ExportTranspose(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim0,
        long dim1
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var rank = GetRequiredRank(input, "transpose");
        var normalizedDim0 = NormalizeAxis(dim0, rank, "transpose");
        var normalizedDim1 = NormalizeAxis(dim1, rank, "transpose");
        var permutation = Enumerable.Range(0, rank).Select(static index => (long)index).ToArray();
        (permutation[normalizedDim0], permutation[normalizedDim1]) =
            (permutation[normalizedDim1], permutation[normalizedDim0]);

        return graph.Transpose(
            name: graph.NextName("transpose"),
            options: new TransposeInputOptions
            {
                Data = input,
                Perm = permutation,
            }
        );
    }

    [TorchOp("aten::t")]
    public static IOnnxGraphEdge ExportT(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var rank = GetRequiredRank(input, "t");
        if (rank != 2)
        {
            throw new NotSupportedException($"t export expects a rank-2 tensor, but got rank {rank}.");
        }

        return graph.ExportTranspose(input, 0, 1);
    }

    [TorchOp("aten::unsqueeze")]
    public static IOnnxGraphEdge ExportUnsqueeze(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var rank = GetRequiredRank(input, "unsqueeze");
        var axis = NormalizeInsertionAxis(dim, rank, "unsqueeze");
        var axes = graph.AddTensor<long>(
            name: $"{graph.NextName("unsqueeze")}_axes",
            shape: [1],
            value: [axis]
        );

        return graph.Unsqueeze(
            name: graph.NextName("unsqueeze"),
            options: new UnsqueezeInputOptions
            {
                Data = input,
                Axes = axes,
            }
        );
    }

    [TorchOp("aten::slice.Tensor")]
    public static IOnnxGraphEdge ExportSlice(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        long? start = null,
        long? end = null,
        long step = 1
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        if (step == 0)
        {
            throw new NotSupportedException("slice export does not support step = 0.");
        }

        var rank = GetRequiredRank(input, "slice");
        var axis = NormalizeAxis(dim, rank, "slice");
        var name = graph.NextName("slice");

        var starts = graph.AddTensor<long>(
            name: $"{name}_starts",
            shape: [1],
            value: [start ?? (step > 0 ? 0L : long.MaxValue)]
        );

        var ends = graph.AddTensor<long>(
            name: $"{name}_ends",
            shape: [1],
            value: [end ?? (step > 0 ? long.MaxValue : long.MinValue)]
        );

        var axes = graph.AddTensor<long>(
            name: $"{name}_axes",
            shape: [1],
            value: [axis]
        );

        var steps = graph.AddTensor<long>(
            name: $"{name}_steps",
            shape: [1],
            value: [step]
        );

        return graph.Slice(
            name: name,
            options: new SliceInputOptions
            {
                Data = input,
                Starts = starts,
                Ends = ends,
                Axes = axes,
                Steps = steps,
            }
        );
    }

    [TorchOp("aten::expand")]
    public static IOnnxGraphEdge ExportExpand(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        params long[] shape
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(shape);

        var resolvedShape = ResolveExpandShape(input, shape);
        var name = graph.NextName("expand");
        var shapeTensor = graph.AddTensor<long>(
            name: $"{name}_shape",
            shape: [resolvedShape.LongLength],
            value: resolvedShape
        );

        return graph.Expand(
            name: name,
            options: new ExpandInputOptions
            {
                Input = input,
                Shape = shapeTensor,
            }
        );
    }

    [TorchOp("aten::expand_as")]
    public static IOnnxGraphEdge ExportExpandAs(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        var name = graph.NextName("expand");
        var shape = graph.Shape(
            name: $"{name}_shape",
            options: new ShapeInputOptions
            {
                Data = other,
            }
        );

        return graph.Expand(
            name: name,
            options: new ExpandInputOptions
            {
                Input = input,
                Shape = shape,
            }
        );
    }

    [TorchOp("aten::cat")]
    [TorchOp("aten::concat")]
    [TorchOp("aten::concatenate")]
    public static IOnnxGraphEdge ExportConcat(
        this OnnxGraph graph,
        IReadOnlyList<IOnnxGraphEdge> inputs,
        long dim = 0
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(inputs);

        if (inputs.Count == 0)
        {
            throw new NotSupportedException("concat export requires at least one input tensor.");
        }

        return graph.Concat(
            name: graph.NextName("concat"),
            options: new ConcatInputOptions
            {
                In = inputs.ToArray(),
                Axis = dim,
            }
        );
    }

    [TorchOp("aten::split")]
    public static IReadOnlyList<IOnnxGraphEdge> ExportSplit(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long splitSize,
        long dim = 0
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        if (splitSize <= 0)
        {
            throw new NotSupportedException("split export requires splitSize > 0.");
        }

        var axisSize = GetRequiredStaticDimensionSize(input, dim, "split");
        var outputCount = checked((int)((axisSize + splitSize - 1) / splitSize));

        return graph.Split(
            name: graph.NextName("split"),
            options: new SplitInputOutputOptions
            {
                Input = input,
                Axis = dim,
                NumOutputs = outputCount,
                Out = CreateOutputEdges(graph, "split", outputCount),
            }
        );
    }

    [TorchOp("aten::split.Tensor")]
    public static IReadOnlyList<IOnnxGraphEdge> ExportSplit(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IReadOnlyList<long> splitSizes,
        long dim = 0
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(splitSizes);

        if (splitSizes.Count == 0)
        {
            throw new NotSupportedException("split export requires at least one split size.");
        }

        if (splitSizes.Any(static x => x < 0))
        {
            throw new NotSupportedException("split export does not support negative split sizes.");
        }

        var name = graph.NextName("split");
        var splitTensor = graph.AddTensor<long>(
            name: $"{name}_sizes",
            shape: [splitSizes.Count],
            value: splitSizes.ToArray()
        );

        return graph.Split(
            name: name,
            options: new SplitInputOutputOptions
            {
                Input = input,
                InputSplit = splitTensor,
                Axis = dim,
                Out = CreateOutputEdges(graph, "split", splitSizes.Count),
            }
        );
    }

    [TorchOp("aten::chunk")]
    public static IReadOnlyList<IOnnxGraphEdge> ExportChunk(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long chunks,
        long dim = 0
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        if (chunks <= 0)
        {
            throw new NotSupportedException("chunk export requires chunks > 0.");
        }

        var axisSize = GetRequiredStaticDimensionSize(input, dim, "chunk");
        var splitSize = (axisSize + chunks - 1) / chunks;

        return graph.ExportSplit(input, splitSize, dim);
    }

    [TorchOp("aten::select.int")]
    public static IOnnxGraphEdge ExportSelect(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        long index
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var indexTensor = graph.AddTensor<long>(
            name: $"{graph.NextName("select")}_index",
            shape: [],
            value: [index]
        );

        return graph.Gather(
            name: graph.NextName("select"),
            options: new GatherInputOptions
            {
                Data = input,
                Indices = indexTensor,
                Axis = dim,
            }
        );
    }

    [TorchOp("aten::squeeze")]
    public static IOnnxGraphEdge ExportSqueeze(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.Squeeze(
            name: graph.NextName("squeeze"),
            options: new SqueezeInputOptions
            {
                Data = input,
            }
        );
    }

    [TorchOp("aten::squeeze.dim")]
    public static IOnnxGraphEdge ExportSqueeze(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var axis = TryGetRank(input) is int rank
            ? NormalizeAxis(dim, rank, "squeeze")
            : dim;
        var axes = graph.AddTensor<long>(
            name: $"{graph.NextName("squeeze")}_axes",
            shape: [1],
            value: [axis]
        );

        return graph.Squeeze(
            name: graph.NextName("squeeze"),
            options: new SqueezeInputOptions
            {
                Data = input,
                Axes = axes,
            }
        );
    }

    [TorchOp("aten::gather")]
    public static IOnnxGraphEdge ExportGather(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        IOnnxGraphEdge index
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(index);

        return graph.GatherElements(
            name: graph.NextName("gather"),
            options: new GatherElementsInputOptions
            {
                Data = input,
                Indices = index,
                Axis = dim,
            }
        );
    }

    [TorchOp("aten::where.self")]
    public static IOnnxGraphEdge ExportWhere(
        this OnnxGraph graph,
        IOnnxGraphEdge condition,
        IOnnxGraphEdge self,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(condition);
        ArgumentNullException.ThrowIfNull(self);
        ArgumentNullException.ThrowIfNull(other);

        return graph.Where(
            name: graph.NextName("where"),
            options: new WhereInputOptions
            {
                Condition = condition,
                X = self,
                Y = other,
            }
        );
    }

    [TorchOp("aten::where.ScalarSelf")]
    public static IOnnxGraphEdge ExportWhere(
        this OnnxGraph graph,
        IOnnxGraphEdge condition,
        double self,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(condition);
        ArgumentNullException.ThrowIfNull(other);

        return graph.ExportWhere(condition, AddScalarLike(graph, other, "where", self), other);
    }

    [TorchOp("aten::where.ScalarOther")]
    public static IOnnxGraphEdge ExportWhere(
        this OnnxGraph graph,
        IOnnxGraphEdge condition,
        IOnnxGraphEdge self,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(condition);
        ArgumentNullException.ThrowIfNull(self);

        return graph.ExportWhere(condition, self, AddScalarLike(graph, self, "where", other));
    }

    [TorchOp("aten::where.Scalar")]
    public static IOnnxGraphEdge ExportWhere(
        this OnnxGraph graph,
        IOnnxGraphEdge condition,
        double self,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(condition);

        var name = graph.NextName("where");
        var selfTensor = graph.AddTensor<float>($"{name}_self", [], [checked((float)self)]);
        var otherTensor = graph.AddTensor<float>($"{name}_other", [], [checked((float)other)]);

        return graph.Where(
            name: name,
            options: new WhereInputOptions
            {
                Condition = condition,
                X = selfTensor,
                Y = otherTensor,
            }
        );
    }

    [TorchOp("aten::masked_fill.Scalar")]
    public static IOnnxGraphEdge ExportMaskedFill(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge mask,
        double value
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(mask);

        return graph.ExportWhere(mask, AddScalarLike(graph, input, "masked_fill", value), input);
    }

    [TorchOp("aten::masked_fill.Tensor")]
    public static IOnnxGraphEdge ExportMaskedFill(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge mask,
        IOnnxGraphEdge value
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(mask);
        ArgumentNullException.ThrowIfNull(value);

        return graph.ExportWhere(mask, value, input);
    }

    [TorchOp("aten::triu")]
    public static IOnnxGraphEdge ExportTriu(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long diagonal = 0
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var name = graph.NextName("triu");
        IOnnxGraphEdge? diagonalTensor = diagonal == 0
            ? null
            : graph.AddTensor<long>($"{name}_diagonal", [], [diagonal]);

        return graph.Trilu(
            name: name,
            options: new TriluInputOptions
            {
                Input = input,
                K = diagonalTensor,
                Upper = 1,
            }
        );
    }

    [TorchOp("aten::sum")]
    public static IOnnxGraphEdge ExportSum(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ReduceSum(
            name: graph.NextName("sum"),
            options: new ReduceSumInputOptions
            {
                Data = input,
                Keepdims = 0,
            }
        );
    }

    [TorchOp("aten::sum.dim_IntList")]
    public static IOnnxGraphEdge ExportSum(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IReadOnlyList<long> dims,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(dims);

        return graph.ReduceSum(
            name: graph.NextName("sum"),
            options: new ReduceSumInputOptions
            {
                Data = input,
                Axes = AddAxesTensor(graph, "sum", dims),
                Keepdims = keepdim ? 1 : 0,
            }
        );
    }

    [TorchOp("aten::mean")]
    public static IOnnxGraphEdge ExportMean(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ReduceMean(
            name: graph.NextName("mean"),
            options: new ReduceMeanInputOptions
            {
                Data = input,
                Keepdims = 0,
            }
        );
    }

    [TorchOp("aten::mean.dim")]
    public static IOnnxGraphEdge ExportMean(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IReadOnlyList<long> dims,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(dims);

        return graph.ReduceMean(
            name: graph.NextName("mean"),
            options: new ReduceMeanInputOptions
            {
                Data = input,
                Axes = AddAxesTensor(graph, "mean", dims),
                Keepdims = keepdim ? 1 : 0,
            }
        );
    }

    [TorchOp("aten::repeat")]
    public static IOnnxGraphEdge ExportRepeat(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        params long[] repeats
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(repeats);

        var name = graph.NextName("repeat");
        var repeatsTensor = graph.AddTensor<long>(
            name: $"{name}_repeats",
            shape: [repeats.LongLength],
            value: repeats
        );

        return graph.Tile(
            name: name,
            options: new TileInputOptions
            {
                Input = input,
                Repeats = repeatsTensor,
            }
        );
    }

    [TorchOp("aten::stack")]
    public static IOnnxGraphEdge ExportStack(
        this OnnxGraph graph,
        IReadOnlyList<IOnnxGraphEdge> inputs,
        long dim = 0
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(inputs);

        if (inputs.Count == 0)
        {
            throw new NotSupportedException("stack export requires at least one input tensor.");
        }

        var rank = GetRequiredRank(inputs[0], "stack");
        var axis = NormalizeInsertionAxis(dim, rank, "stack");
        var unsqueezed = inputs
            .Select(x => graph.ExportUnsqueeze(x, axis))
            .ToArray();

        return graph.ExportConcat(unsqueezed, axis);
    }

    [TorchOp("aten::topk")]
    public static TopKOutput ExportTopK(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long k,
        long dim = -1,
        bool largest = true,
        bool sorted = true
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        if (k <= 0)
        {
            throw new NotSupportedException("topk export requires k > 0.");
        }

        var name = graph.NextName("topk");
        var kTensor = graph.AddTensor<long>(
            name: $"{name}_k",
            shape: [1],
            value: [k]
        );

        return graph.TopK(
            name: name,
            options: new TopKInputOptions
            {
                X = input,
                K = kTensor,
                Axis = dim,
                Largest = largest ? 1 : 0,
                Sorted = sorted ? 1 : 0,
            }
        );
    }

    [TorchOp("aten::argmax")]
    public static IOnnxGraphEdge ExportArgMax(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ArgMax(
            name: graph.NextName("argmax"),
            options: new ArgMaxInputOptions
            {
                Data = input,
                Axis = dim,
                Keepdims = keepdim ? 1 : 0,
            }
        );
    }

    [TorchOp("aten::amax")]
    public static IOnnxGraphEdge ExportAMax(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IReadOnlyList<long> dims,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(dims);

        return graph.ReduceMax(
            name: graph.NextName("amax"),
            options: new ReduceMaxInputOptions
            {
                Data = input,
                Axes = AddAxesTensor(graph, "amax", dims),
                Keepdims = keepdim ? 1 : 0,
            }
        );
    }

    [TorchOp("aten::sort")]
    public static TopKOutput ExportSort(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim = -1,
        bool descending = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var axisSize = GetRequiredStaticDimensionSize(input, dim, "sort");
        return graph.ExportTopK(
            input,
            k: axisSize,
            dim: dim,
            largest: descending,
            sorted: true
        );
    }

    [TorchOp("aten::pow.Tensor_Scalar")]
    public static IOnnxGraphEdge ExportPow(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double exponent
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.Pow(
            name: graph.NextName("pow"),
            options: new PowInputOptions
            {
                X = input,
                Y = AddScalarLike(graph, input, "pow", exponent),
            }
        );
    }

    [TorchOp("aten::pow.Tensor_Tensor")]
    public static IOnnxGraphEdge ExportPow(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge exponent
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(exponent);

        return graph.Pow(
            name: graph.NextName("pow"),
            options: new PowInputOptions
            {
                X = input,
                Y = exponent,
            }
        );
    }

    [TorchOp("aten::sqrt")]
    public static IOnnxGraphEdge ExportSqrt(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.Sqrt(
            name: graph.NextName("sqrt"),
            options: new SqrtInputOptions
            {
                X = input,
            }
        );
    }

    [TorchOp("aten::rsqrt")]
    public static IOnnxGraphEdge ExportRSqrt(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var sqrt = graph.ExportSqrt(input);
        return graph.Div(
            name: graph.NextName("rsqrt"),
            options: new DivInputOptions
            {
                A = AddScalarLike(graph, input, "rsqrt", 1d),
                B = sqrt,
            }
        );
    }

    private static IOnnxGraphEdge ScaleIfNeeded(
        OnnxGraph graph,
        IOnnxGraphEdge input,
        float scale,
        string prefix
    )
    {
        if (scale == 1f)
        {
            return input;
        }

        var scalar = AddScalar(graph, prefix, scale);
        return graph.Mul(
            name: graph.NextName($"{prefix}_scale"),
            options: new MulInputOptions
            {
                A = input,
                B = scalar,
            }
        );
    }

    private static IOnnxGraphEdge AddScalar(OnnxGraph graph, string prefix, float value)
    {
        return graph.AddTensor(
            name: $"{graph.NextName(prefix)}_scalar",
            shape: [],
            value: [value]
        );
    }

    private static IOnnxGraphEdge AddScalarLike(
        OnnxGraph graph,
        IOnnxGraphEdge reference,
        string prefix,
        double value
    )
    {
        var dataType = GetTensorDataType(reference)
            ?? throw new NotSupportedException(
                $"{prefix} export requires the reference tensor element type to be known."
            );

        var name = $"{graph.NextName(prefix)}_scalar";

        if (dataType == typeof(float))
        {
            return graph.AddTensor<float>(name, [], [checked((float)value)]);
        }

        if (dataType == typeof(double))
        {
            return graph.AddTensor<double>(name, [], [value]);
        }

        if (dataType == typeof(long))
        {
            return graph.AddTensor<long>(name, [], [checked((long)value)]);
        }

        if (dataType == typeof(int))
        {
            return graph.AddTensor<int>(name, [], [checked((int)value)]);
        }

        if (dataType == typeof(short))
        {
            return graph.AddTensor<short>(name, [], [checked((short)value)]);
        }

        if (dataType == typeof(byte))
        {
            return graph.AddTensor<byte>(name, [], [checked((byte)value)]);
        }

        if (dataType == typeof(sbyte))
        {
            return graph.AddTensor<sbyte>(name, [], [checked((sbyte)value)]);
        }

        if (dataType == typeof(uint))
        {
            return graph.AddTensor<uint>(name, [], [checked((uint)value)]);
        }

        if (dataType == typeof(ulong))
        {
            return graph.AddTensor<ulong>(name, [], [checked((ulong)value)]);
        }

        if (dataType == typeof(ushort))
        {
            return graph.AddTensor<ushort>(name, [], [checked((ushort)value)]);
        }

        if (dataType == typeof(bool))
        {
            return graph.AddTensor<bool>(name, [], [value != 0d]);
        }

        throw new NotSupportedException(
            $"{prefix} export does not support scalar literals for element type '{dataType.Name}'."
        );
    }

    private static IOnnxGraphEdge AddAxesTensor(
        OnnxGraph graph,
        string prefix,
        IReadOnlyList<long> axes
    )
    {
        return graph.AddTensor<long>(
            name: $"{graph.NextName(prefix)}_axes",
            shape: [axes.Count],
            value: axes.ToArray()
        );
    }

    private static int GetRequiredRank(IOnnxGraphEdge input, string opName)
    {
        if (TryGetRank(input) is int rank)
        {
            return rank;
        }

        throw new NotSupportedException(
            $"{opName} export requires a statically known input rank."
        );
    }

    private static int? TryGetRank(IOnnxGraphEdge input)
    {
        if (input is OnnxValue<OnnxTensorType> value && value.Type.Shape is not null)
        {
            return value.Type.Shape.Dimensions.Length;
        }

        return null;
    }

    private static long GetRequiredStaticDimensionSize(IOnnxGraphEdge input, long axis, string opName)
    {
        if (input is OnnxValue<OnnxTensorType> value && value.Type.Shape is not null)
        {
            var rank = value.Type.Shape.Dimensions.Length;
            var normalizedAxis = NormalizeAxis(axis, rank, opName);

            if (value.Type.Shape.Dimensions[normalizedAxis] is OnnxDimension<long> dimension)
            {
                return dimension.Value;
            }
        }

        throw new NotSupportedException(
            $"{opName} export requires a statically known size for axis {axis}."
        );
    }

    private static long[] ResolveExpandShape(IOnnxGraphEdge input, IReadOnlyList<long> shape)
    {
        var result = shape.ToArray();
        if (!result.Contains(-1))
        {
            return result;
        }

        if (input is not OnnxValue<OnnxTensorType> value || value.Type.Shape is null)
        {
            throw new NotSupportedException(
                "expand export requires a statically known input shape when the target shape contains -1."
            );
        }

        var inputShape = value.Type.Shape.Dimensions;
        if (result.Length < inputShape.Length)
        {
            throw new NotSupportedException(
                "expand export does not support target shapes with fewer dimensions than the input rank when using -1."
            );
        }

        var offset = result.Length - inputShape.Length;
        for (var i = 0; i < result.Length; i++)
        {
            if (result[i] != -1)
            {
                continue;
            }

            var inputIndex = i - offset;
            if (inputIndex < 0 || inputShape[inputIndex] is not OnnxDimension<long> dimension)
            {
                throw new NotSupportedException(
                    "expand export requires every -1 target dimension to map to a statically known input dimension."
                );
            }

            result[i] = dimension.Value;
        }

        return result;
    }

    private static IOnnxGraphEdge[] CreateOutputEdges(OnnxGraph graph, string prefix, int count)
    {
        var result = new IOnnxGraphEdge[count];
        for (var i = 0; i < count; i++)
        {
            result[i] = graph.AddEdge(graph.NextName($"{prefix}_{i}"));
        }

        return result;
    }

    private static Type? GetTensorDataType(IOnnxGraphEdge edge)
    {
        return edge switch
        {
            OnnxValue value when value.Type is OnnxTensorType tensorType => tensorType.Type,
            OnnxTensor tensor => tensor.DataType,
            _ => null,
        };
    }

    private static long NormalizeInsertionAxis(long axis, int inputRank, string opName)
    {
        var outputRank = inputRank + 1;
        var normalized = axis < 0 ? axis + outputRank : axis;
        if (normalized < 0 || normalized >= outputRank)
        {
            throw new NotSupportedException(
                $"{opName} export axis {axis} is out of range for output rank {outputRank}."
            );
        }

        return normalized;
    }

    private static int NormalizeAxis(long axis, int rank, string opName)
    {
        var normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank)
        {
            throw new NotSupportedException(
                $"{opName} export axis {axis} is out of range for rank {rank}."
            );
        }

        return checked((int)normalized);
    }
}
