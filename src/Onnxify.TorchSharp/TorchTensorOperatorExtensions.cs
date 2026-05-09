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

    private static int GetRequiredRank(IOnnxGraphEdge input, string opName)
    {
        if (input is OnnxValue<OnnxTensorType> value && value.Type.Shape is not null)
        {
            return value.Type.Shape.Dimensions.Length;
        }

        throw new NotSupportedException(
            $"{opName} export requires a statically known input rank."
        );
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
