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
    [TorchOp("_operator::add")]
    [TorchOp("prims::add")]
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
    [TorchOp("aten::subtract.Tensor")]
    [TorchOp("_operator::sub")]
    [TorchOp("prims::sub")]
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
    [TorchOp("aten::subtract.Scalar")]
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
    [TorchOp("aten::multiply.Tensor")]
    [TorchOp("_operator::mul")]
    [TorchOp("prims::mul")]
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
    [TorchOp("prims::div")]
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

    [TorchOp("aten::abs")]
    [TorchOp("_operator::abs")]
    [TorchOp("prims::abs")]
    public static IOnnxGraphEdge ExportAbs(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "abs", "Abs", input);
    }

    [TorchOp("aten::neg")]
    [TorchOp("_operator::neg")]
    public static IOnnxGraphEdge ExportNeg(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "neg", "Neg", input);
    }

    [TorchOp("aten::exp")]
    [TorchOp("prims::exp")]
    public static IOnnxGraphEdge ExportExp(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "exp", "Exp", input);
    }

    [TorchOp("aten::log")]
    [TorchOp("prims::log")]
    public static IOnnxGraphEdge ExportLog(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "log", "Log", input);
    }

    [TorchOp("aten::sin")]
    [TorchOp("prims::sin")]
    public static IOnnxGraphEdge ExportSin(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "sin", "Sin", input);
    }

    [TorchOp("aten::cos")]
    public static IOnnxGraphEdge ExportCos(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "cos", "Cos", input);
    }

    [TorchOp("aten::tan")]
    [TorchOp("prims::tan")]
    public static IOnnxGraphEdge ExportTan(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "tan", "Tan", input);
    }

    [TorchOp("aten::floor")]
    [TorchOp("math::floor")]
    [TorchOp("prims::floor")]
    public static IOnnxGraphEdge ExportFloor(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "floor", "Floor", input);
    }

    [TorchOp("aten::ceil")]
    [TorchOp("math::ceil")]
    [TorchOp("prims::ceil")]
    public static IOnnxGraphEdge ExportCeil(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "ceil", "Ceil", input);
    }

    [TorchOp("aten::round")]
    [TorchOp("prims::round")]
    public static IOnnxGraphEdge ExportRound(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "round", "Round", input);
    }

    [TorchOp("aten::trunc")]
    [TorchOp("math::trunc")]
    public static IOnnxGraphEdge ExportTrunc(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "trunc", "Trunc", input);
    }

    [TorchOp("aten::reciprocal")]
    public static IOnnxGraphEdge ExportReciprocal(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "reciprocal", "Reciprocal", input);
    }

    [TorchOp("aten::sign")]
    public static IOnnxGraphEdge ExportSign(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "sign", "Sign", input);
    }

    [TorchOp("aten::acos")]
    [TorchOp("prims::acos")]
    public static IOnnxGraphEdge ExportAcos(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "acos", "Acos", input);
    }

    [TorchOp("aten::acosh")]
    [TorchOp("prims::acosh")]
    public static IOnnxGraphEdge ExportAcosh(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "acosh", "Acosh", input);
    }

    [TorchOp("aten::asin")]
    [TorchOp("prims::asin")]
    public static IOnnxGraphEdge ExportAsin(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "asin", "Asin", input);
    }

    [TorchOp("aten::asinh")]
    [TorchOp("prims::asinh")]
    public static IOnnxGraphEdge ExportAsinh(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "asinh", "Asinh", input);
    }

    [TorchOp("aten::atan")]
    [TorchOp("prims::atan")]
    public static IOnnxGraphEdge ExportAtan(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "atan", "Atan", input);
    }

    [TorchOp("aten::atanh")]
    [TorchOp("prims::atanh")]
    public static IOnnxGraphEdge ExportAtanh(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "atanh", "Atanh", input);
    }

    [TorchOp("aten::cosh")]
    [TorchOp("prims::cosh")]
    public static IOnnxGraphEdge ExportCosh(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "cosh", "Cosh", input);
    }

    [TorchOp("aten::sinh")]
    [TorchOp("prims::sinh")]
    public static IOnnxGraphEdge ExportSinh(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "sinh", "Sinh", input);
    }

    [TorchOp("aten::erf")]
    [TorchOp("aten::special_erf")]
    [TorchOp("prims::erf")]
    public static IOnnxGraphEdge ExportErf(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "erf", "Erf", input);
    }

    [TorchOp("aten::erfc")]
    [TorchOp("aten::special_erfc")]
    public static IOnnxGraphEdge ExportErfc(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var erf = graph.ExportErf(input);
        return graph.ExportSub(AddScalarLike(graph, input, "erfc", 1d), erf);
    }

    [TorchOp("aten::expm1")]
    [TorchOp("aten::special_expm1")]
    public static IOnnxGraphEdge ExportExpm1(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var exp = graph.ExportExp(input);
        return graph.ExportSub(exp, AddScalarLike(graph, input, "expm1", 1d));
    }

    [TorchOp("aten::log1p")]
    public static IOnnxGraphEdge ExportLog1P(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var shifted = graph.ExportAdd(input, AddScalarLike(graph, input, "log1p", 1d));
        return graph.ExportLog(shifted);
    }

    [TorchOp("aten::log2")]
    public static IOnnxGraphEdge ExportLog2(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var log = graph.ExportLog(input);
        return graph.ExportDiv(log, AddScalarLike(graph, input, "log2", Math.Log(2d)));
    }

    [TorchOp("aten::signbit")]
    public static IOnnxGraphEdge ExportSignBit(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportLess(input, AddScalarLike(graph, input, "signbit", 0d));
    }

    [TorchOp("aten::eq.Tensor")]
    [TorchOp("_operator::eq")]
    [TorchOp("prims::eq")]
    public static IOnnxGraphEdge ExportEqual(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "eq", "Equal", input, other);
    }

    [TorchOp("aten::eq.Scalar")]
    public static IOnnxGraphEdge ExportEqual(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportEqual(input, AddScalarLike(graph, input, "eq", other));
    }

    [TorchOp("aten::ne")]
    [TorchOp("aten::ne.Tensor")]
    [TorchOp("_operator::ne")]
    [TorchOp("prims::ne")]
    public static IOnnxGraphEdge ExportNotEqual(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return graph.ExportLogicalNot(graph.ExportEqual(input, other));
    }

    [TorchOp("aten::ne.Scalar")]
    public static IOnnxGraphEdge ExportNotEqual(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportNotEqual(input, AddScalarLike(graph, input, "ne", other));
    }

    [TorchOp("aten::lt.Tensor")]
    [TorchOp("_operator::lt")]
    [TorchOp("aten::less.Tensor")]
    public static IOnnxGraphEdge ExportLess(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "lt", "Less", input, other);
    }

    [TorchOp("aten::lt.Scalar")]
    public static IOnnxGraphEdge ExportLess(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportLess(input, AddScalarLike(graph, input, "lt", other));
    }

    [TorchOp("aten::le.Tensor")]
    [TorchOp("_operator::le")]
    [TorchOp("aten::less_equal.Tensor")]
    public static IOnnxGraphEdge ExportLessOrEqual(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "le", "LessOrEqual", input, other);
    }

    [TorchOp("aten::le.Scalar")]
    public static IOnnxGraphEdge ExportLessOrEqual(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportLessOrEqual(input, AddScalarLike(graph, input, "le", other));
    }

    [TorchOp("aten::gt.Tensor")]
    [TorchOp("_operator::gt")]
    [TorchOp("aten::greater.Tensor")]
    public static IOnnxGraphEdge ExportGreater(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "gt", "Greater", input, other);
    }

    [TorchOp("aten::gt.Scalar")]
    public static IOnnxGraphEdge ExportGreater(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportGreater(input, AddScalarLike(graph, input, "gt", other));
    }

    [TorchOp("aten::ge.Tensor")]
    [TorchOp("_operator::ge")]
    [TorchOp("aten::greater_equal.Tensor")]
    public static IOnnxGraphEdge ExportGreaterOrEqual(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "ge", "GreaterOrEqual", input, other);
    }

    [TorchOp("aten::ge.Scalar")]
    public static IOnnxGraphEdge ExportGreaterOrEqual(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportGreaterOrEqual(input, AddScalarLike(graph, input, "ge", other));
    }

    [TorchOp("aten::logical_not")]
    public static IOnnxGraphEdge ExportLogicalNot(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "logical_not", "Not", input);
    }

    [TorchOp("aten::logical_and")]
    public static IOnnxGraphEdge ExportLogicalAnd(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "logical_and", "And", input, other);
    }

    [TorchOp("aten::logical_or")]
    public static IOnnxGraphEdge ExportLogicalOr(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "logical_or", "Or", input, other);
    }

    [TorchOp("aten::logical_xor")]
    public static IOnnxGraphEdge ExportLogicalXor(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "logical_xor", "Xor", input, other);
    }

    [TorchOp("aten::maximum")]
    public static IOnnxGraphEdge ExportMaximum(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "maximum", "Max", input, other);
    }

    [TorchOp("aten::minimum")]
    public static IOnnxGraphEdge ExportMinimum(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(graph, "minimum", "Min", input, other);
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
    [TorchOp("prims::reshape")]
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
    [TorchOp("prims::transpose")]
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
    [TorchOp("aten::broadcast_to")]
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

    [TorchOp("aten::view_as")]
    public static IOnnxGraphEdge ExportViewAs(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        var name = graph.NextName("view_as");
        var shape = graph.Shape(
            name: $"{name}_shape",
            options: new ShapeInputOptions
            {
                Data = other,
            }
        );

        return graph.Reshape(
            name: name,
            options: new ReshapeInputOptions
            {
                Data = input,
                Shape = shape,
            }
        );
    }

    [TorchOp("aten::alias")]
    [TorchOp("aten::clone")]
    [TorchOp("aten::contiguous")]
    [TorchOp("aten::detach")]
    [TorchOp("aten::resolve_neg")]
    public static IOnnxGraphEdge ExportIdentity(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "identity", "Identity", input);
    }

    [TorchOp("aten::index_select")]
    public static IOnnxGraphEdge ExportIndexSelect(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        IOnnxGraphEdge index
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(index);

        return graph.Gather(
            name: graph.NextName("index_select"),
            options: new GatherInputOptions
            {
                Data = input,
                Indices = index,
                Axis = dim,
            }
        );
    }

    [TorchOp("aten::narrow")]
    public static IOnnxGraphEdge ExportNarrow(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        long start,
        long length
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        if (length < 0)
        {
            throw new NotSupportedException("narrow export requires length >= 0.");
        }

        return graph.ExportSlice(
            input,
            dim: dim,
            start: start,
            end: checked(start + length),
            step: 1
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
    [TorchOp("prims::squeeze")]
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
    [TorchOp("prims::where")]
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

    [TorchOp("aten::all")]
    public static IOnnxGraphEdge ExportAll(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportTruthReduction(graph, "all", input, null, keepdim: false, useMin: true);
    }

    [TorchOp("aten::all.dim")]
    public static IOnnxGraphEdge ExportAll(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportTruthReduction(graph, "all", input, [dim], keepdim, useMin: true);
    }

    [TorchOp("aten::all.dims")]
    public static IOnnxGraphEdge ExportAll(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IReadOnlyList<long> dims,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(dims);

        return ExportTruthReduction(graph, "all", input, dims, keepdim, useMin: true);
    }

    [TorchOp("aten::any")]
    public static IOnnxGraphEdge ExportAny(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportTruthReduction(graph, "any", input, null, keepdim: false, useMin: false);
    }

    [TorchOp("aten::any.dim")]
    public static IOnnxGraphEdge ExportAny(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportTruthReduction(graph, "any", input, [dim], keepdim, useMin: false);
    }

    [TorchOp("aten::any.dims")]
    public static IOnnxGraphEdge ExportAny(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IReadOnlyList<long> dims,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(dims);

        return ExportTruthReduction(graph, "any", input, dims, keepdim, useMin: false);
    }

    [TorchOp("aten::nonzero")]
    public static IOnnxGraphEdge ExportNonZero(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportUnaryNode(graph, "nonzero", "NonZero", input);
    }

    [TorchOp("aten::cumsum")]
    public static IOnnxGraphEdge ExportCumSum(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var name = graph.NextName("cumsum");
        var axis = graph.AddTensor<long>($"{name}_axis", [1], [dim]);

        return graph.CumSum(
            name: name,
            options: new CumSumInputOptions
            {
                X = input,
                Axis = axis,
            }
        );
    }

    [TorchOp("aten::clamp")]
    public static IOnnxGraphEdge ExportClamp(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double? min = null,
        double? max = null
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportClampCore(
            graph,
            input,
            min is null ? null : AddScalarLike(graph, input, "clamp_min", min.Value),
            max is null ? null : AddScalarLike(graph, input, "clamp_max", max.Value)
        );
    }

    [TorchOp("aten::clamp.Tensor")]
    public static IOnnxGraphEdge ExportClamp(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge? min,
        IOnnxGraphEdge? max
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return ExportClampCore(graph, input, min, max);
    }

    [TorchOp("aten::clamp_min")]
    public static IOnnxGraphEdge ExportClampMin(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double min
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportMaximum(input, AddScalarLike(graph, input, "clamp_min", min));
    }

    [TorchOp("aten::clamp_min.Tensor")]
    public static IOnnxGraphEdge ExportClampMin(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge min
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(min);

        return graph.ExportMaximum(input, min);
    }

    [TorchOp("aten::clamp_max")]
    public static IOnnxGraphEdge ExportClampMax(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double max
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportMinimum(input, AddScalarLike(graph, input, "clamp_max", max));
    }

    [TorchOp("aten::clamp_max.Tensor")]
    public static IOnnxGraphEdge ExportClampMax(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge max
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(max);

        return graph.ExportMinimum(input, max);
    }

    [TorchOp("aten::fmod.Tensor")]
    public static IOnnxGraphEdge ExportFMod(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        return ExportBinaryNode(
            graph,
            "fmod",
            "Mod",
            input,
            other,
            [new OnnxAttribute<long>("fmod", 1)]
        );
    }

    [TorchOp("aten::fmod.Scalar")]
    public static IOnnxGraphEdge ExportFMod(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportFMod(input, AddScalarLike(graph, input, "fmod", other));
    }

    [TorchOp("aten::remainder.Tensor")]
    public static IOnnxGraphEdge ExportRemainder(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        if (IsFloatingPointTensorType(input))
        {
            var quotient = graph.ExportDiv(input, other);
            var floored = graph.ExportFloor(quotient);
            return graph.ExportSub(input, graph.ExportMul(floored, other));
        }

        return ExportBinaryNode(graph, "remainder", "Mod", input, other, [new OnnxAttribute<long>("fmod", 0)]);
    }

    [TorchOp("aten::remainder.Scalar")]
    public static IOnnxGraphEdge ExportRemainder(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        double other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportRemainder(input, AddScalarLike(graph, input, "remainder", other));
    }

    [TorchOp("aten::remainder.Scalar_Tensor")]
    public static IOnnxGraphEdge ExportRemainder(
        this OnnxGraph graph,
        double input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(other);

        return graph.ExportRemainder(AddScalarLike(graph, other, "remainder", input), other);
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

    [TorchOp("aten::tril")]
    public static IOnnxGraphEdge ExportTril(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long diagonal = 0
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var name = graph.NextName("tril");
        IOnnxGraphEdge? diagonalTensor = diagonal == 0
            ? null
            : graph.AddTensor<long>($"{name}_diagonal", [], [diagonal]);

        return graph.Trilu(
            name: name,
            options: new TriluInputOptions
            {
                Input = input,
                K = diagonalTensor,
                Upper = 0,
            }
        );
    }

    [TorchOp("aten::sum")]
    [TorchOp("prims::sum")]
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

    [TorchOp("aten::prod")]
    public static IOnnxGraphEdge ExportProd(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ReduceProd(
            name: graph.NextName("prod"),
            options: new ReduceProdInputOptions
            {
                Data = input,
                Keepdims = 0,
            }
        );
    }

    [TorchOp("aten::prod.dim_int")]
    public static IOnnxGraphEdge ExportProd(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ReduceProd(
            name: graph.NextName("prod"),
            options: new ReduceProdInputOptions
            {
                Data = input,
                Axes = AddAxesTensor(graph, "prod", [dim]),
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
    [TorchOp("aten::tile")]
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

    [TorchOp("aten::argmin")]
    public static IOnnxGraphEdge ExportArgMin(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        long dim,
        bool keepdim = false
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ArgMin(
            name: graph.NextName("argmin"),
            options: new ArgMinInputOptions
            {
                Data = input,
                Axis = dim,
                Keepdims = keepdim ? 1 : 0,
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

    [TorchOp("aten::deg2rad")]
    public static IOnnxGraphEdge ExportDeg2Rad(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportMul(input, AddScalarLike(graph, input, "deg2rad", Math.PI / 180d));
    }

    [TorchOp("aten::rad2deg")]
    public static IOnnxGraphEdge ExportRad2Deg(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        return graph.ExportMul(input, AddScalarLike(graph, input, "rad2deg", 180d / Math.PI));
    }

    [TorchOp("aten::exp2")]
    public static IOnnxGraphEdge ExportExp2(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var scaled = graph.ExportMul(input, AddScalarLike(graph, input, "exp2", Math.Log(2d)));
        return graph.ExportExp(scaled);
    }

    [TorchOp("aten::frac")]
    public static IOnnxGraphEdge ExportFrac(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var absolute = graph.ExportAbs(input);
        var floored = graph.ExportFloor(absolute);
        var fractional = graph.ExportSub(absolute, floored);
        return graph.ExportMul(fractional, graph.ExportSign(input));
    }

    [TorchOp("aten::log10")]
    public static IOnnxGraphEdge ExportLog10(
        this OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        var log = graph.ExportLog(input);
        return graph.ExportDiv(log, AddScalarLike(graph, input, "log10", Math.Log(10d)));
    }

    [TorchOp("aten::type_as")]
    public static IOnnxGraphEdge ExportTypeAs(
        this OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge other
    )
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(other);

        var otherType = GetTensorDataType(other)
            ?? throw new NotSupportedException("type_as export requires the target tensor element type to be known.");

        return ExportCastTo(graph, "type_as", input, GetOnnxTensorDataType(otherType));
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
    [TorchOp("prims::pow")]
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
    [TorchOp("prims::sqrt")]
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

    private static IOnnxGraphEdge ExportUnaryNode(
        OnnxGraph graph,
        string prefix,
        string opType,
        IOnnxGraphEdge input,
        IEnumerable<OnnxAttribute>? attributes = null
    )
    {
        var name = graph.NextName(prefix);
        var output = graph.AddEdge($"{name}_output");
        graph.AddNode(
            name: name,
            opType: opType,
            domain: string.Empty,
            docString: string.Empty,
            inputs: [input],
            outputs: [output],
            attributes: attributes ?? []
        );

        return output;
    }

    private static IOnnxGraphEdge ExportBinaryNode(
        OnnxGraph graph,
        string prefix,
        string opType,
        IOnnxGraphEdge left,
        IOnnxGraphEdge right,
        IEnumerable<OnnxAttribute>? attributes = null
    )
    {
        var name = graph.NextName(prefix);
        var output = graph.AddEdge($"{name}_output");
        graph.AddNode(
            name: name,
            opType: opType,
            domain: string.Empty,
            docString: string.Empty,
            inputs: [left, right],
            outputs: [output],
            attributes: attributes ?? []
        );

        return output;
    }

    private static IOnnxGraphEdge ExportClampCore(
        OnnxGraph graph,
        IOnnxGraphEdge input,
        IOnnxGraphEdge? min,
        IOnnxGraphEdge? max
    )
    {
        if (min is null && max is null)
        {
            throw new NotSupportedException("clamp export requires at least one bound.");
        }

        var current = input;
        if (min is not null)
        {
            current = graph.ExportMaximum(current, min);
        }

        if (max is not null)
        {
            current = graph.ExportMinimum(current, max);
        }

        return current;
    }

    private static IOnnxGraphEdge ExportTruthReduction(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        IReadOnlyList<long>? dims,
        bool keepdim,
        bool useMin
    )
    {
        var truth = graph.ExportNotEqual(input, AddScalarLike(graph, input, $"{prefix}_truth", 0d));
        var truthInt = ExportCastTo(graph, $"{prefix}_cast", truth, 7L);

        var reduced = useMin
            ? graph.ReduceMin(
                name: graph.NextName(prefix),
                options: new ReduceMinInputOptions
                {
                    Data = truthInt,
                    Axes = dims is null ? null : AddAxesTensor(graph, prefix, dims),
                    Keepdims = keepdim ? 1 : 0,
                }
            )
            : graph.ReduceMax(
                name: graph.NextName(prefix),
                options: new ReduceMaxInputOptions
                {
                    Data = truthInt,
                    Axes = dims is null ? null : AddAxesTensor(graph, prefix, dims),
                    Keepdims = keepdim ? 1 : 0,
                }
            );

        var zero = graph.AddTensor<long>(
            name: $"{prefix}_{graph.Initializers.Count}_zero",
            shape: [],
            value: [0L]
        );

        return graph.ExportGreater(reduced, zero);
    }

    private static bool IsFloatingPointTensorType(IOnnxGraphEdge edge)
    {
        var dataType = GetTensorDataType(edge);
        return dataType == typeof(float) || dataType == typeof(double) || dataType == typeof(Half);
    }

    private static IOnnxGraphEdge ExportCastTo(
        OnnxGraph graph,
        string prefix,
        IOnnxGraphEdge input,
        long to
    )
    {
        return ExportUnaryNode(
            graph,
            prefix,
            "Cast",
            input,
            [new OnnxAttribute<long>("to", to)]
        );
    }

    private static long GetOnnxTensorDataType(Type type)
    {
        if (type == typeof(float))
        {
            return 1L;
        }

        if (type == typeof(double))
        {
            return 11L;
        }

        if (type == typeof(long))
        {
            return 7L;
        }

        if (type == typeof(int))
        {
            return 6L;
        }

        if (type == typeof(short))
        {
            return 5L;
        }

        if (type == typeof(byte))
        {
            return 2L;
        }

        if (type == typeof(sbyte))
        {
            return 3L;
        }

        if (type == typeof(uint))
        {
            return 12L;
        }

        if (type == typeof(ulong))
        {
            return 13L;
        }

        if (type == typeof(ushort))
        {
            return 4L;
        }

        if (type == typeof(bool))
        {
            return 9L;
        }

        if (type == typeof(Half))
        {
            return 10L;
        }

        throw new NotSupportedException($"ONNX cast export does not support tensor element type '{type.Name}'.");
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
            name: $"{prefix}_{graph.Initializers.Count}_scalar",
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

        var name = $"{prefix}_{graph.Initializers.Count}_scalar";

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
