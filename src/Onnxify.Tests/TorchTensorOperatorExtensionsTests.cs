using System.Reflection;
using Onnxify.TorchSharp;

namespace Onnxify.Tests;

public sealed class TorchTensorOperatorExtensionsTests
{
    [Fact]
    public void ExportAdd_WithTensorAlpha_EmitsMulThenAdd()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([2L, 3L]));

        var output = graph.ExportAdd(input, other, alpha: 0.5f);

        Assert.NotNull(output);
        Assert.Collection(
            graph.Nodes,
            mul => Assert.Equal("Mul", mul.OpType),
            add => Assert.Equal("Add", add.OpType)
        );
    }

    [Fact]
    public void ExportSub_WithScalarAlpha_EmitsSubAndScalarInitializer()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([4L]));

        var output = graph.ExportSub(input, 3f, alpha: 2f);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("Sub", node.OpType);

        var scalar = Assert.Single(graph.Initializers);
        var typed = Assert.IsType<OnnxTensor<float>>(scalar);
        Assert.Equal([6f], typed.Value.ToArray());
    }

    [Fact]
    public void ExportMul_EmitsMulNode()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([2L, 3L]));

        var output = graph.ExportMul(input, other);

        Assert.NotNull(output);
        Assert.Equal("Mul", Assert.Single(graph.Nodes).OpType);
    }

    [Fact]
    public void ExportDiv_WithScalar_EmitsDivNode()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));

        var output = graph.ExportDiv(input, 4f);

        Assert.NotNull(output);
        Assert.Equal("Div", Assert.Single(graph.Nodes).OpType);
    }

    [Fact]
    public void ExportNewUnaryOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));

        graph.ExportAbs(input);
        graph.ExportNeg(input);
        graph.ExportExp(input);
        graph.ExportLog(input);
        graph.ExportSin(input);
        graph.ExportCos(input);
        graph.ExportTan(input);
        graph.ExportFloor(input);
        graph.ExportCeil(input);
        graph.ExportRound(input);
        graph.ExportTrunc(input);
        graph.ExportReciprocal(input);
        graph.ExportSign(input);

        Assert.Equal(
            ["Abs", "Neg", "Exp", "Log", "Sin", "Cos", "Tan", "Floor", "Ceil", "Round", "Trunc", "Reciprocal", "Sign"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void ExportNewComparisonAndLogicalOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([2L, 3L]));
        var boolLeft = graph.AddInput("bool_left", OnnxTensorType.Create<bool>([2L, 3L]));
        var boolRight = graph.AddInput("bool_right", OnnxTensorType.Create<bool>([2L, 3L]));

        graph.ExportEqual(input, other);
        graph.ExportEqual(input, 0.0);
        graph.ExportLess(input, other);
        graph.ExportLessOrEqual(input, other);
        graph.ExportGreater(input, other);
        graph.ExportGreaterOrEqual(input, other);
        graph.ExportLogicalNot(boolLeft);
        graph.ExportLogicalAnd(boolLeft, boolRight);
        graph.ExportLogicalOr(boolLeft, boolRight);
        graph.ExportLogicalXor(boolLeft, boolRight);
        graph.ExportMaximum(input, other);
        graph.ExportMinimum(input, other);

        Assert.Equal(
            ["Equal", "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Not", "And", "Or", "Xor", "Max", "Min"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void ExportMatMul_EmitsMatMulNode()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([3L, 5L]));

        var output = graph.ExportMatMul(input, other);

        Assert.NotNull(output);
        Assert.Equal("MatMul", Assert.Single(graph.Nodes).OpType);
    }

    [Fact]
    public void ExportView_EmitsReshapeAndShapeTensor()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L, 4L]));

        var output = graph.ExportView(input, 0L, 12L);

        Assert.NotNull(output);
        Assert.Equal("Reshape", Assert.Single(graph.Nodes).OpType);

        var shapeTensor = Assert.Single(graph.Initializers);
        var typed = Assert.IsType<OnnxTensor<long>>(shapeTensor);
        Assert.Equal([0L, 12L], typed.Value.ToArray());
    }

    [Fact]
    public void ExportPermute_EmitsTransposeWithRequestedPermutation()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L, 4L]));

        var output = graph.ExportPermute(input, 2L, 0L, 1L);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("Transpose", node.OpType);
        Assert.Equal(
            [2L, 0L, 1L],
            Assert.IsType<long[]>(node.Attributes.Single(x => x.Name == "perm").GetValue())
        );
    }

    [Fact]
    public void ExportTranspose_EmitsSwapPermutation()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L, 4L]));

        var output = graph.ExportTranspose(input, 0L, 2L);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("Transpose", node.OpType);
        Assert.Equal(
            [2L, 1L, 0L],
            Assert.IsType<long[]>(node.Attributes.Single(x => x.Name == "perm").GetValue())
        );
    }

    [Fact]
    public void ExportT_EmitsMatrixTranspose()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));

        var output = graph.ExportT(input);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("Transpose", node.OpType);
        Assert.Equal(
            [1L, 0L],
            Assert.IsType<long[]>(node.Attributes.Single(x => x.Name == "perm").GetValue())
        );
    }

    [Fact]
    public void ExportUnsqueezeAndSqueeze_EmitAxesTensors()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L, 4L]));

        var expanded = graph.ExportUnsqueeze(input, -1);
        var squeezed = graph.ExportSqueeze(expanded, -1);

        Assert.NotNull(squeezed);
        Assert.Collection(
            graph.Nodes,
            unsqueeze => Assert.Equal("Unsqueeze", unsqueeze.OpType),
            squeeze => Assert.Equal("Squeeze", squeeze.OpType)
        );

        Assert.Collection(
            graph.Initializers.OfType<OnnxTensor<long>>(),
            unsqueezeAxes => Assert.Equal([3L], unsqueezeAxes.Value.ToArray()),
            squeezeAxes => Assert.Equal([-1L], squeezeAxes.Value.ToArray())
        );
    }

    [Fact]
    public void ExportSlice_EmitsSliceNodeWithFourIndexTensors()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 8L, 4L]));

        var output = graph.ExportSlice(input, dim: 1, start: 2, end: 7, step: 2);

        Assert.NotNull(output);
        Assert.Equal("Slice", Assert.Single(graph.Nodes).OpType);

        var values = graph.Initializers.OfType<OnnxTensor<long>>().Select(x => x.Value.Single()).ToArray();
        Assert.Equal([2L, 7L, 1L, 2L], values);
    }

    [Fact]
    public void ExportExpandAs_EmitsShapeThenExpand()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([1L, 3L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([2L, 3L]));

        var output = graph.ExportExpandAs(input, other);

        Assert.NotNull(output);
        Assert.Collection(
            graph.Nodes,
            shape => Assert.Equal("Shape", shape.OpType),
            expand => Assert.Equal("Expand", expand.OpType)
        );
    }

    [Fact]
    public void ExportConcatAndStack_EmitConcatBasedGraphs()
    {
        var graph = CreateGraph();
        var left = graph.AddInput("left", OnnxTensorType.Create<float>([2L, 3L]));
        var right = graph.AddInput("right", OnnxTensorType.Create<float>([2L, 3L]));

        var concat = graph.ExportConcat(new IOnnxGraphEdge[] { left, right }, dim: 1);
        var stack = graph.ExportStack(new IOnnxGraphEdge[] { left, right }, dim: 0);

        Assert.NotNull(concat);
        Assert.NotNull(stack);
        Assert.Equal(
            ["Concat", "Unsqueeze", "Unsqueeze", "Concat"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void ExportSplitAndChunk_EmitSplitNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 5L]));

        var split = graph.ExportSplit(input, splitSize: 2, dim: 1);
        var chunk = graph.ExportChunk(input, chunks: 2, dim: 1);

        Assert.Equal(3, split.Count);
        Assert.Equal(2, chunk.Count);
        Assert.Equal(["Split", "Split"], graph.Nodes.Select(x => x.OpType).ToArray());
    }

    [Fact]
    public void ExportSelectAndGather_EmitGatherFamilyNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L, 4L]));
        var index = graph.AddInput("index", OnnxTensorType.Create<long>([2L, 3L, 4L]));

        var selected = graph.ExportSelect(input, dim: 1, index: 0);
        var gathered = graph.ExportGather(input, dim: 2, index);

        Assert.NotNull(selected);
        Assert.NotNull(gathered);
        Assert.Equal(["Gather", "GatherElements"], graph.Nodes.Select(x => x.OpType).ToArray());
    }

    [Fact]
    public void ExportWhereAndMaskedFill_EmitWhereNodes()
    {
        var graph = CreateGraph();
        var condition = graph.AddInput("condition", OnnxTensorType.Create<bool>([2L, 3L]));
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([2L, 3L]));

        var where = graph.ExportWhere(condition, input, other);
        var masked = graph.ExportMaskedFill(input, condition, 0.0);

        Assert.NotNull(where);
        Assert.NotNull(masked);
        Assert.Equal(["Where", "Where"], graph.Nodes.Select(x => x.OpType).ToArray());
    }

    [Fact]
    public void ExportTriuAndReductions_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L, 4L]));

        var triu = graph.ExportTriu(input, diagonal: 1);
        var sum = graph.ExportSum(input, new long[] { 1L }, keepdim: false);
        var mean = graph.ExportMean(input);
        var amax = graph.ExportAMax(input, new long[] { 2L }, keepdim: true);

        Assert.NotNull(triu);
        Assert.NotNull(sum);
        Assert.NotNull(mean);
        Assert.NotNull(amax);
        Assert.Equal(
            ["Trilu", "ReduceSum", "ReduceMean", "ReduceMax"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void ExportNewReductionOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L, 4L]));

        var tril = graph.ExportTril(input, diagonal: -1);
        var prod = graph.ExportProd(input);
        var prodDim = graph.ExportProd(input, dim: 1, keepdim: true);
        var argmin = graph.ExportArgMin(input, dim: 2, keepdim: false);

        Assert.NotNull(tril);
        Assert.NotNull(prod);
        Assert.NotNull(prodDim);
        Assert.NotNull(argmin);
        Assert.Equal(
            ["Trilu", "ReduceProd", "ReduceProd", "ArgMin"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void ExportModeratelyTrivialUnaryOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));

        graph.ExportAcos(input);
        graph.ExportAcosh(input);
        graph.ExportAsin(input);
        graph.ExportAsinh(input);
        graph.ExportAtan(input);
        graph.ExportAtanh(input);
        graph.ExportCosh(input);
        graph.ExportSinh(input);
        graph.ExportErf(input);
        graph.ExportErfc(input);
        graph.ExportExpm1(input);
        graph.ExportLog1P(input);
        graph.ExportLog2(input);
        graph.ExportSignBit(input);

        Assert.Equal(
            ["Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Cosh", "Sinh", "Erf", "Erf", "Sub", "Exp", "Sub", "Add", "Log", "Log", "Div", "Less"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void ExportModeratelyTrivialIndexAndSearchOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 8L, 4L]));
        var index = graph.AddInput("index", OnnxTensorType.Create<long>([3L]));

        var selected = graph.ExportIndexSelect(input, dim: 1, index: index);
        var narrowed = graph.ExportNarrow(input, dim: 1, start: 2, length: 3);
        var nonZero = graph.ExportNonZero(input);
        var cumsum = graph.ExportCumSum(input, dim: 2);

        Assert.NotNull(selected);
        Assert.NotNull(narrowed);
        Assert.NotNull(nonZero);
        Assert.NotNull(cumsum);
        Assert.Equal(
            ["Gather", "Slice", "NonZero", "CumSum"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );

        var gather = graph.Nodes[0];
        Assert.Equal(1L, Assert.IsType<long>(gather.Attributes.Single(x => x.Name == "axis").GetValue()));

        var axisTensor = graph.Initializers
            .OfType<OnnxTensor<long>>()
            .Single(tensor => tensor.Name.EndsWith("_axis", StringComparison.Ordinal));
        Assert.Equal([2L], axisTensor.Value.ToArray());
    }

    [Fact]
    public void ExportClampAndModOperators_EmitExpectedNodesAndAttributes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([2L, 3L]));
        var min = graph.AddInput("min", OnnxTensorType.Create<float>([2L, 3L]));
        var max = graph.AddInput("max", OnnxTensorType.Create<float>([2L, 3L]));

        graph.ExportClamp(input, min: -1d, max: 1d);
        graph.ExportClamp(input, min, max);
        graph.ExportClampMin(input, -2d);
        graph.ExportClampMax(input, max);
        graph.ExportFMod(input, other);
        graph.ExportFMod(input, 2d);
        graph.ExportRemainder(input, other);
        graph.ExportRemainder(input, 3d);
        graph.ExportRemainder(7d, other);

        Assert.Equal(
            ["Max", "Min", "Max", "Min", "Max", "Min", "Mod", "Mod", "Mod", "Mod", "Mod"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );

        var modNodes = graph.Nodes.Where(x => x.OpType == "Mod").ToArray();
        Assert.Equal(5, modNodes.Length);
        Assert.Equal([1L, 1L, 0L, 0L, 0L], modNodes
            .Select(node => Assert.IsType<long>(node.Attributes.Single(x => x.Name == "fmod").GetValue()))
            .ToArray());
    }

    [Fact]
    public void ExportRepeatTopKArgMaxAndSort_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 4L]));

        var repeated = graph.ExportRepeat(input, 1, 2);
        var topk = graph.ExportTopK(input, k: 2, dim: 1);
        var argmax = graph.ExportArgMax(input, dim: 1, keepdim: false);
        var sorted = graph.ExportSort(input, dim: 1, descending: false);

        Assert.NotNull(repeated);
        Assert.NotNull(topk.Values);
        Assert.NotNull(topk.Indices);
        Assert.NotNull(argmax);
        Assert.NotNull(sorted.Values);
        Assert.NotNull(sorted.Indices);
        Assert.Equal(
            ["Tile", "TopK", "ArgMax", "TopK"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void ExportPowSqrtAndRsqrt_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));
        var exponent = graph.AddInput("exponent", OnnxTensorType.Create<float>([2L, 3L]));

        var powScalar = graph.ExportPow(input, 2.0);
        var powTensor = graph.ExportPow(input, exponent);
        var sqrt = graph.ExportSqrt(input);
        var rsqrt = graph.ExportRSqrt(input);

        Assert.NotNull(powScalar);
        Assert.NotNull(powTensor);
        Assert.NotNull(sqrt);
        Assert.NotNull(rsqrt);
        Assert.Equal(
            ["Pow", "Pow", "Sqrt", "Sqrt", "Div"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void TorchModuleExtensions_ExposeRequestedAliasOperators()
    {
        var coveredOperators = typeof(TorchModuleExtensions)
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .SelectMany(static method => method.GetCustomAttributes<TorchOpAttribute>(inherit: false))
            .Select(static attribute => attribute.Name)
            .ToHashSet(StringComparer.Ordinal);

        Assert.Contains("aten::special_softmax", coveredOperators);
        Assert.Contains("aten::native_dropout", coveredOperators);
        Assert.Contains("aten::native_group_norm", coveredOperators);
        Assert.Contains("aten::native_layer_norm", coveredOperators);
        Assert.Contains("aten::convolution", coveredOperators);
    }

    [Fact]
    public void TorchTensorOperatorExtensions_ExposeNewHighValueOperators()
    {
        var coveredOperators = typeof(TorchTensorOperatorExtensions)
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .SelectMany(static method => method.GetCustomAttributes<TorchOpAttribute>(inherit: false))
            .Select(static attribute => attribute.Name)
            .ToHashSet(StringComparer.Ordinal);

        Assert.Contains("aten::unsqueeze", coveredOperators);
        Assert.Contains("aten::slice.Tensor", coveredOperators);
        Assert.Contains("aten::expand_as", coveredOperators);
        Assert.Contains("aten::cat", coveredOperators);
        Assert.Contains("aten::split.Tensor", coveredOperators);
        Assert.Contains("aten::chunk", coveredOperators);
        Assert.Contains("aten::select.int", coveredOperators);
        Assert.Contains("aten::gather", coveredOperators);
        Assert.Contains("aten::where.self", coveredOperators);
        Assert.Contains("aten::masked_fill.Scalar", coveredOperators);
        Assert.Contains("aten::triu", coveredOperators);
        Assert.Contains("aten::sum.dim_IntList", coveredOperators);
        Assert.Contains("aten::mean.dim", coveredOperators);
        Assert.Contains("aten::repeat", coveredOperators);
        Assert.Contains("aten::stack", coveredOperators);
        Assert.Contains("aten::topk", coveredOperators);
        Assert.Contains("aten::argmax", coveredOperators);
        Assert.Contains("aten::amax", coveredOperators);
        Assert.Contains("aten::sort", coveredOperators);
        Assert.Contains("aten::pow.Tensor_Scalar", coveredOperators);
        Assert.Contains("aten::rsqrt", coveredOperators);
        Assert.Contains("aten::abs", coveredOperators);
        Assert.Contains("aten::neg", coveredOperators);
        Assert.Contains("aten::exp", coveredOperators);
        Assert.Contains("aten::log", coveredOperators);
        Assert.Contains("aten::sin", coveredOperators);
        Assert.Contains("aten::cos", coveredOperators);
        Assert.Contains("aten::tan", coveredOperators);
        Assert.Contains("aten::floor", coveredOperators);
        Assert.Contains("aten::ceil", coveredOperators);
        Assert.Contains("aten::round", coveredOperators);
        Assert.Contains("aten::reciprocal", coveredOperators);
        Assert.Contains("aten::sign", coveredOperators);
        Assert.Contains("aten::acos", coveredOperators);
        Assert.Contains("aten::acosh", coveredOperators);
        Assert.Contains("aten::asin", coveredOperators);
        Assert.Contains("aten::asinh", coveredOperators);
        Assert.Contains("aten::atan", coveredOperators);
        Assert.Contains("aten::atanh", coveredOperators);
        Assert.Contains("aten::cosh", coveredOperators);
        Assert.Contains("aten::sinh", coveredOperators);
        Assert.Contains("aten::erf", coveredOperators);
        Assert.Contains("aten::erfc", coveredOperators);
        Assert.Contains("aten::special_erf", coveredOperators);
        Assert.Contains("aten::special_erfc", coveredOperators);
        Assert.Contains("aten::expm1", coveredOperators);
        Assert.Contains("aten::special_expm1", coveredOperators);
        Assert.Contains("aten::log1p", coveredOperators);
        Assert.Contains("aten::log2", coveredOperators);
        Assert.Contains("aten::signbit", coveredOperators);
        Assert.Contains("aten::eq.Tensor", coveredOperators);
        Assert.Contains("aten::eq.Scalar", coveredOperators);
        Assert.Contains("aten::lt.Tensor", coveredOperators);
        Assert.Contains("aten::le.Tensor", coveredOperators);
        Assert.Contains("aten::gt.Tensor", coveredOperators);
        Assert.Contains("aten::ge.Tensor", coveredOperators);
        Assert.Contains("aten::logical_not", coveredOperators);
        Assert.Contains("aten::logical_and", coveredOperators);
        Assert.Contains("aten::logical_or", coveredOperators);
        Assert.Contains("aten::logical_xor", coveredOperators);
        Assert.Contains("aten::maximum", coveredOperators);
        Assert.Contains("aten::minimum", coveredOperators);
        Assert.Contains("aten::index_select", coveredOperators);
        Assert.Contains("aten::narrow", coveredOperators);
        Assert.Contains("aten::nonzero", coveredOperators);
        Assert.Contains("aten::cumsum", coveredOperators);
        Assert.Contains("aten::clamp", coveredOperators);
        Assert.Contains("aten::clamp.Tensor", coveredOperators);
        Assert.Contains("aten::clamp_min", coveredOperators);
        Assert.Contains("aten::clamp_min.Tensor", coveredOperators);
        Assert.Contains("aten::clamp_max", coveredOperators);
        Assert.Contains("aten::clamp_max.Tensor", coveredOperators);
        Assert.Contains("aten::fmod.Tensor", coveredOperators);
        Assert.Contains("aten::fmod.Scalar", coveredOperators);
        Assert.Contains("aten::remainder.Tensor", coveredOperators);
        Assert.Contains("aten::remainder.Scalar", coveredOperators);
        Assert.Contains("aten::remainder.Scalar_Tensor", coveredOperators);
        Assert.Contains("aten::prod", coveredOperators);
        Assert.Contains("aten::prod.dim_int", coveredOperators);
        Assert.Contains("aten::argmin", coveredOperators);
        Assert.Contains("aten::tril", coveredOperators);
        Assert.Contains("_operator::abs", coveredOperators);
        Assert.Contains("_operator::add", coveredOperators);
        Assert.Contains("_operator::eq", coveredOperators);
        Assert.Contains("_operator::ge", coveredOperators);
        Assert.Contains("_operator::gt", coveredOperators);
        Assert.Contains("_operator::le", coveredOperators);
        Assert.Contains("_operator::lt", coveredOperators);
        Assert.Contains("_operator::mul", coveredOperators);
        Assert.Contains("_operator::neg", coveredOperators);
        Assert.Contains("_operator::sub", coveredOperators);
        Assert.Contains("aten::multiply.Tensor", coveredOperators);
        Assert.Contains("aten::subtract.Tensor", coveredOperators);
        Assert.Contains("aten::subtract.Scalar", coveredOperators);
        Assert.Contains("aten::greater.Tensor", coveredOperators);
        Assert.Contains("aten::greater_equal.Tensor", coveredOperators);
        Assert.Contains("aten::less.Tensor", coveredOperators);
        Assert.Contains("aten::less_equal.Tensor", coveredOperators);
        Assert.Contains("math::ceil", coveredOperators);
        Assert.Contains("math::floor", coveredOperators);
        Assert.Contains("math::trunc", coveredOperators);
        Assert.Contains("prims::abs", coveredOperators);
        Assert.Contains("prims::add", coveredOperators);
        Assert.Contains("prims::div", coveredOperators);
        Assert.Contains("prims::eq", coveredOperators);
        Assert.Contains("prims::exp", coveredOperators);
    }

    private static OnnxGraph CreateGraph()
    {
        return OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 21,
            ProducerName = "torch-tensor-tests",
        }).Graph;
    }
}
