using System.Reflection;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
        var index = graph.AddInput("index", OnnxTensorType.Create<int>([3L]));

        var selected = graph.ExportIndexSelect(input, dim: 1, index: index);
        var narrowed = graph.ExportNarrow(input, dim: 1, start: 2, length: 3);
        var nonZero = graph.ExportNonZero(input);
        var cumsum = graph.ExportCumSum(input, dim: 2);

        Assert.NotNull(selected);
        Assert.NotNull(narrowed);
        Assert.NotNull(nonZero);
        Assert.NotNull(cumsum);
        Assert.Equal(
            ["Reshape", "Cast", "Gather", "Slice", "NonZero", "Transpose", "CumSum"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );

        var gather = graph.Nodes[2];
        Assert.Equal(1L, Assert.IsType<long>(gather.Attributes.Single(x => x.Name == "axis").GetValue()));

        var axisTensor = graph.Initializers
            .OfType<OnnxTensor<long>>()
            .Single(tensor => tensor.Name.EndsWith("_axis", StringComparison.Ordinal));
        Assert.Equal([2L], axisTensor.Value.ToArray());
    }

    [Fact]
    public void ExportCumSum_WithScalarInput_EmitsIdentity()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([]));

        var output = graph.ExportCumSum(input, dim: 0);

        Assert.NotNull(output);
        Assert.Equal("Identity", Assert.Single(graph.Nodes).OpType);
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
            [
                "Max", "Min", "Max", "Min", "Max", "Min",
                "Mod", "Mod",
                "Div", "Floor", "Mul", "Sub",
                "Div", "Floor", "Mul", "Sub",
                "Div", "Floor", "Mul", "Sub",
            ],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );

        var modNodes = graph.Nodes.Where(x => x.OpType == "Mod").ToArray();
        Assert.Equal(2, modNodes.Length);
        Assert.Equal([1L, 1L], modNodes
            .Select(node => Assert.IsType<long>(node.Attributes.Single(x => x.Name == "fmod").GetValue()))
            .ToArray());
    }

    [Fact]
    public void ExportIdentityAliasAndTypeOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([6L]));
        var otherShape = graph.AddInput("other_shape", OnnxTensorType.Create<float>([2L, 3L]));
        var otherType = graph.AddInput("other_type", OnnxTensorType.Create<long>([6L]));

        graph.ExportIdentity(input);
        graph.ExportIdentity(input);
        graph.ExportIdentity(input);
        graph.ExportIdentity(input);
        graph.ExportIdentity(input);
        graph.ExportExpand(input, 2L, 3L);
        graph.ExportRepeat(input, 2L);
        graph.ExportViewAs(input, otherShape);
        graph.ExportTypeAs(input, otherType);

        Assert.Equal(
            ["Identity", "Identity", "Identity", "Identity", "Identity", "Expand", "Tile", "Shape", "Reshape", "CastLike"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );

        var reshape = graph.Nodes.Single(x => x.OpType == "Reshape");
        Assert.Equal(1L, Assert.IsType<long>(reshape.Attributes.Single(x => x.Name == "allowzero").GetValue()));
    }

    [Fact]
    public void ExportAdditionalMathOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([2L, 3L]));

        graph.ExportNotEqual(input, other);
        graph.ExportNotEqual(input, 0d);
        graph.ExportDeg2Rad(input);
        graph.ExportRad2Deg(input);
        graph.ExportExp2(input);
        graph.ExportFrac(input);
        graph.ExportLog10(input);

        Assert.Equal(
            ["Equal", "Not", "Equal", "Not", "Mul", "Mul", "Mul", "Exp", "Abs", "Floor", "Sub", "Sign", "Mul", "Log", "Div"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void ExportAllAndAnyOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 3L, 4L]));

        graph.ExportAll(input);
        graph.ExportAll(input, dim: 1, keepdim: true);
        graph.ExportAll(input, new long[] { 1L, 2L }, keepdim: false);
        graph.ExportAny(input);
        graph.ExportAny(input, dim: 2, keepdim: true);
        graph.ExportAny(input, new long[] { 0L, 2L }, keepdim: false);

        Assert.Equal(
            [
                "Cast", "Cast", "ReduceMin", "Cast",
                "Cast", "Cast", "ReduceMin", "Cast",
                "Cast", "Cast", "ReduceMin", "Cast",
                "Cast", "Cast", "ReduceMax", "Cast",
                "Cast", "Cast", "ReduceMax", "Cast",
                "Cast", "Cast", "ReduceMax", "Cast",
            ],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void Smoke_RuntimeExecutesCompositeFloatOperators()
    {
        var model = CreateRuntimeModel();
        var unit = model.Graph.AddInput("unit", OnnxTensorType.Create<float>([2L]));
        var positive = model.Graph.AddInput("positive", OnnxTensorType.Create<float>([2L]));
        var acosh = model.Graph.AddInput("acosh", OnnxTensorType.Create<float>([2L]));
        var frac = model.Graph.AddInput("frac", OnnxTensorType.Create<float>([2L]));
        var degrees = model.Graph.AddInput("degrees", OnnxTensorType.Create<float>([2L]));
        var radians = model.Graph.AddInput("radians", OnnxTensorType.Create<float>([2L]));

        var acos = model.Graph.ExportAcos(unit);
        var asin = model.Graph.ExportAsin(unit);
        var atan = model.Graph.ExportAtan(unit);
        var atanh = model.Graph.ExportAtanh(unit);
        var cosh = model.Graph.ExportCosh(unit);
        var sinh = model.Graph.ExportSinh(unit);
        var erf = model.Graph.ExportErf(unit);
        var erfc = model.Graph.ExportErfc(unit);
        var expm1 = model.Graph.ExportExpm1(unit);
        var log1p = model.Graph.ExportLog1P(unit);
        var log2 = model.Graph.ExportLog2(positive);
        var log10 = model.Graph.ExportLog10(positive);
        var exp2 = model.Graph.ExportExp2(unit);
        var acoshOutput = model.Graph.ExportAcosh(acosh);
        var fracOutput = model.Graph.ExportFrac(frac);
        var deg2rad = model.Graph.ExportDeg2Rad(degrees);
        var rad2deg = model.Graph.ExportRad2Deg(radians);

        AddFloatOutput(model, "acos", acos, 2);
        AddFloatOutput(model, "asin", asin, 2);
        AddFloatOutput(model, "atan", atan, 2);
        AddFloatOutput(model, "atanh", atanh, 2);
        AddFloatOutput(model, "cosh", cosh, 2);
        AddFloatOutput(model, "sinh", sinh, 2);
        AddFloatOutput(model, "erf", erf, 2);
        AddFloatOutput(model, "erfc", erfc, 2);
        AddFloatOutput(model, "expm1", expm1, 2);
        AddFloatOutput(model, "log1p", log1p, 2);
        AddFloatOutput(model, "log2", log2, 2);
        AddFloatOutput(model, "log10", log10, 2);
        AddFloatOutput(model, "exp2", exp2, 2);
        AddFloatOutput(model, "acosh_out", acoshOutput, 2);
        AddFloatOutput(model, "frac_out", fracOutput, 2);
        AddFloatOutput(model, "deg2rad", deg2rad, 2);
        AddFloatOutput(model, "rad2deg", rad2deg, 2);

        var outputs = RunModel<float>(
            model,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("unit", new DenseTensor<float>(new[] { -0.5f, 0.5f }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("positive", new DenseTensor<float>(new[] { 0.25f, 10f }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("acosh", new DenseTensor<float>(new[] { 1f, 2f }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("frac", new DenseTensor<float>(new[] { -1.75f, 2.25f }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("degrees", new DenseTensor<float>(new[] { 180f, 90f }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("radians", new DenseTensor<float>(new[] { (float)Math.PI, (float)(Math.PI / 2d) }, new[] { 2 })),
            });

        AssertTensorValues(outputs["acos"], [MathF.Acos(-0.5f), MathF.Acos(0.5f)]);
        AssertTensorValues(outputs["asin"], [MathF.Asin(-0.5f), MathF.Asin(0.5f)]);
        AssertTensorValues(outputs["atan"], [MathF.Atan(-0.5f), MathF.Atan(0.5f)]);
        AssertTensorValues(outputs["atanh"], [MathF.Atanh(-0.5f), MathF.Atanh(0.5f)]);
        AssertTensorValues(outputs["cosh"], [MathF.Cosh(-0.5f), MathF.Cosh(0.5f)]);
        AssertTensorValues(outputs["sinh"], [MathF.Sinh(-0.5f), MathF.Sinh(0.5f)]);
        AssertTensorValues(outputs["erf"], [-0.5204999f, 0.5204999f]);
        AssertTensorValues(outputs["erfc"], [1.5204999f, 0.4795001f]);
        AssertTensorValues(outputs["expm1"], [MathF.Exp(-0.5f) - 1f, MathF.Exp(0.5f) - 1f]);
        AssertTensorValues(outputs["log1p"], [MathF.Log(0.5f), MathF.Log(1.5f)]);
        AssertTensorValues(outputs["log2"], [MathF.Log2(0.25f), MathF.Log2(10f)]);
        AssertTensorValues(outputs["log10"], [MathF.Log10(0.25f), MathF.Log10(10f)]);
        AssertTensorValues(outputs["exp2"], [MathF.Pow(2f, -0.5f), MathF.Pow(2f, 0.5f)]);
        AssertTensorValues(outputs["acosh_out"], [0f, MathF.Acosh(2f)]);
        AssertTensorValues(outputs["frac_out"], [-0.75f, 0.25f]);
        AssertTensorValues(outputs["deg2rad"], [MathF.PI, MathF.PI / 2f]);
        AssertTensorValues(outputs["rad2deg"], [180f, 90f]);
    }

    [Fact]
    public void Smoke_RuntimeExecutesIndexClampAndTileOperators()
    {
        var model = CreateRuntimeModel();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([4L]));
        var index = model.Graph.AddInput("index", OnnxTensorType.Create<int>([2L]));
        var scalar = model.Graph.AddInput("scalar", OnnxTensorType.Create<float>([1L]));
        var broadcastInput = model.Graph.AddInput("broadcast_input", OnnxTensorType.Create<float>([1L, 2L]));
        var viewSource = model.Graph.AddInput("view_source", OnnxTensorType.Create<float>([4L]));
        var viewTarget = model.Graph.AddInput("view_target", OnnxTensorType.Create<float>([2L, 2L]));

        var identity = model.Graph.ExportIdentity(input);
        var indexSelect = model.Graph.ExportIndexSelect(input, dim: 0, index);
        var narrow = model.Graph.ExportNarrow(input, dim: 0, start: 1, length: 2);
        var cumsum = model.Graph.ExportCumSum(input, dim: 0);
        var clamp = model.Graph.ExportClamp(input, min: 2d, max: 3.5d);
        var fmod = model.Graph.ExportFMod(input, 2.5d);
        var remainder = model.Graph.ExportRemainder(input, 2d);
        var tile = model.Graph.ExportRepeat(scalar, 4L);
        var broadcast = model.Graph.ExportExpand(broadcastInput, 3L, 2L);
        var viewAs = model.Graph.ExportViewAs(viewSource, viewTarget);

        AddFloatOutput(model, "identity", identity, 4);
        AddFloatOutput(model, "index_select", indexSelect, 2);
        AddFloatOutput(model, "narrow", narrow, 2);
        AddFloatOutput(model, "cumsum", cumsum, 4);
        AddFloatOutput(model, "clamp", clamp, 4);
        AddFloatOutput(model, "fmod", fmod, 4);
        AddFloatOutput(model, "remainder", remainder, 4);
        AddFloatOutput(model, "tile", tile, 4);
        AddFloatOutput(model, "broadcast", broadcast, 3, 2);
        AddFloatOutput(model, "view_as", viewAs, 2, 2);

        var outputs = RunModel<float>(
            model,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 })),
                NamedOnnxValue.CreateFromTensor("index", new DenseTensor<int>(new[] { 3, 1 }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("scalar", new DenseTensor<float>(new[] { 2f }, new[] { 1 })),
                NamedOnnxValue.CreateFromTensor("broadcast_input", new DenseTensor<float>(new[] { 5f, 6f }, new[] { 1, 2 })),
                NamedOnnxValue.CreateFromTensor("view_source", new DenseTensor<float>(new[] { 9f, 8f, 7f, 6f }, new[] { 4 })),
                NamedOnnxValue.CreateFromTensor("view_target", new DenseTensor<float>(new[] { 0f, 0f, 0f, 0f }, new[] { 2, 2 })),
            });

        AssertTensorValues(outputs["identity"], [1f, 2f, 3f, 4f]);
        AssertTensorValues(outputs["index_select"], [4f, 2f]);
        AssertTensorValues(outputs["narrow"], [2f, 3f]);
        AssertTensorValues(outputs["cumsum"], [1f, 3f, 6f, 10f]);
        AssertTensorValues(outputs["clamp"], [2f, 2f, 3f, 3.5f]);
        AssertTensorValues(outputs["fmod"], [1f, 2f, 0.5f, 1.5f]);
        AssertTensorValues(outputs["remainder"], [1f, 0f, 1f, 0f]);
        AssertTensorValues(outputs["tile"], [2f, 2f, 2f, 2f]);
        AssertTensorValues(outputs["broadcast"], [5f, 6f, 5f, 6f, 5f, 6f], 3, 2);
        AssertTensorValues(outputs["view_as"], [9f, 8f, 7f, 6f], 2, 2);
    }

    [Fact]
    public void Smoke_RuntimeExecutesBooleanAndCastingOperators()
    {
        var boolModel = CreateRuntimeModel();
        var lhs = boolModel.Graph.AddInput("lhs", OnnxTensorType.Create<float>([2L, 2L]));
        var rhs = boolModel.Graph.AddInput("rhs", OnnxTensorType.Create<float>([2L, 2L]));
        var boolInput = boolModel.Graph.AddInput("bool_input", OnnxTensorType.Create<bool>([2L, 2L]));

        var ne = boolModel.Graph.ExportNotEqual(lhs, rhs);
        var signbit = boolModel.Graph.ExportSignBit(lhs);
        var signbitBool = boolModel.Graph.ExportSignBit(boolInput);
        var all = boolModel.Graph.ExportAll(lhs, new long[] { 1L }, keepdim: false);
        var any = boolModel.Graph.ExportAny(lhs, new long[] { 1L }, keepdim: false);

        AddBoolOutput(boolModel, "ne", ne, 2, 2);
        AddBoolOutput(boolModel, "signbit", signbit, 2, 2);
        AddBoolOutput(boolModel, "signbit_bool", signbitBool, 2, 2);
        AddBoolOutput(boolModel, "all", all, 2);
        AddBoolOutput(boolModel, "any", any, 2);

        var boolOutputs = RunModel<bool>(
            boolModel,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("lhs", new DenseTensor<float>(new[] { -1f, 0f, 2f, 3f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("rhs", new DenseTensor<float>(new[] { -1f, 5f, 2f, -3f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("bool_input", new DenseTensor<bool>(new[] { true, false, true, false }, new[] { 2, 2 })),
            });

        AssertTensorValues(boolOutputs["ne"], [false, true, false, true], 2, 2);
        AssertTensorValues(boolOutputs["signbit"], [true, false, false, false], 2, 2);
        AssertTensorValues(boolOutputs["signbit_bool"], [false, false, false, false], 2, 2);
        AssertTensorValues(boolOutputs["all"], [false, true]);
        AssertTensorValues(boolOutputs["any"], [true, true]);

        var castModel = CreateRuntimeModel();
        var castInput = castModel.Graph.AddInput("input", OnnxTensorType.Create<float>([3L]));
        var castOther = castModel.Graph.AddInput("other", OnnxTensorType.Create<long>([3L]));
        var casted = castModel.Graph.ExportTypeAs(castInput, castOther);
        var nonzero = castModel.Graph.ExportNonZero(castInput);

        AddLongOutput(castModel, "type_as", casted, 3);
        AddLongOutput(castModel, "nonzero", nonzero, 2, 1);

        var longOutputs = RunModel<long>(
            castModel,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 0f, 2f, -3f }, new[] { 3 })),
                NamedOnnxValue.CreateFromTensor("other", new DenseTensor<long>(new long[] { 1L, 1L, 1L }, new[] { 3 })),
            });

        AssertTensorValues(longOutputs["type_as"], [0L, 2L, -3L]);
        AssertTensorValues(longOutputs["nonzero"], [1L, 2L], 2, 1);
    }

    [Fact]
    public void Smoke_RuntimeExecutesNewCoverageOperators()
    {
        var floatModel = CreateRuntimeModel();
        var input = floatModel.Graph.AddInput("input", OnnxTensorType.Create<float>([2L, 2L]));
        var other = floatModel.Graph.AddInput("other", OnnxTensorType.Create<float>([2L, 2L]));
        var special = floatModel.Graph.AddInput("special", OnnxTensorType.Create<float>([2L]));

        var amin = floatModel.Graph.ExportAMin(input, new long[] { 1L }, keepdim: false);
        var atan2 = floatModel.Graph.ExportAtan2(input, other);
        var full = floatModel.Graph.ExportFull(new long[] { 2L, 2L }, 3d);
        var fullLike = floatModel.Graph.ExportFullLike(input, 4d);
        var ones = floatModel.Graph.ExportOnes(new long[] { 2L, 2L });
        var onesLike = floatModel.Graph.ExportOnesLike(input);
        var zeros = floatModel.Graph.ExportZeros(new long[] { 2L, 2L });
        var zerosLike = floatModel.Graph.ExportZerosLike(input);
        var scalarPow = floatModel.Graph.ExportPow(2d, special);
        var rounded = floatModel.Graph.ExportRound(special, decimals: 1);
        var sinc = floatModel.Graph.ExportSinc(special);
        var erfcx = floatModel.Graph.ExportErfcx(special);
        var tanh = floatModel.Graph.ExportTanh(special);

        AddFloatOutput(floatModel, "amin", amin, 2);
        AddFloatOutput(floatModel, "atan2", atan2, 2, 2);
        AddFloatOutput(floatModel, "full", full, 2, 2);
        AddFloatOutput(floatModel, "full_like", fullLike, 2, 2);
        AddFloatOutput(floatModel, "ones", ones, 2, 2);
        AddFloatOutput(floatModel, "ones_like", onesLike, 2, 2);
        AddFloatOutput(floatModel, "zeros", zeros, 2, 2);
        AddFloatOutput(floatModel, "zeros_like", zerosLike, 2, 2);
        AddFloatOutput(floatModel, "scalar_pow", scalarPow, 2);
        AddFloatOutput(floatModel, "rounded", rounded, 2);
        AddFloatOutput(floatModel, "sinc", sinc, 2);
        AddFloatOutput(floatModel, "erfcx", erfcx, 2);
        AddFloatOutput(floatModel, "tanh", tanh, 2);

        var floatResults = RunModel<float>(
            floatModel,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("other", new DenseTensor<float>(new[] { 1f, -1f, -2f, 2f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("special", new DenseTensor<float>(new[] { 0f, 0.5f }, new[] { 2 })),
            });

        AssertTensorValues(floatResults["amin"], [1f, 3f]);
        AssertTensorValues(floatResults["atan2"], [MathF.Atan2(1f, 1f), MathF.Atan2(2f, -1f), MathF.Atan2(3f, -2f), MathF.Atan2(4f, 2f)], 2, 2);
        AssertTensorValues(floatResults["full"], [3f, 3f, 3f, 3f], 2, 2);
        AssertTensorValues(floatResults["full_like"], [4f, 4f, 4f, 4f], 2, 2);
        AssertTensorValues(floatResults["ones"], [1f, 1f, 1f, 1f], 2, 2);
        AssertTensorValues(floatResults["ones_like"], [1f, 1f, 1f, 1f], 2, 2);
        AssertTensorValues(floatResults["zeros"], [0f, 0f, 0f, 0f], 2, 2);
        AssertTensorValues(floatResults["zeros_like"], [0f, 0f, 0f, 0f], 2, 2);
        AssertTensorValues(floatResults["scalar_pow"], [1f, MathF.Sqrt(2f)]);
        AssertTensorValues(floatResults["rounded"], [0f, 0.5f]);
        AssertTensorValues(floatResults["sinc"], [1f, 2f / MathF.PI]);
        AssertTensorValues(floatResults["erfcx"], [1f, MathF.Exp(0.25f) * 0.4795001f]);
        AssertTensorValues(floatResults["tanh"], [0f, MathF.Tanh(0.5f)]);

        var boolModel = CreateRuntimeModel();
        var infInput = boolModel.Graph.AddInput("inf_input", OnnxTensorType.Create<float>([3L]));
        var isNegInf = boolModel.Graph.ExportIsNegInf(infInput);
        var isPosInf = boolModel.Graph.ExportIsPosInf(infInput);
        AddBoolOutput(boolModel, "isneginf", isNegInf, 3);
        AddBoolOutput(boolModel, "isposinf", isPosInf, 3);

        var boolResults = RunModel<bool>(
            boolModel,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("inf_input", new DenseTensor<float>(new[] { float.NegativeInfinity, 0f, float.PositiveInfinity }, new[] { 3 })),
            });

        AssertTensorValues(boolResults["isneginf"], [true, false, false]);
        AssertTensorValues(boolResults["isposinf"], [false, false, true]);

        var intModel = CreateRuntimeModel();
        var left = intModel.Graph.AddInput("left", OnnxTensorType.Create<long>([4L]));
        var right = intModel.Graph.AddInput("right", OnnxTensorType.Create<long>([4L]));
        var floorDivInt = intModel.Graph.ExportFloorDivide(left, right);
        AddLongOutput(intModel, "floor_divide", floorDivInt, 4);

        var intOutputs = RunModel<long>(
            intModel,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<long>(new long[] { -5L, 5L, 5L, -5L }, new[] { 4 })),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<long>(new long[] { 2L, -2L, 2L, -2L }, new[] { 4 })),
            });

        AssertTensorValues(intOutputs["floor_divide"], [-3L, -3L, 2L, 2L]);
    }

    [Fact]
    public void ExportArangeLinspaceAndExtrema_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 2L]));

        var max = graph.ExportMax(input);
        var maxDim = graph.ExportMax(input, dim: 1, keepdim: false);
        var min = graph.ExportMin(input);
        var minDim = graph.ExportMin(input, dim: 1, keepdim: true);
        var arange = graph.ExportArange(5L);
        var arangeStart = graph.ExportArange(2L, 7L);
        var arangeStep = graph.ExportArange(10L, 2L, -3L);
        var linspace = graph.ExportLinspace(-1f, 1f, 5L);

        Assert.NotNull(max);
        Assert.NotNull(maxDim.Values);
        Assert.NotNull(maxDim.Indices);
        Assert.NotNull(min);
        Assert.NotNull(minDim.Values);
        Assert.NotNull(minDim.Indices);
        Assert.NotNull(arange);
        Assert.NotNull(arangeStart);
        Assert.NotNull(arangeStep);
        Assert.NotNull(linspace);

        Assert.Equal(
            ["ReduceMax", "TopK", "Squeeze", "Squeeze", "ReduceMin", "TopK", "Range", "Range", "Range", "Range", "Sub", "Div", "Div", "Mul", "Add", "Sub", "Mul", "Sub", "Less", "Where"],
            graph.Nodes.Select(x => x.OpType).ToArray()
        );
    }

    [Fact]
    public void Smoke_RuntimeExecutesArangeLinspaceAndExtrema()
    {
        var reduceValueModel = CreateRuntimeModel();
        var reduceValueInput = reduceValueModel.Graph.AddInput("input", OnnxTensorType.Create<float>([2L, 2L]));

        var max = reduceValueModel.Graph.ExportMax(reduceValueInput);
        var maxDim = reduceValueModel.Graph.ExportMax(reduceValueInput, dim: 1, keepdim: false);
        var min = reduceValueModel.Graph.ExportMin(reduceValueInput);
        var minDim = reduceValueModel.Graph.ExportMin(reduceValueInput, dim: 1, keepdim: false);

        AddFloatOutput(reduceValueModel, "max", max);
        AddFloatOutput(reduceValueModel, "max_values", maxDim.Values!, 2);
        AddFloatOutput(reduceValueModel, "min", min);
        AddFloatOutput(reduceValueModel, "min_values", minDim.Values!, 2);

        var reduceValueResults = RunModel<float>(
            reduceValueModel,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 1f, 4f, 3f, 2f }, new[] { 2, 2 })),
            });

        AssertTensorValues(reduceValueResults["max"], [4f]);
        AssertTensorValues(reduceValueResults["max_values"], [4f, 3f], 2);
        AssertTensorValues(reduceValueResults["min"], [1f]);
        AssertTensorValues(reduceValueResults["min_values"], [1f, 2f], 2);

        var reduceIndexModel = CreateRuntimeModel();
        var reduceIndexInput = reduceIndexModel.Graph.AddInput("input", OnnxTensorType.Create<float>([2L, 2L]));
        var maxIndexDim = reduceIndexModel.Graph.ExportMax(reduceIndexInput, dim: 1, keepdim: false);
        var minIndexDim = reduceIndexModel.Graph.ExportMin(reduceIndexInput, dim: 1, keepdim: false);

        AddLongOutput(reduceIndexModel, "max_indices", maxIndexDim.Indices!, 2);
        AddLongOutput(reduceIndexModel, "min_indices", minIndexDim.Indices!, 2);

        var reduceIndexResults = RunModel<long>(
            reduceIndexModel,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 1f, 4f, 3f, 2f }, new[] { 2, 2 })),
            });

        AssertTensorValues(reduceIndexResults["max_indices"], [1L, 0L], 2);
        AssertTensorValues(reduceIndexResults["min_indices"], [0L, 1L], 2);

        var arangeModel = CreateRuntimeModel();
        AddLongOutput(arangeModel, "arange", arangeModel.Graph.ExportArange(5L), 5);
        AddLongOutput(arangeModel, "arange_start", arangeModel.Graph.ExportArange(2L, 7L), 5);
        AddLongOutput(arangeModel, "arange_step", arangeModel.Graph.ExportArange(10L, 2L, -3L), 3);

        var arangeResults = RunModel<long>(arangeModel, Array.Empty<NamedOnnxValue>());

        AssertTensorValues(arangeResults["arange"], [0L, 1L, 2L, 3L, 4L], 5);
        AssertTensorValues(arangeResults["arange_start"], [2L, 3L, 4L, 5L, 6L], 5);
        AssertTensorValues(arangeResults["arange_step"], [10L, 7L, 4L], 3);

        var linspaceModel = CreateRuntimeModel();
        AddFloatOutput(linspaceModel, "linspace", linspaceModel.Graph.ExportLinspace(-1f, 1f, 5L), 5);

        var linspaceResults = RunModel<float>(linspaceModel, Array.Empty<NamedOnnxValue>());

        AssertTensorValues(linspaceResults["linspace"], [-1f, -0.5f, 0f, 0.5f, 1f], 5);
    }

    [Fact]
    public void ExportCloseAndLogOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 2L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([2L, 2L]));

        var equalAll = graph.ExportEqualAll(input, other);
        var allClose = graph.ExportAllClose(input, other);
        var isClose = graph.ExportIsClose(input, other);
        var isFinite = graph.ExportIsFinite(input);
        var isInf = graph.ExportIsInf(input);
        var isNaN = graph.ExportIsNaN(input);
        var logAddExp = graph.ExportLogAddExp(input, other);
        var logAddExp2 = graph.ExportLogAddExp2(input, other);
        var logit = graph.ExportLogit(input, eps: 1e-3);
        var logSumExp = graph.ExportLogSumExp(input, new long[] { 1L }, keepdim: false);

        Assert.NotNull(equalAll);
        Assert.NotNull(allClose);
        Assert.NotNull(isClose);
        Assert.NotNull(isFinite);
        Assert.NotNull(isInf);
        Assert.NotNull(isNaN);
        Assert.NotNull(logAddExp);
        Assert.NotNull(logAddExp2);
        Assert.NotNull(logit);
        Assert.NotNull(logSumExp);

        var opTypes = graph.Nodes.Select(x => x.OpType).ToArray();
        Assert.Contains("ReduceMin", opTypes);
        Assert.Contains("IsInf", opTypes);
        Assert.Contains("IsNaN", opTypes);
        Assert.Contains("ReduceLogSumExp", opTypes);
        Assert.Contains("Log", opTypes);
        Assert.Contains("Where", opTypes);
    }

    [Fact]
    public void Smoke_RuntimeExecutesCloseAndLogOperators()
    {
        var boolModel = CreateRuntimeModel();
        var input = boolModel.Graph.AddInput("input", OnnxTensorType.Create<float>([2L, 2L]));
        var other = boolModel.Graph.AddInput("other", OnnxTensorType.Create<float>([2L, 2L]));
        var special = boolModel.Graph.AddInput("special", OnnxTensorType.Create<float>([4L]));

        var equalAll = boolModel.Graph.ExportEqualAll(input, other);
        var allClose = boolModel.Graph.ExportAllClose(input, other, rtol: 1e-04, atol: 1e-05);
        var isClose = boolModel.Graph.ExportIsClose(input, other, rtol: 1e-04, atol: 1e-05);
        var isFinite = boolModel.Graph.ExportIsFinite(special);
        var isInf = boolModel.Graph.ExportIsInf(special);
        var isNaN = boolModel.Graph.ExportIsNaN(special);

        AddBoolOutput(boolModel, "equal", equalAll);
        AddBoolOutput(boolModel, "allclose", allClose);
        AddBoolOutput(boolModel, "isclose", isClose, 2, 2);
        AddBoolOutput(boolModel, "isfinite", isFinite, 4);
        AddBoolOutput(boolModel, "isinf", isInf, 4);
        AddBoolOutput(boolModel, "isnan", isNaN, 4);

        var boolResults = RunModel<bool>(
            boolModel,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("other", new DenseTensor<float>(new[] { 1f, 2.00001f, 3f, 4.1f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("special", new DenseTensor<float>(new[] { 0f, float.PositiveInfinity, float.NaN, -2f }, new[] { 4 })),
            });

        AssertTensorValues(boolResults["equal"], [false]);
        AssertTensorValues(boolResults["allclose"], [false]);
        AssertTensorValues(boolResults["isclose"], [true, true, true, false], 2, 2);
        AssertTensorValues(boolResults["isfinite"], [true, false, false, true], 4);
        AssertTensorValues(boolResults["isinf"], [false, true, false, false], 4);
        AssertTensorValues(boolResults["isnan"], [false, false, true, false], 4);

        var floatModel = CreateRuntimeModel();
        var left = floatModel.Graph.AddInput("left", OnnxTensorType.Create<float>([2L, 2L]));
        var right = floatModel.Graph.AddInput("right", OnnxTensorType.Create<float>([2L, 2L]));
        var probability = floatModel.Graph.AddInput("probability", OnnxTensorType.Create<float>([3L]));

        AddFloatOutput(floatModel, "logaddexp", floatModel.Graph.ExportLogAddExp(left, right), 2, 2);
        AddFloatOutput(floatModel, "logaddexp2", floatModel.Graph.ExportLogAddExp2(left, right), 2, 2);
        AddFloatOutput(floatModel, "logit", floatModel.Graph.ExportLogit(probability, eps: 1e-3), 3);
        AddFloatOutput(floatModel, "logsumexp", floatModel.Graph.ExportLogSumExp(left, new long[] { 1L }, keepdim: false), 2);

        var floatResults = RunModel<float>(
            floatModel,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<float>(new[] { 0f, 1f, 2f, 3f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<float>(new[] { 1f, 0f, -1f, 2f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("probability", new DenseTensor<float>(new[] { 0f, 0.25f, 1f }, new[] { 3 })),
            });

        AssertTensorValues(floatResults["logaddexp"], [MathF.Log(MathF.Exp(0f) + MathF.Exp(1f)), MathF.Log(MathF.Exp(1f) + MathF.Exp(0f)), MathF.Log(MathF.Exp(2f) + MathF.Exp(-1f)), MathF.Log(MathF.Exp(3f) + MathF.Exp(2f))], 2, 2);
        AssertTensorValues(floatResults["logaddexp2"], [MathF.Log2(MathF.Pow(2f, 0f) + MathF.Pow(2f, 1f)), MathF.Log2(MathF.Pow(2f, 1f) + MathF.Pow(2f, 0f)), MathF.Log2(MathF.Pow(2f, 2f) + MathF.Pow(2f, -1f)), MathF.Log2(MathF.Pow(2f, 3f) + MathF.Pow(2f, 2f))], 2, 2);
        AssertTensorValues(floatResults["logit"], [MathF.Log(0.001f / 0.999f), MathF.Log(0.25f / 0.75f), MathF.Log(0.999f / 0.001f)], 3);
        AssertTensorValues(floatResults["logsumexp"], [MathF.Log(MathF.Exp(0f) + MathF.Exp(1f)), MathF.Log(MathF.Exp(2f) + MathF.Exp(3f))], 2);
    }

    [Fact]
    public void ExportLinearAlgebraBlendOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var matrixInput = graph.AddInput("matrix_input", OnnxTensorType.Create<float>([2L, 2L]));
        var matrixLeft = graph.AddInput("matrix_left", OnnxTensorType.Create<float>([2L, 2L]));
        var matrixRight = graph.AddInput("matrix_right", OnnxTensorType.Create<float>([2L, 2L]));
        var batchInput = graph.AddInput("batch_input", OnnxTensorType.Create<float>([2L, 2L]));
        var batchLeft = graph.AddInput("batch_left", OnnxTensorType.Create<float>([2L, 2L, 2L]));
        var batchRight = graph.AddInput("batch_right", OnnxTensorType.Create<float>([2L, 2L, 2L]));
        var batchAccumulator = graph.AddInput("batch_accumulator", OnnxTensorType.Create<float>([2L, 2L, 2L]));
        var vectorInput = graph.AddInput("vector_input", OnnxTensorType.Create<float>([2L]));
        var vectorLeft = graph.AddInput("vector_left", OnnxTensorType.Create<float>([2L]));
        var vectorRight = graph.AddInput("vector_right", OnnxTensorType.Create<float>([2L]));
        var interpolationWeight = graph.AddInput("interpolation_weight", OnnxTensorType.Create<float>([2L, 2L]));

        graph.ExportAddBmm(batchInput, batchLeft, batchRight, beta: 0.5f, alpha: 2f);
        graph.ExportAddCDiv(matrixInput, matrixLeft, matrixRight, value: 0.5f);
        graph.ExportAddCMul(matrixInput, matrixLeft, matrixRight, value: 0.5f);
        graph.ExportAddMM(matrixInput, matrixLeft, matrixRight, beta: 2f, alpha: 0.5f);
        graph.ExportAddMV(vectorInput, matrixLeft, vectorLeft, beta: 2f, alpha: 0.5f);
        graph.ExportAddr(matrixInput, vectorLeft, vectorRight, beta: 0f, alpha: 0.5f);
        graph.ExportBAddBmm(batchAccumulator, batchLeft, batchRight, beta: 2f, alpha: 0.5f);
        graph.ExportLerp(matrixInput, matrixRight, interpolationWeight);
        graph.ExportLerp(matrixInput, matrixRight, 0.25f);
        graph.ExportMV(matrixLeft, vectorLeft);

        var opTypes = graph.Nodes.Select(x => x.OpType).ToArray();
        Assert.Equal(1, opTypes.Count(x => x == "Gemm"));
        Assert.Equal(1, opTypes.Count(x => x == "ReduceSum"));
        Assert.Equal(5, opTypes.Count(x => x == "MatMul"));
        Assert.Equal(2, opTypes.Count(x => x == "Reshape"));
        Assert.Equal(2, opTypes.Count(x => x == "Where"));
        Assert.Contains("Div", opTypes);
        Assert.Contains("Less", opTypes);
    }

    [Fact]
    public void Smoke_RuntimeExecutesLinearAlgebraBlendOperators()
    {
        var model = CreateRuntimeModel();
        var matrixInput = model.Graph.AddInput("matrix_input", OnnxTensorType.Create<float>([2L, 2L]));
        var matrixLeft = model.Graph.AddInput("matrix_left", OnnxTensorType.Create<float>([2L, 2L]));
        var matrixRight = model.Graph.AddInput("matrix_right", OnnxTensorType.Create<float>([2L, 2L]));
        var batchInput = model.Graph.AddInput("batch_input", OnnxTensorType.Create<float>([2L, 2L]));
        var batchLeft = model.Graph.AddInput("batch_left", OnnxTensorType.Create<float>([2L, 2L, 2L]));
        var batchRight = model.Graph.AddInput("batch_right", OnnxTensorType.Create<float>([2L, 2L, 2L]));
        var batchAccumulator = model.Graph.AddInput("batch_accumulator", OnnxTensorType.Create<float>([2L, 2L, 2L]));
        var vectorInput = model.Graph.AddInput("vector_input", OnnxTensorType.Create<float>([2L]));
        var vectorLeft = model.Graph.AddInput("vector_left", OnnxTensorType.Create<float>([2L]));
        var vectorRight = model.Graph.AddInput("vector_right", OnnxTensorType.Create<float>([2L]));
        var interpolationWeight = model.Graph.AddInput("interpolation_weight", OnnxTensorType.Create<float>([2L, 2L]));

        AddFloatOutput(model, "addbmm", model.Graph.ExportAddBmm(batchInput, batchLeft, batchRight, beta: 0.5f, alpha: 2f), 2, 2);
        AddFloatOutput(model, "addcdiv", model.Graph.ExportAddCDiv(matrixInput, matrixLeft, matrixRight, value: 0.5f), 2, 2);
        AddFloatOutput(model, "addcmul", model.Graph.ExportAddCMul(matrixInput, matrixLeft, matrixRight, value: 0.5f), 2, 2);
        AddFloatOutput(model, "addmm", model.Graph.ExportAddMM(matrixInput, matrixLeft, matrixRight, beta: 2f, alpha: 0.5f), 2, 2);
        AddFloatOutput(model, "addmv", model.Graph.ExportAddMV(vectorInput, matrixLeft, vectorLeft, beta: 2f, alpha: 0.5f), 2);
        AddFloatOutput(model, "addr", model.Graph.ExportAddr(matrixInput, vectorLeft, vectorRight, beta: 0f, alpha: 0.5f), 2, 2);
        AddFloatOutput(model, "baddbmm", model.Graph.ExportBAddBmm(batchAccumulator, batchLeft, batchRight, beta: 2f, alpha: 0.5f), 2, 2, 2);
        AddFloatOutput(model, "lerp_tensor", model.Graph.ExportLerp(matrixInput, matrixRight, interpolationWeight), 2, 2);
        AddFloatOutput(model, "lerp_scalar", model.Graph.ExportLerp(matrixInput, matrixRight, 0.25f), 2, 2);
        AddFloatOutput(model, "mv", model.Graph.ExportMV(matrixLeft, vectorLeft), 2);

        var results = RunModel<float>(
            model,
            new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("matrix_input", new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("matrix_left", new DenseTensor<float>(new[] { 2f, 4f, 6f, 8f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("matrix_right", new DenseTensor<float>(new[] { 1f, 0.5f, 0.5f, 0.25f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("batch_input", new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 })),
                NamedOnnxValue.CreateFromTensor("batch_left", new DenseTensor<float>(new[] { 1f, 0f, 0f, 1f, 2f, 1f, 1f, 0f }, new[] { 2, 2, 2 })),
                NamedOnnxValue.CreateFromTensor("batch_right", new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f, 0f, 1f, 2f, 3f }, new[] { 2, 2, 2 })),
                NamedOnnxValue.CreateFromTensor("batch_accumulator", new DenseTensor<float>(new[] { 1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f }, new[] { 2, 2, 2 })),
                NamedOnnxValue.CreateFromTensor("vector_input", new DenseTensor<float>(new[] { 1f, 2f }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("vector_left", new DenseTensor<float>(new[] { 1f, 2f }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("vector_right", new DenseTensor<float>(new[] { 3f, 4f }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("interpolation_weight", new DenseTensor<float>(new[] { 0.25f, 0.75f, 0.4f, 0.6f }, new[] { 2, 2 })),
            });

        AssertTensorValues(results["addbmm"], [6.5f, 15f, 7.5f, 12f], 2, 2);
        AssertTensorValues(results["addcdiv"], [2f, 6f, 9f, 20f], 2, 2);
        AssertTensorValues(results["addcmul"], [2f, 3f, 4.5f, 5f], 2, 2);
        AssertTensorValues(results["addmm"], [4f, 5f, 11f, 10.5f], 2, 2);
        AssertTensorValues(results["addmv"], [7f, 15f], 2);
        AssertTensorValues(results["addr"], [1.5f, 2f, 3f, 4f], 2, 2);
        AssertTensorValues(results["baddbmm"], [2.5f, 3f, 3.5f, 4f, 5f, 6.5f, 4f, 4.5f], 2, 2, 2);
        AssertTensorValues(results["lerp_tensor"], [1f, 0.875f, 2f, 1.75f], 2, 2);
        AssertTensorValues(results["lerp_scalar"], [1f, 1.625f, 2.375f, 3.0625f], 2, 2);
        AssertTensorValues(results["mv"], [10f, 22f], 2);
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
    public void ExportNewCreatorAndSpecialOperators_EmitExpectedNodes()
    {
        var graph = CreateGraph();
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2L, 2L]));
        var other = graph.AddInput("other", OnnxTensorType.Create<float>([2L, 2L]));
        var longInput = graph.AddInput("long_input", OnnxTensorType.Create<long>([2L, 2L]));
        var longOther = graph.AddInput("long_other", OnnxTensorType.Create<long>([2L, 2L]));

        graph.ExportAMin(input, new long[] { 1L }, keepdim: false);
        graph.ExportAtan2(input, other);
        graph.ExportFloorDivide(input, other);
        graph.ExportFloorDivide(longInput, longOther);
        graph.ExportFull(new long[] { 2L, 2L }, 3d);
        graph.ExportFullLike(input, 4d);
        graph.ExportOnes(new long[] { 2L, 2L });
        graph.ExportOnesLike(input);
        graph.ExportZeros(new long[] { 2L, 2L });
        graph.ExportZerosLike(input);
        graph.ExportPow(2d, input);
        graph.ExportRound(input, decimals: 2);
        graph.ExportSinc(input);
        graph.ExportErfcx(input);
        graph.ExportIsNegInf(input);
        graph.ExportIsPosInf(input);
        graph.ExportTanh(input);

        var opTypes = graph.Nodes.Select(x => x.OpType).ToArray();
        Assert.Contains("ReduceMin", opTypes);
        Assert.Contains("Atan", opTypes);
        Assert.Contains("Floor", opTypes);
        Assert.Contains("Expand", opTypes);
        Assert.Contains("Pow", opTypes);
        Assert.Contains("Round", opTypes);
        Assert.Contains("IsInf", opTypes);
        Assert.Contains("Tanh", opTypes);
        Assert.Contains("Where", opTypes);
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
        Assert.Contains("aten::amin", coveredOperators);
        Assert.Contains("aten::sort", coveredOperators);
        Assert.Contains("aten::atan2", coveredOperators);
        Assert.Contains("aten::eq", coveredOperators);
        Assert.Contains("aten::floor_divide", coveredOperators);
        Assert.Contains("aten::full", coveredOperators);
        Assert.Contains("aten::full_like", coveredOperators);
        Assert.Contains("aten::ones", coveredOperators);
        Assert.Contains("aten::ones_like", coveredOperators);
        Assert.Contains("aten::pow.Tensor_Scalar", coveredOperators);
        Assert.Contains("aten::pow.Scalar", coveredOperators);
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
        Assert.Contains("aten::equal", coveredOperators);
        Assert.Contains("aten::allclose", coveredOperators);
        Assert.Contains("aten::isclose", coveredOperators);
        Assert.Contains("aten::isfinite", coveredOperators);
        Assert.Contains("aten::isinf", coveredOperators);
        Assert.Contains("aten::isnan", coveredOperators);
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
        Assert.Contains("aten::ne", coveredOperators);
        Assert.Contains("aten::ne.Scalar", coveredOperators);
        Assert.Contains("aten::ne.Tensor", coveredOperators);
        Assert.Contains("aten::alias", coveredOperators);
        Assert.Contains("aten::clone", coveredOperators);
        Assert.Contains("aten::contiguous", coveredOperators);
        Assert.Contains("aten::detach", coveredOperators);
        Assert.Contains("aten::_conj", coveredOperators);
        Assert.Contains("aten::conj", coveredOperators);
        Assert.Contains("aten::resolve_conj", coveredOperators);
        Assert.Contains("aten::resolve_neg", coveredOperators);
        Assert.Contains("aten::arange", coveredOperators);
        Assert.Contains("aten::arange.start", coveredOperators);
        Assert.Contains("aten::arange.start_step", coveredOperators);
        Assert.Contains("aten::broadcast_to", coveredOperators);
        Assert.Contains("aten::view_as", coveredOperators);
        Assert.Contains("aten::type_as", coveredOperators);
        Assert.Contains("aten::deg2rad", coveredOperators);
        Assert.Contains("aten::rad2deg", coveredOperators);
        Assert.Contains("aten::exp2", coveredOperators);
        Assert.Contains("aten::frac", coveredOperators);
        Assert.Contains("aten::log10", coveredOperators);
        Assert.Contains("aten::logaddexp", coveredOperators);
        Assert.Contains("aten::logaddexp2", coveredOperators);
        Assert.Contains("aten::logit", coveredOperators);
        Assert.Contains("aten::logsumexp", coveredOperators);
        Assert.Contains("aten::addbmm", coveredOperators);
        Assert.Contains("aten::addcdiv", coveredOperators);
        Assert.Contains("aten::addcmul", coveredOperators);
        Assert.Contains("aten::addmm", coveredOperators);
        Assert.Contains("aten::addmv", coveredOperators);
        Assert.Contains("aten::addr", coveredOperators);
        Assert.Contains("aten::baddbmm", coveredOperators);
        Assert.Contains("aten::lerp.Tensor", coveredOperators);
        Assert.Contains("aten::lerp.Scalar", coveredOperators);
        Assert.Contains("aten::mv", coveredOperators);
        Assert.Contains("aten::round.decimals", coveredOperators);
        Assert.Contains("aten::sinc", coveredOperators);
        Assert.Contains("aten::special_sinc", coveredOperators);
        Assert.Contains("aten::special_erfcx", coveredOperators);
        Assert.Contains("aten::isneginf", coveredOperators);
        Assert.Contains("aten::isposinf", coveredOperators);
        Assert.Contains("aten::all", coveredOperators);
        Assert.Contains("aten::all.dim", coveredOperators);
        Assert.Contains("aten::all.dims", coveredOperators);
        Assert.Contains("aten::any", coveredOperators);
        Assert.Contains("aten::any.dim", coveredOperators);
        Assert.Contains("aten::any.dims", coveredOperators);
        Assert.Contains("aten::tile", coveredOperators);
        Assert.Contains("aten::prod", coveredOperators);
        Assert.Contains("aten::prod.dim_int", coveredOperators);
        Assert.Contains("aten::max", coveredOperators);
        Assert.Contains("aten::max.dim", coveredOperators);
        Assert.Contains("aten::min", coveredOperators);
        Assert.Contains("aten::min.dim", coveredOperators);
        Assert.Contains("aten::argmin", coveredOperators);
        Assert.Contains("aten::linspace", coveredOperators);
        Assert.Contains("aten::tril", coveredOperators);
        Assert.Contains("aten::zeros", coveredOperators);
        Assert.Contains("aten::zeros_like", coveredOperators);
        Assert.Contains("_operator::abs", coveredOperators);
        Assert.Contains("_operator::add", coveredOperators);
        Assert.Contains("_operator::eq", coveredOperators);
        Assert.Contains("_operator::floordiv", coveredOperators);
        Assert.Contains("_operator::ge", coveredOperators);
        Assert.Contains("_operator::gt", coveredOperators);
        Assert.Contains("_operator::le", coveredOperators);
        Assert.Contains("_operator::lt", coveredOperators);
        Assert.Contains("_operator::mul", coveredOperators);
        Assert.Contains("_operator::ne", coveredOperators);
        Assert.Contains("_operator::neg", coveredOperators);
        Assert.Contains("_operator::pow", coveredOperators);
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
        Assert.Contains("prims::acos", coveredOperators);
        Assert.Contains("prims::acosh", coveredOperators);
        Assert.Contains("prims::add", coveredOperators);
        Assert.Contains("prims::asin", coveredOperators);
        Assert.Contains("prims::asinh", coveredOperators);
        Assert.Contains("prims::atan", coveredOperators);
        Assert.Contains("prims::atanh", coveredOperators);
        Assert.Contains("prims::ceil", coveredOperators);
        Assert.Contains("prims::cos", coveredOperators);
        Assert.Contains("prims::cosh", coveredOperators);
        Assert.Contains("prims::div", coveredOperators);
        Assert.Contains("prims::eq", coveredOperators);
        Assert.Contains("prims::erf", coveredOperators);
        Assert.Contains("prims::exp", coveredOperators);
        Assert.Contains("prims::floor", coveredOperators);
        Assert.Contains("prims::ge", coveredOperators);
        Assert.Contains("prims::gt", coveredOperators);
        Assert.Contains("prims::le", coveredOperators);
        Assert.Contains("prims::lt", coveredOperators);
        Assert.Contains("prims::log", coveredOperators);
        Assert.Contains("prims::mul", coveredOperators);
        Assert.Contains("prims::ne", coveredOperators);
        Assert.Contains("prims::neg", coveredOperators);
        Assert.Contains("prims::pow", coveredOperators);
        Assert.Contains("prims::reshape", coveredOperators);
        Assert.Contains("prims::round", coveredOperators);
        Assert.Contains("prims::sin", coveredOperators);
        Assert.Contains("prims::sinh", coveredOperators);
        Assert.Contains("prims::sqrt", coveredOperators);
        Assert.Contains("prims::squeeze", coveredOperators);
        Assert.Contains("prims::sub", coveredOperators);
        Assert.Contains("prims::sum", coveredOperators);
        Assert.Contains("prims::tan", coveredOperators);
        Assert.Contains("prims::tanh", coveredOperators);
        Assert.Contains("prims::transpose", coveredOperators);
        Assert.Contains("prims::where", coveredOperators);
    }

    private static OnnxModel CreateModel()
    {
        return OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 21,
            ProducerName = "torch-tensor-tests",
        });
    }

    private static OnnxModel CreateRuntimeModel()
    {
        return OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 23,
            ProducerName = "torch-tensor-runtime-tests",
        });
    }

    private static OnnxGraph CreateGraph()
    {
        return CreateModel().Graph;
    }

    private static void AddFloatOutput(OnnxModel model, string name, IOnnxGraphEdge edge, params long[] shape)
    {
        model.Graph.AddOutput(name, OnnxTensorType.Create<float>(shape.Select(static x => (OnnxDimension)x)));
        model.Graph.AddNode(
            name: $"{name}_identity",
            opType: "Identity",
            domain: string.Empty,
            docString: string.Empty,
            inputs: [edge],
            outputs: [model.Graph.GetValue(name)!],
            attributes: []);
    }

    private static void AddBoolOutput(OnnxModel model, string name, IOnnxGraphEdge edge, params long[] shape)
    {
        model.Graph.AddOutput(name, OnnxTensorType.Create<bool>(shape.Select(static x => (OnnxDimension)x)));
        model.Graph.AddNode(
            name: $"{name}_identity",
            opType: "Identity",
            domain: string.Empty,
            docString: string.Empty,
            inputs: [edge],
            outputs: [model.Graph.GetValue(name)!],
            attributes: []);
    }

    private static void AddLongOutput(OnnxModel model, string name, IOnnxGraphEdge edge, params long[] shape)
    {
        model.Graph.AddOutput(name, OnnxTensorType.Create<long>(shape.Select(static x => (OnnxDimension)x)));
        model.Graph.AddNode(
            name: $"{name}_identity",
            opType: "Identity",
            domain: string.Empty,
            docString: string.Empty,
            inputs: [edge],
            outputs: [model.Graph.GetValue(name)!],
            attributes: []);
    }

    private static IReadOnlyDictionary<string, DenseTensor<T>> RunModel<T>(
        OnnxModel model,
        IReadOnlyCollection<NamedOnnxValue> inputs
    )
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            model.Save(path);

            using var session = new InferenceSession(path);
            using var results = session.Run(inputs);

            return results.ToDictionary(
                result => result.Name,
                result =>
                {
                    var tensor = result.AsTensor<T>();
                    return new DenseTensor<T>(tensor.ToArray(), tensor.Dimensions.ToArray());
                },
                StringComparer.Ordinal);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static void AssertTensorValues(DenseTensor<float> actual, IReadOnlyList<float> expected, params int[] shape)
    {
        if (shape.Length > 0)
        {
            Assert.Equal(shape, actual.Dimensions.ToArray());
        }

        Assert.Equal(expected.Count, actual.Length);
        var actualValues = actual.Buffer.ToArray();

        for (var i = 0; i < expected.Count; i++)
        {
            Assert.InRange(actualValues[i], expected[i] - 1e-4f, expected[i] + 1e-4f);
        }
    }

    private static void AssertTensorValues(DenseTensor<bool> actual, IReadOnlyList<bool> expected, params int[] shape)
    {
        if (shape.Length > 0)
        {
            Assert.Equal(shape, actual.Dimensions.ToArray());
        }

        Assert.Equal(expected, actual.Buffer.ToArray());
    }

    private static void AssertTensorValues(DenseTensor<long> actual, IReadOnlyList<long> expected, params int[] shape)
    {
        if (shape.Length > 0)
        {
            Assert.Equal(shape, actual.Dimensions.ToArray());
        }

        Assert.Equal(expected, actual.Buffer.ToArray());
    }
}
