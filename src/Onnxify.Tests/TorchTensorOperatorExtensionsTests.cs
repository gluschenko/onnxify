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

    private static OnnxGraph CreateGraph()
    {
        return OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 21,
            ProducerName = "torch-tensor-tests",
        }).Graph;
    }
}
