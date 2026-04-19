using Onnxify.TorchSharp;
using System.Reflection;
using System.Runtime.CompilerServices;
using TorchModule = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>;

namespace Onnxify.Tests;

public sealed class TorchModuleExtensionsTests
{
    [Fact]
    public void Export_ForGlu_EmitsSplitSigmoidAndMul()
    {
        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2, 4, 6]));
        var module = CreateModule<global::TorchSharp.Modules.GLU>(("dim", "_dim", 1L));

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        Assert.Collection(
            graph.Nodes,
            split =>
            {
                Assert.Equal("Split", split.OpType);
                Assert.Equal(1L, Convert.ToInt64(split.Attributes.Single(x => x.Name == "axis").GetValue()));
                Assert.Equal(2L, Convert.ToInt64(split.Attributes.Single(x => x.Name == "num_outputs").GetValue()));
            },
            sigmoidNode => Assert.Equal("Sigmoid", sigmoidNode.OpType),
            mulNode => Assert.Equal("Mul", mulNode.OpType)
        );
    }

    [Fact]
    public void Export_ForGroupNorm_EmitsGroupNormalizationNode()
    {
        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>([2, 4, 8, 8]));
        var module = CreateModule<global::TorchSharp.Modules.GroupNorm>(
            ("num_groups", "_num_groups", 2L),
            ("eps", "_eps", 1e-5)
        );

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("GroupNormalization", node.OpType);
        Assert.Equal(2L, Convert.ToInt64(node.Attributes.Single(x => x.Name == "num_groups").GetValue()));
        Assert.Equal(1e-05f, Convert.ToSingle(node.Attributes.Single(x => x.Name == "epsilon").GetValue()));

        Assert.Collection(
            graph.Initializers,
            scale =>
            {
                var tensor = Assert.IsType<OnnxTensor<float>>(scale);
                Assert.Equal([4L], tensor.Shape);
            },
            bias =>
            {
                var tensor = Assert.IsType<OnnxTensor<float>>(bias);
                Assert.Equal([4L], tensor.Shape);
            }
        );
    }

    [Fact]
    public void Export_ForAdditionalPadModules_EmitsExpectedPadModesAndVectors()
    {
        AssertPadExport(
            module: CreateModule<global::TorchSharp.Modules.ReflectionPad3d>(("padding", "_padding", new long[] { 1L, 2L, 3L, 4L, 5L, 6L })),
            inputShape: [1L, 2L, 7L, 8L, 9L],
            expectedMode: "reflect",
            expectedPads: [0L, 0L, 5L, 3L, 1L, 0L, 0L, 6L, 4L, 2L]
        );

        AssertPadExport(
            module: CreateModule<global::TorchSharp.Modules.ReplicationPad1d>(("padding", "_padding", new long[] { 2L })),
            inputShape: [1L, 3L, 9L],
            expectedMode: "edge",
            expectedPads: [0L, 0L, 2L, 0L, 0L, 2L]
        );

        AssertPadExport(
            module: CreateModule<global::TorchSharp.Modules.ReplicationPad2d>(("padding", "_padding", new long[] { 1L, 2L, 3L, 4L })),
            inputShape: [1L, 3L, 7L, 8L],
            expectedMode: "edge",
            expectedPads: [0L, 0L, 3L, 1L, 0L, 0L, 4L, 2L]
        );

        AssertPadExport(
            module: CreateModule<global::TorchSharp.Modules.ReplicationPad3d>(("padding", "_padding", new long[] { 1L, 2L, 3L, 4L, 5L, 6L })),
            inputShape: [1L, 2L, 7L, 8L, 9L],
            expectedMode: "edge",
            expectedPads: [0L, 0L, 5L, 3L, 1L, 0L, 0L, 6L, 4L, 2L]
        );
    }

    private static void AssertPadExport(
        TorchModule module,
        long[] inputShape,
        string expectedMode,
        long[] expectedPads
    )
    {
        var graph = CreateGraph(opset: 21);
        var input = graph.AddInput("input", OnnxTensorType.Create<float>(inputShape.Select(static x => (OnnxDimension)x)));

        var output = module.Export(graph, input);

        Assert.NotNull(output);
        var node = Assert.Single(graph.Nodes);
        Assert.Equal("Pad", node.OpType);
        Assert.Equal(expectedMode, Assert.IsType<string>(node.Attributes.Single(x => x.Name == "mode").GetValue()));

        var padsTensor = Assert.Single(graph.Initializers);
        var typedTensor = Assert.IsType<OnnxTensor<long>>(padsTensor);
        Assert.Equal(expectedPads, typedTensor.Value.ToArray());
    }

    private static OnnxGraph CreateGraph(int opset)
    {
        return OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = opset,
            ProducerName = "torch-module-tests",
        }).Graph;
    }

    private static TModule CreateModule<TModule>(params (string Name, string BackingName, object Value)[] assignments)
        where TModule : TorchModule
    {
        var module = (TModule)RuntimeHelpers.GetUninitializedObject(typeof(TModule));

        foreach (var (name, backingName, value) in assignments)
        {
            SetMember(module, value, name, backingName);
        }

        return module;
    }

    private static void SetMember(object instance, object value, params string[] candidateNames)
    {
        const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        foreach (var name in candidateNames)
        {
            var property = instance.GetType().GetProperty(name, Flags);
            if (property is not null && property.SetMethod is not null)
            {
                property.SetValue(instance, value);
                return;
            }

            var field = instance.GetType().GetField(name, Flags);
            if (field is not null)
            {
                field.SetValue(instance, value);
                return;
            }
        }

        throw new InvalidOperationException(
            $"Could not find any writable member named '{string.Join("' or '", candidateNames)}' on '{instance.GetType().FullName}'."
        );
    }
}
