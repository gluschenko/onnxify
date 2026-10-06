using Onnxify;
using Onnxify.Compiler;
using Microsoft.ML.OnnxRuntime;
using CompilerBFloat16 = Onnxify.Data.Numerics.BFloat16;

namespace Onnxify.Compiler.Tests;

public sealed class OnnxCompilerTests
{
    [Fact]
    public void MinimalGraphImportsAndEmitsWithOrderedWiring()
    {
        var model = OnnxModel.Create();
        model.ProducerVersion = "test-version";
        model.Document = "model-document";
        model.Domain = "test-domain";
        model.Graph.Name = "minimal";
        model.Graph.Document = "graph-document";
        model.Graph.AddMetadataProps("purpose", "compiler-test");

        var input = model.Graph.AddInput(
            "input",
            OnnxTensorType.Create<float>([new OnnxDimension<long>(1), new OnnxDimension<string>("features")]));
        var output = model.Graph.AddOutput(
            "output",
            OnnxTensorType.Create<float>([new OnnxDimension<long>(1), new OnnxDimension<string>("features")]));
        model.Graph.AddNode(
            "identity",
            "Identity",
            string.Empty,
            string.Empty,
            [input],
            [output],
            []);

        var imported = Compiler.CreateTreeFromOnnx(model, "memory.onnx");

        Assert.True(imported.IsSuccess);
        var tree = imported.Value!;
        Assert.Equal("minimal", tree.Name);
        Assert.Equal("graph-document", tree.Document);
        Assert.Equal("test-domain", tree.ModelEnvelope!.Domain);
        Assert.Equal(2, tree.Inputs[0].Type is CompilerTensorType tensor
            ? tensor.Dimensions!.Count
            : -1);
        Assert.Single(tree.Operations);
        Assert.Equal("identity", tree.Operations[0].Name);
        Assert.Equal("Identity", ((CompilerOperation)tree.Operations[0]).Descriptor.Name);

        var emitted = Compiler.GenerateOnnx(imported.Value!);

        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        Assert.Equal("minimal", emitted.Value!.Graph.Name);
        Assert.Equal("graph-document", emitted.Value.Graph.Document);
        Assert.Equal("identity", emitted.Value.Graph.Nodes[0].Name);
        Assert.Equal("Identity", emitted.Value.Graph.Nodes[0].OpType);
        Assert.Equal("input", emitted.Value.Graph.Nodes[0].Inputs[0].Name);
        Assert.Equal("output", emitted.Value.Graph.Nodes[0].Outputs[0].Name);
    }

    [Fact]
    public void UnknownOperatorIsPreservedWithWarningDiagnostic()
    {
        var model = OnnxModel.Create();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1]));
        model.Graph.AddNode(
            "custom",
            "CustomOp",
            "com.example",
            string.Empty,
            [input],
            [output],
            []);

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.True(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Equal(CompilerDiagnosticSeverity.Warning, diagnostic.Severity);
        Assert.Equal(CompilerDiagnosticStage.Analyze, diagnostic.Stage);
        Assert.Equal(CompilerSourceSpanKind.Onnx, diagnostic.Span!.Kind);
        Assert.Equal("<graph>", diagnostic.Context!.Caller);
        Assert.Equal("CustomOp", diagnostic.Context.Callee);
        Assert.Equal(CompilerOperationCapability.Unsupported, ((CompilerOperation)result.Value!.Operations[0]).Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(result.Value!);
        Assert.True(emitted.IsSuccess);
        Assert.Equal("CustomOp", emitted.Value!.Graph.Nodes.Single().OpType);
        Assert.Contains(emitted.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Severity == CompilerDiagnosticSeverity.Warning);
    }

    [Fact]
    public void ReluUsesSharedMappingAndMatchesTorchSharpRuntime()
    {
        // Source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (nn.functional.relu).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/activation/activation_op_test.cc (Relu).
        var model = OnnxModel.Create();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([4]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([4]));
        model.Graph.AddNode("relu", "Relu", string.Empty, string.Empty, [input], [output], []);

        var imported = Compiler.CreateTreeFromOnnx(model);

        Assert.True(imported.IsSuccess, string.Join(" | ", imported.Diagnostics.Select(x => x.Message)));
        var operation = Assert.IsType<CompilerOperation>(imported.Value!.Operations.Single());
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);
        Assert.Equal("Relu", operation.Descriptor.Name);
        Assert.DoesNotContain(imported.Diagnostics, diagnostic => diagnostic.Code == CompilerDiagnosticCodes.Unsupported);

        var csharp = Compiler.GenerateCSharp(imported.Value);
        Assert.True(csharp.IsSuccess, string.Join(" | ", csharp.Diagnostics.Select(x => x.Message)));
        Assert.Contains("torch.nn.functional.relu(input)", csharp.Value);

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        Assert.Equal("Relu", emitted.Value!.Graph.Nodes.Single().OpType);

        var path = Path.Combine(Path.GetTempPath(), $"onnxify-compiler-relu-{Guid.NewGuid():N}.onnx");
        var inputValues = new[] { -2f, -0.5f, 0f, 3f };
        try
        {
            emitted.Value.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                global::Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
                    "input",
                    new global::Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(inputValues, [4])),
            ]);
            var runtimeValues = results.Single().AsTensor<float>().ToArray();

            using var torchInput = global::TorchSharp.torch.tensor(
                inputValues,
                [4L],
                dtype: global::TorchSharp.torch.ScalarType.Float32);
            using var torchOutput = global::TorchSharp.torch.nn.functional.relu(torchInput);
            Assert.Equal(torchOutput.data<float>().ToArray(), runtimeValues);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public async Task FileAndStreamEntryPointsImportTheSameTree()
    {
        var model = OnnxModel.Create();
        model.Graph.AddInput("input", OnnxTensorType.Create<float>([1]));
        model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1]));
        model.Graph.AddNode(
            "identity",
            "Identity",
            string.Empty,
            string.Empty,
            [new OnnxEdge("input")],
            [new OnnxEdge("output")],
            []);

        var path = Path.Combine(Path.GetTempPath(), $"onnxify-compiler-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            var fromFile = Compiler.CreateTreeFromOnnx(path);
            await using var stream = File.OpenRead(path);
            var fromStream = Compiler.CreateTreeFromOnnx(stream, document: Path.GetFullPath(path));
            var fromAsync = await Compiler.CreateTreeFromOnnxAsync(path);

            Assert.True(fromFile.IsSuccess);
            Assert.True(fromStream.IsSuccess);
            Assert.True(fromAsync.IsSuccess);
            Assert.Equal(fromFile.Value!.Name, fromStream.Value!.Name);
            Assert.Equal(fromFile.Value.Operations, fromStream.Value.Operations);
            Assert.Equal(fromFile.Value.Operations, fromAsync.Value!.Operations);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public void InitializersAndTypedAttributesRoundTripStructurally()
    {
        var model = OnnxModel.Create();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 2]));
        var weights = model.Graph.AddTensor("weights", [1L, 2L], [2f, 3f]);
        var attributeTensorModel = OnnxModel.Create();
        var attributeTensor = attributeTensorModel.Graph.AddTensor("attribute", [1L], [7L]);

        model.Graph.AddNode(
            "add",
            "Add",
            string.Empty,
            string.Empty,
            [input, weights],
            [output],
            [
                new OnnxAttribute<float>("alpha", 1.5f),
                new OnnxAttribute<float[]>("scales", [1f, 2f]),
                new OnnxAttribute<OnnxTensor>("tensor", attributeTensor),
            ]);

        var imported = Compiler.CreateTreeFromOnnx(model);

        Assert.True(imported.IsSuccess);
        Assert.Single(imported.Value!.Initializers);
        var operation = Assert.IsType<CompilerOperation>(imported.Value.Operations.Single());
        Assert.Equal(["alpha", "scales", "tensor"], operation.Attributes.Select(x => x.Name));
        var emitted = Compiler.GenerateOnnx(imported.Value);

        Assert.True(emitted.IsSuccess);
        Assert.Equal("weights", emitted.Value!.Graph.Initializers.Single().Name);
        Assert.Equal(["alpha", "scales", "tensor"], emitted.Value.Graph.Nodes.Single().Attributes.Select(x => x.Name));
    }

    [Fact]
    public void SparseInitializerPreservesShapeValuesAndIndices()
    {
        var model = OnnxModel.Create();
        model.Graph.AddSparseTensor(
            "sparse",
            [4L, 4L],
            [2L],
            [1f, 2f],
            [2L, 2L],
            [0L, 1L, 2L, 3L]);

        var imported = Compiler.CreateTreeFromOnnx(model);

        Assert.True(imported.IsSuccess);
        var sparse = Assert.IsType<CompilerSparseTensorLiteral>(imported.Value!.Initializers.Single().Value);
        Assert.Equal(2, sparse.Dimensions.Count);
        Assert.Equal(2, sparse.Values.Values.Count);
        Assert.Equal(4, sparse.Indices.Values.Count);

        var emitted = Compiler.GenerateOnnx(imported.Value);

        Assert.True(emitted.IsSuccess);
        Assert.Equal([4L, 4L], emitted.Value!.Graph.SparseInitializers.Single().Shape);
        Assert.Equal([0L, 1L, 2L, 3L], emitted.Value.Graph.SparseInitializers.Single().Indices.Values.Select(Convert.ToInt64));
    }

    [Fact]
    public void NestedGraphAttributePreservesOuterScopeCapture()
    {
        var nestedModel = OnnxModel.Create();
        var nestedOutput = nestedModel.Graph.AddOutput("nested_output", OnnxTensorType.Create<float>([1]));
        nestedModel.Graph.AddNode(
            "nested_identity",
            "Identity",
            string.Empty,
            string.Empty,
            [new OnnxEdge("outer")],
            [nestedOutput],
            []);

        var model = OnnxModel.Create();
        var outer = model.Graph.AddInput("outer", OnnxTensorType.Create<float>([1]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1]));
        model.Graph.AddNode(
            "control",
            "If",
            string.Empty,
            string.Empty,
            [outer],
            [output],
            [new OnnxAttribute<OnnxGraph>("then_branch", nestedModel.Graph)]);

        var imported = Compiler.CreateTreeFromOnnx(model);

        Assert.True(imported.IsSuccess);
        var graphLiteral = Assert.IsType<CompilerGraphLiteral>(
            Assert.IsType<CompilerOperation>(imported.Value!.Operations.Single()).Attributes.Single().Value);
        Assert.Single(graphLiteral.Graph.Captures);
        Assert.Equal("outer", graphLiteral.Graph.Captures[0].Name);

        var emitted = Compiler.GenerateOnnx(imported.Value);

        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        var nested = Assert.IsType<OnnxGraph>(emitted.Value!.Graph.Nodes.Single().Attributes.Single().GetValue());
        Assert.Equal("outer", nested.Nodes.Single().Inputs.Single().Name);
    }

    [Fact]
    public void CompactNumericTensorPayloadPreservesEncodedValues()
    {
        var model = OnnxModel.Create();
        var source = model.Graph.AddTensor(
            "bfloat",
            [2L],
            [CompilerBFloat16.FromEncoded(0x3F80), CompilerBFloat16.FromEncoded(0x4000)]);

        var imported = Compiler.CreateTreeFromOnnx(model);
        var emitted = Compiler.GenerateOnnx(imported.Value!);

        Assert.True(imported.IsSuccess);
        Assert.True(emitted.IsSuccess);
        var values = Assert.IsType<OnnxTensor<CompilerBFloat16>>(emitted.Value!.Graph.Initializers.Single()).Value.ToArray();
        Assert.Equal((ushort)0x3F80, values[0].Value);
        Assert.Equal((ushort)0x4000, values[1].Value);
        _ = source;
    }

    [Fact]
    public void EmittedIdentityModelCanCreateAnOnnxRuntimeSession()
    {
        var model = OnnxModel.Create();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1]));
        model.Graph.AddNode("identity", "Identity", string.Empty, string.Empty, [input], [output], []);

        var tree = Compiler.CreateTreeFromOnnx(model);
        var emitted = Compiler.GenerateOnnx(tree.Value!);
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-compiler-runtime-{Guid.NewGuid():N}.onnx");
        try
        {
            emitted.Value!.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            Assert.Single(session.InputMetadata);
            Assert.Single(session.OutputMetadata);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }
}
