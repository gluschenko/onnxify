using Onnxify;
using Onnxify.Compiler;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Onnxify.Compiler.Tests;

public sealed class ActivationCompilerTests
{
    public static IEnumerable<object?[]> Activations =>
    [
        Case("Celu", "torch.nn.functional.celu(input, 1.3f)", "torch.nn.functional.celu", 1.3f),
        Case("Elu", "torch.nn.functional.elu(input, 0.8f)", "torch.nn.functional.elu", 0.8f),
        Case("Gelu", "torch.nn.functional.gelu(input)", "Approximate.tanh", "tanh"),
        Case("HardSigmoid", "torch.nn.functional.hardsigmoid(input)", ".clamp(0.0f, 1.0f)"),
        Case("HardSwish", "torch.nn.functional.hardswish(input)", ".clamp(0.0f, 6.0f)"),
        Case("LeakyRelu", "torch.nn.functional.leaky_relu(input, 0.2f)", "negative_slope:", 0.2f),
        Case("Mish", "torch.nn.functional.mish(input)", ".softplus().tanh()"),
        Case("PRelu", "torch.nn.functional.prelu(input, slope)", "torch.nn.functional.prelu"),
        Case("Selu", "torch.nn.functional.selu(input)", "torch.nn.functional.selu"),
        Case("Sigmoid", "input.sigmoid()", ".sigmoid()"),
        Case("Softplus", "input.softplus()", ".softplus()"),
        Case("Softsign", "torch.nn.functional.softsign(input)", ".abs()"),
        Case("Swish", "torch.nn.functional.silu(input)", ".sigmoid()"),
        Case("Tanh", "input.tanh()", ".tanh()"),
        Case("ThresholdedRelu", "torch.nn.functional.threshold(input, 1.0f, 0.0f)", "torch.nn.functional.threshold", 1f),
    ];

    [Theory]
    [MemberData(nameof(Activations))]
    public void Activation_mapping_is_bidirectional_and_matches_onnxruntime(
        string onnxName,
        string torchCall,
        string expectedCSharp,
        object? firstAttribute,
        object? secondAttribute
    )
    {
        // Torch source provenance: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (the activation TorchLibOpInfo entry).
        // Lowering provenance: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/nn.py (the corresponding aten activation converter).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/activation/activation_op_test.cc.
        var model = CreateModel(onnxName, firstAttribute, secondAttribute);
        var onnxTree = Compiler.CreateTreeFromOnnx(model);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));

        var onnxComputationTree = Assert.IsType<CompilerComputationTree>(onnxTree.Value);
        var onnxOperation = Assert.IsType<CompilerOperation>(onnxComputationTree.Operations.Single());
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxOperation.Descriptor.Capability);
        Assert.Equal(onnxName, onnxOperation.Descriptor.Name);

        var generatedCSharp = Compiler.GenerateCSharp(onnxComputationTree);
        Assert.True(generatedCSharp.IsSuccess, FormatDiagnostics(generatedCSharp.Diagnostics));
        Assert.Contains(expectedCSharp, generatedCSharp.Value);

        var source = $"return {torchCall};";
        if (onnxName == "PRelu")
        {
            source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor slope) { "
                + source
                + " }";
        }

        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var torchComputationTree = Assert.IsType<CompilerComputationTree>(torchTree.Value);
        var torchOperation = Assert.IsType<CompilerOperation>(torchComputationTree.Operations.Single());
        Assert.Equal(onnxOperation.Descriptor, torchOperation.Descriptor);

        var emitted = Compiler.GenerateOnnx(torchComputationTree);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        var emittedModel = Assert.IsType<OnnxModel>(emitted.Value);
        var emittedOperations = emittedModel.Graph.Nodes.Select(node => node.OpType).ToArray();
        if (onnxName == "Swish")
        {
            Assert.Equal(["Sigmoid", "Mul"], emittedOperations);
        }
        else
        {
            Assert.Equal([onnxName], emittedOperations);
        }

        var inputValues = new[] { -2.25f, -0.5f, 0.25f, 2.5f };
        var slopeValues = new[] { 0.2f };
        using var input = global::TorchSharp.torch.tensor(
            inputValues,
            [4L],
            dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var slope = global::TorchSharp.torch.tensor(
            slopeValues,
            [1L],
            dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var eager = EvaluateTorch(onnxName, input, slope, firstAttribute, secondAttribute);
        var expected = eager.data<float>().ToArray();
        using var sourceEager = EvaluateTorch(
            onnxName,
            input,
            slope,
            onnxName == "Gelu" ? null : firstAttribute,
            onnxName == "Gelu" ? null : secondAttribute);
        var sourceExpected = sourceEager.data<float>().ToArray();

        var onnxRoundtripModel = Assert.IsType<OnnxModel>(Compiler.GenerateOnnx(onnxComputationTree).Value);
        AssertRuntimeParity(onnxRoundtripModel, inputValues, slopeValues, onnxName, expected);
        AssertRuntimeParity(emittedModel, inputValues, slopeValues, onnxName, sourceExpected);
    }

    [Fact]
    public void Unsupported_activation_attributes_remain_generic_with_diagnostics()
    {
        var model = CreateModel("Sigmoid", 0.25f, null);

        var imported = Compiler.CreateTreeFromOnnx(model);

        Assert.True(imported.IsSuccess);
        var computationTree = Assert.IsType<CompilerComputationTree>(imported.Value);
        var operation = Assert.IsType<CompilerOperation>(computationTree.Operations.Single());
        Assert.Equal(CompilerOperationCapability.Unsupported, operation.Descriptor.Capability);
        Assert.Contains(imported.Diagnostics, diagnostic => diagnostic.Code == CompilerDiagnosticCodes.Unsupported);
    }

    private static OnnxModel CreateModel(string opType, object? firstAttribute, object? secondAttribute)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([4]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([4]));
        var inputs = new List<IOnnxGraphEdge> { input };
        if (opType == "PRelu")
        {
            var slope = model.Graph.AddInput("slope", OnnxTensorType.Create<float>([1]));
            inputs.Add(slope);
        }

        var attributes = new List<OnnxAttribute>();
        AddFloatAttribute(attributes, "alpha", firstAttribute);
        AddFloatAttribute(attributes, "beta", secondAttribute);
        if (opType == "Gelu" && firstAttribute is string approximate)
        {
            attributes.Add(new OnnxAttribute<string>("approximate", approximate));
        }

        model.Graph.AddNode(
            name: "activation",
            opType: opType,
            domain: string.Empty,
            docString: string.Empty,
            inputs: inputs,
            outputs: [output],
            attributes: attributes);
        return model;
    }

    private static void AddFloatAttribute(List<OnnxAttribute> attributes, string name, object? value)
    {
        if (value is float floatValue)
        {
            attributes.Add(new OnnxAttribute<float>(name, floatValue));
        }
    }

    private static global::TorchSharp.torch.Tensor EvaluateTorch(
        string name,
        global::TorchSharp.torch.Tensor input,
        global::TorchSharp.torch.Tensor slope,
        object? firstAttribute,
        object? secondAttribute
    )
    {
        var alpha = firstAttribute is float alphaValue ? alphaValue : name switch
        {
            "HardSigmoid" => 0.2f,
            "LeakyRelu" => 0.01f,
            "Selu" => 1.6732632f,
            "Swish" => 1f,
            "ThresholdedRelu" => 1f,
            _ => 1f,
        };
        var beta = secondAttribute is float betaValue ? betaValue : 0.5f;
        return name switch
        {
            "Celu" => global::TorchSharp.torch.nn.functional.celu(input, alpha: alpha),
            "Elu" => global::TorchSharp.torch.nn.functional.elu(input, alpha: alpha),
            "Gelu" => global::TorchSharp.torch.nn.functional.gelu(
                input,
                approximate: (string?)firstAttribute == "tanh"
                    ? global::TorchSharp.Modules.GELU.Approximate.tanh
                    : global::TorchSharp.Modules.GELU.Approximate.none),
            "HardSigmoid" => (input * alpha + beta).clamp(0f, 1f),
            "HardSwish" => input * (input + 3f).clamp(0f, 6f) / 6f,
            "LeakyRelu" => global::TorchSharp.torch.nn.functional.leaky_relu(input, negative_slope: alpha),
            "Mish" => input * input.softplus().tanh(),
            "PRelu" => global::TorchSharp.torch.nn.functional.prelu(input, slope),
            "Selu" => global::TorchSharp.torch.nn.functional.selu(input),
            "Sigmoid" => input.sigmoid(),
            "Softplus" => input.softplus(),
            "Softsign" => input / (1f + input.abs()),
            "Swish" => input * input.sigmoid(),
            "Tanh" => input.tanh(),
            "ThresholdedRelu" => global::TorchSharp.torch.nn.functional.threshold(input, alpha, 0f),
            _ => throw new NotSupportedException($"No TorchSharp reference for '{name}'."),
        };
    }

    private static void AssertRuntimeParity(
        OnnxModel model,
        float[] inputValues,
        float[] slopeValues,
        string opType,
        float[] expected
    )
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-activation-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(inputValues, [4])),
            };
            if (opType == "PRelu")
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor("slope", new DenseTensor<float>(slopeValues, [1])));
            }

            using var results = session.Run(inputs);
            var actual = results.Single().AsTensor<float>().ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < expected.Length; index++)
            {
                Assert.InRange(Math.Abs(expected[index] - actual[index]), 0f, 2e-5f);
            }
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics) =>
        string.Join(Environment.NewLine, diagnostics.Select(diagnostic => $"{diagnostic.Code}: {diagnostic.Message}"));

    private static object?[] Case(
        string onnxName,
        string torchCall,
        string expectedCSharp,
        object? firstAttribute = null,
        object? secondAttribute = null
    ) => [onnxName, torchCall, expectedCSharp, firstAttribute, secondAttribute];
}
