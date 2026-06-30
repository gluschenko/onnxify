using Onnxify.TorchSharp;
using static TorchSharp.torch;
using TorchModule = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>;
using TorchTensor = TorchSharp.torch.Tensor;

namespace Onnxify.Tests;

public sealed class TorchExportEvaluationTests
{
    [Fact]
    public void Evaluate_ForMatchingIdentityModel_PassesWithLowLoss()
    {
        using var module = new IdentityModule();
        module.eval();

        using var input = tensor(
            new[] { 1f, -2f, 3.5f, 4f },
            [2L, 2L],
            dtype: ScalarType.Float32
        );
        var model = CreateIdentityModel();

        var result = module.Evaluate(model, new[] { input });

        Assert.True(result.Passed);
        Assert.Empty(result.Diagnostics);
        var sample = Assert.Single(result.Samples);
        var output = Assert.Single(sample.Outputs);
        Assert.True(output.Loss <= 1e-10d);
        Assert.Equal([2L, 2L], output.ActualShape);
    }

    [Fact]
    public void Evaluate_ForOneElementInputCollection_UsesCollectionApi()
    {
        using var module = new IdentityModule();
        module.eval();
        using var input = tensor(new[] { 2f, 3f }, [2L], dtype: ScalarType.Float32);

        var result = module.Evaluate(
            CreateIdentityModel(shape: [2L]),
            new List<TorchTensor> { input }
        );

        Assert.True(result.Passed);
        Assert.Single(result.Samples);
    }

    [Fact]
    public void Evaluate_ForMultipleSamples_AggregatesResults()
    {
        using var module = new IdentityModule();
        module.eval();
        using var first = tensor(new[] { 1f, 2f }, [2L], dtype: ScalarType.Float32);
        using var second = tensor(new[] { 3f, 4f }, [2L], dtype: ScalarType.Float32);

        var result = module.Evaluate(
            CreateIdentityModel(shape: [2L]),
            new[] { first, second }
        );

        Assert.True(result.Passed);
        Assert.Equal(2, result.Samples.Count);
    }

    [Fact]
    public void Evaluate_ForOutputMismatch_ReturnsDiagnostics()
    {
        using var module = new AddOneModule();
        module.eval();
        using var input = tensor(new[] { 1f, 2f }, [2L], dtype: ScalarType.Float32);

        var result = module.Evaluate(
            CreateIdentityModel(shape: [2L]),
            new[] { input }
        );

        Assert.False(result.Passed);
        Assert.Contains(result.Diagnostics, diagnostic => diagnostic.Code == "OutputComparisonFailed");
        Assert.False(Assert.Single(Assert.Single(result.Samples).Outputs).Passed);
    }

    [Fact]
    public void Evaluate_UsesInputNameMappingForDictionarySamples()
    {
        using var module = new IdentityModule();
        module.eval();
        using var input = tensor(new[] { 1f, 2f }, [2L], dtype: ScalarType.Float32);

        var result = module.Evaluate(
            CreateIdentityModel(shape: [2L]),
            new[]
            {
                new Dictionary<string, TorchTensor>(StringComparer.Ordinal)
                {
                    ["sample"] = input,
                },
            },
            new TorchExportEvaluationOptions
            {
                InputNameMapping = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["sample"] = "input",
                },
            }
        );

        Assert.True(result.Passed);
    }

    [Fact]
    public void Evaluate_UsesOutputNameMappingForDictionaryOutputs()
    {
        using var module = new DictionaryOutputModule();
        module.eval();
        using var input = tensor(new[] { 1f, 2f }, [2L], dtype: ScalarType.Float32);

        var result = module.Evaluate(
            CreateIdentityModel(shape: [2L]),
            new[] { input },
            new TorchExportEvaluationOptions
            {
                OutputNameMapping = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["sample"] = "output",
                },
            }
        );

        Assert.True(result.Passed);
    }

    [Fact]
    public void Evaluate_UsesToleranceConfiguration()
    {
        using var module = new AddOneModule();
        module.eval();
        using var input = tensor(new[] { 1f, 2f }, [2L], dtype: ScalarType.Float32);

        var result = module.Evaluate(
            CreateAddConstantModel(1.00001f, [2L]),
            new[] { input },
            new TorchExportEvaluationOptions
            {
                AbsoluteTolerance = 1e-3d,
                RelativeTolerance = 0d,
            }
        );

        Assert.True(result.Passed);
    }

    [Fact]
    public void Evaluate_InvokesCustomComparer()
    {
        using var module = new AddOneModule();
        module.eval();
        using var input = tensor(new[] { 1f, 2f }, [2L], dtype: ScalarType.Float32);
        var invoked = false;

        var result = module.Evaluate(
            CreateIdentityModel(shape: [2L]),
            new[] { input },
            new TorchExportEvaluationOptions
            {
                Comparer = context =>
                {
                    invoked = true;
                    return new EvaluationOutputResult(
                        outputName: context.OutputName,
                        loss: 42d,
                        meanAbsoluteError: 0d,
                        maxAbsoluteError: 0d,
                        elementCount: context.Expected.Values.Count,
                        expectedShape: context.Expected.Shape,
                        actualShape: context.Actual.Shape,
                        passed: true,
                        message: null
                    );
                },
            }
        );

        Assert.True(invoked);
        Assert.True(result.Passed);
        Assert.Equal(42d, Assert.Single(Assert.Single(result.Samples).Outputs).Loss);
    }

    [Fact]
    public void Evaluate_ForUnsupportedOutputDtype_ReturnsDiagnostic()
    {
        using var module = new BoolOutputModule();
        module.eval();
        using var input = tensor(new[] { 1f, 2f }, [2L], dtype: ScalarType.Float32);

        var result = module.Evaluate(
            CreateIdentityModel(shape: [2L]),
            new[] { input }
        );

        Assert.False(result.Passed);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal("SampleEvaluationFailed", diagnostic.Code);
        Assert.Contains("Bool", diagnostic.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Evaluate_UsesExportedOnnxModelForSessionCreation()
    {
        using var module = new IdentityModule();
        module.eval();
        using var input = tensor(new[] { 1f, 2f }, [2L], dtype: ScalarType.Float32);
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            var result = module.Evaluate(
                CreateIdentityModel(shape: [2L]),
                new[] { input },
                new TorchExportEvaluationOptions
                {
                    TemporaryModelPathFactory = () => path,
                }
            );

            Assert.True(result.Passed);
            Assert.False(File.Exists(path));
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static OnnxModel CreateIdentityModel(long[]? shape = null)
    {
        shape ??= [2L, 2L];
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 22,
            ProducerName = "onnxify-tests",
        });
        var graph = model.Graph;
        var input = graph.AddInput("input", OnnxTensorType.Create<float>(ToDimensions(shape)));
        var outputEdge = graph.AddEdge("output");
        graph.Identity(
            name: "identity",
            options: new IdentityInputOutputOptions
            {
                Input = input,
                Output = outputEdge,
            }
        );
        graph.AddOutput("output", OnnxTensorType.Create<float>(ToDimensions(shape)));
        return model;
    }

    private static OnnxModel CreateAddConstantModel(float value, long[] shape)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 22,
            ProducerName = "onnxify-tests",
        });
        var graph = model.Graph;
        var input = graph.AddInput("input", OnnxTensorType.Create<float>(ToDimensions(shape)));
        var constant = graph.AddTensor("constant", [], [value]);
        var output = graph.AddEdge("output");
        graph.Add(
            name: "add",
            options: new AddInputOutputOptions
            {
                A = input,
                B = constant,
                C = output,
            }
        );
        graph.AddOutput("output", OnnxTensorType.Create<float>(ToDimensions(shape)));
        return model;
    }

    private static OnnxDimension[] ToDimensions(long[] shape)
    {
        return shape.Select(static dimension => (OnnxDimension)dimension).ToArray();
    }

    private sealed class IdentityModule : TorchModule
    {
        public IdentityModule()
            : base(nameof(IdentityModule))
        {
        }

        public override TorchTensor forward(TorchTensor input)
        {
            return input;
        }
    }

    private sealed class AddOneModule : TorchModule
    {
        public AddOneModule()
            : base(nameof(AddOneModule))
        {
        }

        public override TorchTensor forward(TorchTensor input)
        {
            return input + 1;
        }
    }

    private sealed class DictionaryOutputModule : global::TorchSharp.torch.nn.Module<TorchTensor, IReadOnlyDictionary<string, TorchTensor>>
    {
        public DictionaryOutputModule()
            : base(nameof(DictionaryOutputModule))
        {
        }

        public override IReadOnlyDictionary<string, TorchTensor> forward(TorchTensor input)
        {
            return new Dictionary<string, TorchTensor>(StringComparer.Ordinal)
            {
                ["sample"] = input,
            };
        }
    }

    private sealed class BoolOutputModule : TorchModule
    {
        public BoolOutputModule()
            : base(nameof(BoolOutputModule))
        {
        }

        public override TorchTensor forward(TorchTensor input)
        {
            return input.eq(input);
        }
    }
}
