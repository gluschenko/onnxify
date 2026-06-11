extern alias ModelGen;

using System.Text;
using Google.Protobuf;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;
using Onnx;
using OnnxModelGenerator = ModelGen::Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.Tests;

public sealed class OnnxModelGeneratorTests
{
    [Fact]
    public void Generate_ForAdditionalOnnxFile_ProducesTypedWrapper()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "Models", "sample-classifier.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

        try
        {
            CreateTensorModel(
                modelPath: modelPath,
                inputName: "input_ids",
                inputType: OnnxTensorType.Create<long>(new OnnxDimension[] { 1L, "sequence_length" }, "token_ids"),
                outputName: "logits",
                outputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, "sequence_length", 128L }, "class_scores"));

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("namespace Demo.App", generatedSource);
            Assert.Contains("public sealed class SampleClassifierModel", generatedSource);
            Assert.Contains("/// <summary>", generatedSource);
            Assert.Contains("Provides a typed ONNX Runtime wrapper for the model file 'sample-classifier.onnx'.", generatedSource);
            Assert.Contains("public sealed class SampleClassifierModelInputs", generatedSource);
            Assert.Contains("Input property <c>InputIds</c> maps to ONNX name <c>input_ids</c>; tensor type <c>Tensor&lt;long&gt;</c>; shape <c>[1, sequence_length]</c>; denotation <c>token_ids</c>", generatedSource);
            Assert.Contains("public sealed class SampleClassifierModelOutputs", generatedSource);
            Assert.Contains("Output property <c>Logits</c> maps to ONNX name <c>logits</c>; tensor type <c>Tensor&lt;float&gt;</c>; shape <c>[1, sequence_length, 128]</c>; denotation <c>class_scores</c>", generatedSource);
            Assert.Contains("public required Tensor<long> InputIds { get; init; }", generatedSource);
            Assert.Contains("Gets or initializes the tensor supplied for model input 'input_ids'.", generatedSource);
            Assert.Contains("Tensor type: <c>Tensor&lt;long&gt;</c>", generatedSource);
            Assert.Contains("Element type: <c>long</c>", generatedSource);
            Assert.Contains("Shape: <c>[1, sequence_length]</c>", generatedSource);
            Assert.Contains("Denotation: <c>token_ids</c>", generatedSource);
            Assert.Contains("public Tensor<float> Logits => GetTensor<float>(\"logits\")", generatedSource);
            Assert.Contains("Gets the tensor returned for model output 'logits'.", generatedSource);
            Assert.Contains("Tensor type: <c>Tensor&lt;float&gt;</c>", generatedSource);
            Assert.Contains("Shape: <c>[1, sequence_length, 128]</c>", generatedSource);
            Assert.Contains("Denotation: <c>class_scores</c>", generatedSource);
            Assert.Contains("NamedOnnxValue.CreateFromTensor(\"input_ids\"", generatedSource);
            Assert.Contains("MODEL_PROJECT_RELATIVE_PATH = @\"Models\\sample-classifier.onnx\"", generatedSource);
            Assert.Contains("public static IReadOnlyList<Onnxify.OnnxValue> Inputs { get; } = CreateInputs();", generatedSource);
            Assert.Contains("public static IReadOnlyList<Onnxify.OnnxValue> Outputs { get; } = CreateOutputs();", generatedSource);
            Assert.Contains("<c>input_ids</c>: <c>Tensor&lt;long&gt;</c>, shape <c>[1, sequence_length]</c>, denotation <c>token_ids</c>", generatedSource);
            Assert.Contains("<c>logits</c>: <c>Tensor&lt;float&gt;</c>, shape <c>[1, sequence_length, 128]</c>, denotation <c>class_scores</c>", generatedSource);
            Assert.Contains("/// <param name=\"inputIds\">Tensor value for model input <c>input_ids</c>; parameter type <c>Tensor&lt;long&gt;</c>; shape <c>[1, sequence_length]</c>; denotation <c>token_ids</c></param>", generatedSource);
            Assert.Contains("public SampleClassifierModel()", generatedSource);
            Assert.Contains("public SampleClassifierModel(SessionOptions? sessionOptions)", generatedSource);
            Assert.Contains("new Onnxify.OnnxValue<Onnxify.OnnxTensorType>(", generatedSource);
            Assert.Contains("Onnxify.OnnxTensorType.Create<long>(", generatedSource);
            Assert.Contains("\"sequence_length\"", generatedSource);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_RespectsAdditionalFileMetadataOverrides()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "override-me.onnx");

        try
        {
            CreateTensorModel(
                modelPath: modelPath,
                inputName: "pixel_values",
                inputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 3L, 224L, 224L }),
                outputName: "scores",
                outputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 1000L }));

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Ignored.Root.Namespace",
                },
                fileOptions: new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)
                {
                    [modelPath] = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["build_metadata.additionalfiles.OnnxifyModelClassName"] = "VisionWrapper",
                        ["build_metadata.additionalfiles.OnnxifyModelNamespace"] = "Demo.Custom.Models",
                    }
                });

            var compilation = CreateCompilation();
            driver = driver.RunGenerators(compilation);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("namespace Demo.Custom.Models", generatedSource);
            Assert.Contains("public sealed class VisionWrapperModel", generatedSource);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WithTorchModuleImportType_ProducesGraphShapedTorchModule()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "add-bias.onnx");

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };
            model.Graph.Input.Add(CreateTensorValueInfo("input", TensorProto.Types.DataType.Float, 1L, 2L));
            model.Graph.Output.Add(CreateTensorValueInfo("output", TensorProto.Types.DataType.Float, 1L, 2L));
            model.Graph.Initializer.Add(new TensorProto
            {
                Name = "bias",
                DataType = (int)TensorProto.Types.DataType.Float,
                Dims = { 1L, 2L },
                FloatData = { 1.0f, 2.0f },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "add_bias",
                OpType = "Add",
                Input = { "input", "bias" },
                Output = { "output" },
            });

            File.WriteAllBytes(modelPath, model.ToByteArray());

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                },
                fileOptions: new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)
                {
                    [modelPath] = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["build_metadata.additionalfiles.OnnxifyModelImportType"] = "TorchModule",
                    }
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSources = driver.GetRunResult()
                .Results
                .SelectMany(static x => x.GeneratedSources)
                .Select(static x => x.SourceText.ToString())
                .ToArray();

            var generatedSource = Assert.Single(generatedSources);
            Assert.Contains("public sealed class AddBiasModelTorchModule : torch.nn.Module<Tensor, Tensor>", generatedSource);
            Assert.Contains("var biasParameter = new global::TorchSharp.Modules.Parameter(torch.empty(new long[] { 1L, 2L }, dtype: ScalarType.Float32));", generatedSource);
            Assert.Contains("register_parameter(\"bias\", biasParameter);", generatedSource);
            Assert.Contains("public void LoadWeightsFromOnnx(string modelPath)", generatedSource);
            Assert.Contains("LoadWeightsFromOnnx(model);", generatedSource);
            Assert.Contains("public void LoadWeightsFromOnnx(Onnxify.OnnxModel model)", generatedSource);
            Assert.Contains("throw new ArgumentNullException(nameof(model));", generatedSource);
            Assert.Contains("var output = input + _bias;", generatedSource);
            Assert.Contains("return output;", generatedSource);
            Assert.DoesNotContain("InferenceSession", generatedSource, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WithTorchModuleImportType_EmitsCommonUnaryMathOperators()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "common-unary.onnx");

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };
            model.Graph.Input.Add(CreateTensorValueInfo("input", TensorProto.Types.DataType.Float, 1L, 2L));
            model.Graph.Output.Add(CreateTensorValueInfo("output", TensorProto.Types.DataType.Float, 1L, 2L));
            model.Graph.Initializer.Add(new TensorProto
            {
                Name = "pow_exp",
                DataType = (int)TensorProto.Types.DataType.Float,
                Dims = { 1L },
                FloatData = { 2f },
            });
            AddUnary("abs", "Abs", "input", "abs_out");
            AddUnary("neg", "Neg", "abs_out", "neg_out");
            AddUnary("exp", "Exp", "neg_out", "exp_out");
            AddUnary("log", "Log", "exp_out", "log_out");
            AddUnary("sqrt", "Sqrt", "log_out", "sqrt_out");
            AddUnary("floor", "Floor", "sqrt_out", "floor_out");
            AddUnary("ceil", "Ceil", "floor_out", "ceil_out");
            model.Graph.Node.Add(new NodeProto
            {
                Name = "pow",
                OpType = "Pow",
                Input = { "ceil_out", "pow_exp" },
                Output = { "output" },
            });
            File.WriteAllBytes(modelPath, model.ToByteArray());

            var generatedSource = GenerateSingleTorchModuleSource(tempRoot, modelPath);

            Assert.Contains(".abs()", generatedSource);
            Assert.Contains(".exp()", generatedSource);
            Assert.Contains(".log()", generatedSource);
            Assert.Contains(".sqrt()", generatedSource);
            Assert.Contains(".floor()", generatedSource);
            Assert.Contains(".ceil()", generatedSource);
            Assert.Contains(".pow(_powExp)", generatedSource);

            void AddUnary(string name, string opType, string input, string output)
            {
                model.Graph.Node.Add(new NodeProto
                {
                    Name = name,
                    OpType = opType,
                    Input = { input },
                    Output = { output },
                });
            }
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WithTorchModuleImportType_EmitsCommonPoolingShapeAndReductionOperators()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "common-pool-reduce.onnx");

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };
            model.Graph.Input.Add(CreateTensorValueInfo("input", TensorProto.Types.DataType.Float, 1L, 3L, 8L, 8L));
            model.Graph.Output.Add(CreateTensorValueInfo("output", TensorProto.Types.DataType.Float, 1L));
            model.Graph.Initializer.Add(new TensorProto
            {
                Name = "squeeze_axes",
                DataType = (int)TensorProto.Types.DataType.Int64,
                Dims = { 2L },
                Int64Data = { 2L, 3L },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "average_pool",
                OpType = "AveragePool",
                Input = { "input" },
                Output = { "pooled" },
                Attribute =
                {
                    new AttributeProto { Name = "kernel_shape", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 2L, 2L } },
                    new AttributeProto { Name = "strides", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 2L, 2L } },
                },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "leaky_relu",
                OpType = "LeakyRelu",
                Input = { "pooled" },
                Output = { "activated" },
                Attribute =
                {
                    new AttributeProto { Name = "alpha", Type = AttributeProto.Types.AttributeType.Float, F = 0.2f },
                },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "reduce_mean",
                OpType = "ReduceMean",
                Input = { "activated" },
                Output = { "mean" },
                Attribute =
                {
                    new AttributeProto { Name = "axes", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 2L, 3L } },
                    new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 1L },
                },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "squeeze",
                OpType = "Squeeze",
                Input = { "mean", "squeeze_axes" },
                Output = { "squeezed" },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "reduce_sum",
                OpType = "ReduceSum",
                Input = { "squeezed" },
                Output = { "output" },
                Attribute =
                {
                    new AttributeProto { Name = "axes", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 1L } },
                    new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 0L },
                },
            });
            File.WriteAllBytes(modelPath, model.ToByteArray());

            var generatedSource = GenerateSingleTorchModuleSource(tempRoot, modelPath);

            Assert.Contains("torch.nn.functional.avg_pool2d", generatedSource);
            Assert.Contains("torch.nn.functional.leaky_relu", generatedSource);
            Assert.Contains("ReduceMeanTensor", generatedSource);
            Assert.Contains("SqueezeTensor", generatedSource);
            Assert.Contains("ReduceSumTensor", generatedSource);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WithTorchModuleImportType_EmitsAdditionalMathAndActivationOperators()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "additional-math-activations.onnx");

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };
            model.Graph.Input.Add(CreateTensorValueInfo("input", TensorProto.Types.DataType.Float, 1L, 2L));
            model.Graph.Output.Add(CreateTensorValueInfo("output", TensorProto.Types.DataType.Float, 1L, 2L));
            AddUnary("sin", "Sin", "input", "sin_out");
            AddUnary("cos", "Cos", "sin_out", "cos_out");
            AddUnary("tan", "Tan", "cos_out", "tan_out");
            AddUnary("erf", "Erf", "tan_out", "erf_out");
            AddUnary("reciprocal", "Reciprocal", "erf_out", "reciprocal_out");
            model.Graph.Node.Add(new NodeProto
            {
                Name = "elu",
                OpType = "Elu",
                Input = { "reciprocal_out" },
                Output = { "elu_out" },
                Attribute =
                {
                    new AttributeProto { Name = "alpha", Type = AttributeProto.Types.AttributeType.Float, F = 1.25f },
                },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "hard_sigmoid",
                OpType = "HardSigmoid",
                Input = { "elu_out" },
                Output = { "output" },
                Attribute =
                {
                    new AttributeProto { Name = "alpha", Type = AttributeProto.Types.AttributeType.Float, F = 0.2f },
                    new AttributeProto { Name = "beta", Type = AttributeProto.Types.AttributeType.Float, F = 0.5f },
                },
            });
            File.WriteAllBytes(modelPath, model.ToByteArray());

            var generatedSource = GenerateSingleTorchModuleSource(tempRoot, modelPath);

            Assert.Contains(".sin()", generatedSource);
            Assert.Contains(".cos()", generatedSource);
            Assert.Contains(".tan()", generatedSource);
            Assert.Contains(".erf()", generatedSource);
            Assert.Contains(".reciprocal()", generatedSource);
            Assert.Contains("torch.nn.functional.elu", generatedSource);
            Assert.Contains(".clamp(0.0f, 1.0f)", generatedSource);

            void AddUnary(string name, string opType, string input, string output)
            {
                model.Graph.Node.Add(new NodeProto
                {
                    Name = name,
                    OpType = opType,
                    Input = { input },
                    Output = { output },
                });
            }
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WithTorchModuleImportType_EmitsCompareSelectAndCastOperators()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "compare-select-cast.onnx");

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };
            model.Graph.Input.Add(CreateTensorValueInfo("input", TensorProto.Types.DataType.Float, 1L, 2L));
            model.Graph.Output.Add(CreateTensorValueInfo("output", TensorProto.Types.DataType.Float, 1L, 2L));
            model.Graph.Initializer.Add(new TensorProto
            {
                Name = "threshold",
                DataType = (int)TensorProto.Types.DataType.Float,
                Dims = { 1L, 2L },
                FloatData = { 0.25f, 0.75f },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "greater",
                OpType = "Greater",
                Input = { "input", "threshold" },
                Output = { "greater_out" },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "less",
                OpType = "Less",
                Input = { "input", "threshold" },
                Output = { "less_out" },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "equal",
                OpType = "Equal",
                Input = { "input", "threshold" },
                Output = { "equal_out" },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "where",
                OpType = "Where",
                Input = { "greater_out", "input", "threshold" },
                Output = { "selected" },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "cast",
                OpType = "Cast",
                Input = { "selected" },
                Output = { "output" },
                Attribute =
                {
                    new AttributeProto { Name = "to", Type = AttributeProto.Types.AttributeType.Int, I = (long)TensorProto.Types.DataType.Float },
                },
            });
            File.WriteAllBytes(modelPath, model.ToByteArray());

            var generatedSource = GenerateSingleTorchModuleSource(tempRoot, modelPath);

            Assert.Contains(".gt(_threshold)", generatedSource);
            Assert.Contains(".lt(_threshold)", generatedSource);
            Assert.Contains(".eq(_threshold)", generatedSource);
            Assert.Contains("torch.where", generatedSource);
            Assert.Contains(".to(ScalarType.Float32)", generatedSource);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WithTorchModuleImportType_AllowsUint8Initializers()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "uint8-initializer.onnx");

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };

            model.Graph.Input.Add(CreateTensorValueInfo("input", TensorProto.Types.DataType.Float, 1L, 2L));
            model.Graph.ValueInfo.Add(CreateTensorValueInfo("quantized", TensorProto.Types.DataType.Uint8, 1L, 2L));
            model.Graph.Output.Add(CreateTensorValueInfo("output", TensorProto.Types.DataType.Float, 1L, 2L));
            model.Graph.Node.Add(new NodeProto
            {
                Name = "quantize",
                OpType = "QuantizeLinear",
                Input = { "input", "data_0_scale", "data_0_zero_point" },
                Output = { "quantized" },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "dequantize",
                OpType = "DequantizeLinear",
                Input = { "quantized", "data_0_scale", "data_0_zero_point" },
                Output = { "output" },
            });
            var scale = new TensorProto
            {
                Name = "data_0_scale",
                DataType = (int)TensorProto.Types.DataType.Float,
                Dims = { 1L },
                FloatData = { 0.05f },
            };
            var zeroPoint = new TensorProto
            {
                Name = "data_0_zero_point",
                DataType = (int)TensorProto.Types.DataType.Uint8,
                Dims = { 1L },
                Int32Data = { 128 },
            };
            model.Graph.Initializer.Add(scale);
            model.Graph.Initializer.Add(zeroPoint);
            File.WriteAllBytes(modelPath, model.ToByteArray());

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                },
                fileOptions: new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)
                {
                    [modelPath] = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["build_metadata.additionalfiles.OnnxifyModelImportType"] = "TorchModule",
                    }
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("torch.empty(new long[] { 1L }, dtype: ScalarType.Byte)", generatedSource);
            Assert.Contains("LoadTensor<byte>(tensors, \"data_0_zero_point\", _data0ZeroPoint, ScalarType.Byte);", generatedSource);
            Assert.Contains("QuantizeLinearTensor(input, _data0Scale, _data0ZeroPoint, 1L)", generatedSource);
            Assert.Contains("DequantizeLinearTensor(quantized, _data0Scale, _data0ZeroPoint, 1L)", generatedSource);
            Assert.DoesNotContain("unsupported tensor data type 'Uint8'", generatedSource, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WithTorchModuleImportType_EmitsConv2dModuleFields()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "conv.onnx");

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };
            model.Graph.Input.Add(CreateTensorValueInfo("input", TensorProto.Types.DataType.Float, 1L, 3L, 5L, 5L));
            model.Graph.Output.Add(CreateTensorValueInfo("output", TensorProto.Types.DataType.Float, 1L, 2L, 5L, 5L));
            var weight = new TensorProto
            {
                Name = "conv.weight",
                DataType = (int)TensorProto.Types.DataType.Float,
                Dims = { 2L, 3L, 3L, 3L },
            };
            weight.FloatData.AddRange(Enumerable.Repeat(0.1f, 54));
            model.Graph.Initializer.Add(weight);
            model.Graph.Node.Add(new NodeProto
            {
                Name = "conv1",
                OpType = "Conv",
                Input = { "input", "conv.weight" },
                Output = { "conv.output" },
                Attribute =
                {
                    new AttributeProto { Name = "strides", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 1L, 1L } },
                    new AttributeProto { Name = "pads", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 1L, 1L, 1L, 1L } },
                    new AttributeProto { Name = "dilations", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 1L, 1L } },
                    new AttributeProto { Name = "group", Type = AttributeProto.Types.AttributeType.Int, I = 1L },
                },
            });
            model.Graph.Node.Add(new NodeProto
            {
                Name = "relu1",
                OpType = "Relu",
                Input = { "conv.output" },
                Output = { "output" },
            });

            File.WriteAllBytes(modelPath, model.ToByteArray());

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                },
                fileOptions: new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)
                {
                    [modelPath] = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["build_metadata.additionalfiles.OnnxifyModelImportType"] = "TorchModule",
                    }
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = Assert.Single(driver.GetRunResult()
                .Results
                .SelectMany(static x => x.GeneratedSources)
                .Select(static x => x.SourceText.ToString()));

            Assert.Contains("private readonly TorchModules.Conv2d _conv1;", generatedSource);
            Assert.Contains("private readonly TorchModules.ReLU _relu1;", generatedSource);
            Assert.Contains("private readonly ForwardBlock0Module _forwardBlock0;", generatedSource);
            Assert.Contains("private sealed class ForwardBlock0Module : torch.nn.Module<Tensor, Tensor>", generatedSource);
            Assert.Contains("_conv1 = Conv2d(3, 2, kernel_size: 3L, stride: 1L, padding: 1L, dilation: 1L, groups: 1L, bias: false);", generatedSource);
            Assert.Contains("_relu1 = ReLU();", generatedSource);
            Assert.Contains("LoadFloatTensor(tensors, \"conv.weight\", _conv1.weight);", generatedSource);
            Assert.Contains("var output = _forwardBlock0.forward(input);", generatedSource);
            Assert.Contains("var convOutput = _conv1.forward(input);", generatedSource);
            Assert.Contains("var output = _relu1.forward(convOutput);", generatedSource);
            Assert.DoesNotContain("functional.conv2d", generatedSource, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WithTorchModuleImportType_EmitsMaxPool2dModuleFields()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "maxpool.onnx");

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };
            model.Graph.Input.Add(CreateTensorValueInfo("input", TensorProto.Types.DataType.Float, 1L, 3L, 8L, 8L));
            model.Graph.Output.Add(CreateTensorValueInfo("output", TensorProto.Types.DataType.Float, 1L, 3L, 3L, 3L));
            model.Graph.Node.Add(new NodeProto
            {
                Name = "pool1",
                OpType = "MaxPool",
                Input = { "input" },
                Output = { "output" },
                Attribute =
                {
                    new AttributeProto { Name = "kernel_shape", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 3L, 3L } },
                    new AttributeProto { Name = "strides", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 2L, 2L } },
                },
            });
            File.WriteAllBytes(modelPath, model.ToByteArray());

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                },
                fileOptions: new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)
                {
                    [modelPath] = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["build_metadata.additionalfiles.OnnxifyModelImportType"] = "TorchModule",
                    }
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("private readonly TorchModules.MaxPool2d _pool1;", generatedSource);
            Assert.Contains("MaxPool2d(", generatedSource);
            Assert.Contains("stride:", generatedSource);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WithTorchModuleImportType_EmitsBatchNorm2dModuleFields()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "batch-norm.onnx");

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };
            model.Graph.Input.Add(CreateTensorValueInfo("input", TensorProto.Types.DataType.Float, 1L, 2L, 4L, 4L));
            model.Graph.Output.Add(CreateTensorValueInfo("output", TensorProto.Types.DataType.Float, 1L, 2L, 4L, 4L));
            AddFloatInitializer(model.Graph, "scale", [2L], [1f, 1f]);
            AddFloatInitializer(model.Graph, "bias", [2L], [0f, 0f]);
            AddFloatInitializer(model.Graph, "mean", [2L], [0f, 0f]);
            AddFloatInitializer(model.Graph, "var", [2L], [1f, 1f]);
            model.Graph.Node.Add(new NodeProto
            {
                Name = "bn1",
                OpType = "BatchNormalization",
                Input = { "input", "scale", "bias", "mean", "var" },
                Output = { "output" },
                Attribute =
                {
                    new AttributeProto { Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = 1e-5f },
                    new AttributeProto { Name = "momentum", Type = AttributeProto.Types.AttributeType.Float, F = 0.9f },
                },
            });

            File.WriteAllBytes(modelPath, model.ToByteArray());

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                },
                fileOptions: new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)
                {
                    [modelPath] = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["build_metadata.additionalfiles.OnnxifyModelImportType"] = "TorchModule",
                    }
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = Assert.Single(driver.GetRunResult()
                .Results
                .SelectMany(static x => x.GeneratedSources)
                .Select(static x => x.SourceText.ToString()));

            Assert.Contains("private readonly TorchModules.BatchNorm2d _bn1;", generatedSource);
            Assert.Contains("_bn1 = BatchNorm2d(2);", generatedSource);
            Assert.Contains("LoadFloatTensor(tensors, \"scale\", _bn1.weight!);", generatedSource);
            Assert.Contains("LoadFloatTensor(tensors, \"bias\", _bn1.bias!);", generatedSource);
            Assert.Contains("LoadFloatTensor(tensors, \"mean\", _bn1.running_mean);", generatedSource);
            Assert.Contains("LoadFloatTensor(tensors, \"var\", _bn1.running_var);", generatedSource);
            Assert.Contains("var output = _bn1.forward(input);", generatedSource);
            Assert.DoesNotContain("functional.batch_norm", generatedSource, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_OptionalInputsUseNullablePropertiesAndSortedRunParameters()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "Models", "mixed-optional-inputs.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

        try
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions
            {
                ProducerName = "generator-tests",
                IrVersion = 9,
                Opset = 13,
            });

            model.Graph.AddInput("attention_mask", OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, "sequence_length" }, "mask"));
            model.Graph.AddInput("input_ids", OnnxTensorType.Create<long>(new OnnxDimension[] { 1L, "sequence_length" }, "tokens"));
            model.Graph.AddInput("bias", OnnxTensorType.Create<float>(new OnnxDimension[] { 1L }, "bias_default"));
            model.Graph.AddOutput("logits", OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, "sequence_length", 128L }, "scores"));

            var proto = model.ToProto();
            proto.Graph.Input[0].Type = new TypeProto
            {
                Denotation = "mask",
                OptionalType = new TypeProto.Types.Optional
                {
                    ElemType = new TypeProto
                    {
                        TensorType = new TypeProto.Types.Tensor
                        {
                            ElemType = (int)TensorProto.Types.DataType.Float,
                            Shape = new TensorShapeProto
                            {
                                Dim =
                                {
                                    new TensorShapeProto.Types.Dimension { DimValue = 1L },
                                    new TensorShapeProto.Types.Dimension { DimParam = "sequence_length" },
                                }
                            }
                        }
                    }
                }
            };
            proto.Graph.Initializer.Add(new TensorProto
            {
                Name = "bias",
                DataType = (int)TensorProto.Types.DataType.Float,
                Dims = { 1L },
                FloatData = { 0.0f },
            });

            File.WriteAllBytes(modelPath, proto.ToByteArray());

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("public Tensor<float>? AttentionMask { get; init; }", generatedSource);
            Assert.Contains("public required Tensor<long> InputIds { get; init; }", generatedSource);
            Assert.Contains("public Tensor<float>? Bias { get; init; }", generatedSource);
            Assert.Contains("public MixedOptionalInputsModelOutputs Run(Tensor<long> inputIds, Tensor<float>? attentionMask = null, Tensor<float>? bias = null)", generatedSource);
            Assert.Contains("public MixedOptionalInputsModelOutputs Run(Tensor<long> inputIds, RunOptions? runOptions, Tensor<float>? attentionMask = null, Tensor<float>? bias = null)", generatedSource);
            Assert.Contains("if (inputs.AttentionMask is not null)", generatedSource);
            Assert.Contains("if (inputs.Bias is not null)", generatedSource);
            Assert.Contains("NamedOnnxValue.CreateFromTensor(\"input_ids\", inputs.InputIds ?? throw new InvalidOperationException(\"Model input 'input_ids' must be provided.\"))", generatedSource);
            Assert.Contains("parameter type <c>Tensor&lt;float&gt;?</c>", generatedSource);
            Assert.Contains("pass null to omit this input and let the model use its initializer-backed default", generatedSource);
            Assert.Contains("pass null to omit this optional ONNX input", generatedSource);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_Float16Tensor_UsesOnnxRuntimeFloat16()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "Models", "float16-model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };

            model.Graph.Input.Add(CreateTensorValueInfo("input_half", TensorProto.Types.DataType.Float16, 1L, 8L));
            model.Graph.Output.Add(CreateTensorValueInfo("output_half", TensorProto.Types.DataType.Float16, 1L, 8L));

            File.WriteAllBytes(modelPath, model.ToByteArray());

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("public required Tensor<Float16> InputHalf { get; init; }", generatedSource);
            Assert.Contains("public Tensor<Float16> OutputHalf => GetTensor<Float16>(\"output_half\")", generatedSource);
            Assert.Contains("Element type: <c>Float16</c>", generatedSource);
            Assert.DoesNotContain("Tensor<Half>", generatedSource, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_BFloat16Tensor_UsesOnnxRuntimeBFloat16()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "Models", "bfloat16-model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

        try
        {
            var model = new ModelProto
            {
                Graph = new GraphProto()
            };

            model.Graph.Input.Add(CreateTensorValueInfo("input_bfloat", TensorProto.Types.DataType.Bfloat16, 1L, 8L));
            model.Graph.Output.Add(CreateTensorValueInfo("output_bfloat", TensorProto.Types.DataType.Bfloat16, 1L, 8L));

            File.WriteAllBytes(modelPath, model.ToByteArray());

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("public required Tensor<BFloat16> InputBfloat { get; init; }", generatedSource);
            Assert.Contains("public Tensor<BFloat16> OutputBfloat => GetTensor<BFloat16>(\"output_bfloat\")", generatedSource);
            Assert.Contains("Element type: <c>BFloat16</c>", generatedSource);
            Assert.DoesNotContain("Tensor<Half>", generatedSource, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_InvalidOnnxFile_ReportsInvalidModelDiagnostic()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "broken.onnx");

        try
        {
            File.WriteAllBytes(modelPath, new byte[] { 0x3A, 0x02, 0x0A });

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                });

            var compilation = CreateCompilation();
            driver = driver.RunGenerators(compilation);

            var diagnostics = driver.GetRunResult().Diagnostics;
            var diagnostic = Assert.Single(diagnostics, static x => x.Id == "OMG001");

            Assert.Contains("broken.onnx", diagnostic.GetMessage(), StringComparison.Ordinal);
            Assert.Contains("Unable to parse ONNX protobuf payload.", diagnostic.GetMessage(), StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_ModelWithoutGraph_ProducesParameterlessWrapper()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "empty-graph.onnx");

        try
        {
            File.WriteAllBytes(modelPath, new ModelProto().ToByteArray());

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("public sealed class EmptyGraphModel", generatedSource);
            Assert.Contains("public EmptyGraphModelOutputs Run()", generatedSource);
            Assert.Contains("public EmptyGraphModelOutputs Run(RunOptions? runOptions)", generatedSource);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_ReportsDuplicateTypeNames()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelAPath = Path.Combine(tempRoot, "A", "duplicate.onnx");
        var modelBPath = Path.Combine(tempRoot, "B", "duplicate.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(modelAPath)!);
        Directory.CreateDirectory(Path.GetDirectoryName(modelBPath)!);

        try
        {
            CreateTensorModel(
                modelPath: modelAPath,
                inputName: "input",
                inputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 4L }),
                outputName: "output",
                outputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 4L }));

            CreateTensorModel(
                modelPath: modelBPath,
                inputName: "input",
                inputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 4L }),
                outputName: "output",
                outputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 4L }));

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelAPath), new BinaryAdditionalText(modelBPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                });

            var compilation = CreateCompilation();
            driver = driver.RunGenerators(compilation);

            var diagnostics = driver.GetRunResult().Diagnostics;
            Assert.Contains(diagnostics, static x => x.Id == "OMG003");
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    private static void CreateTensorModel(
        string modelPath,
        string inputName,
        OnnxTensorType inputType,
        string outputName,
        OnnxTensorType outputType)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "generator-tests",
            IrVersion = 9,
            Opset = 13,
        });

        model.Graph.AddInput(inputName, inputType);
        model.Graph.AddOutput(outputName, outputType);
        model.Save(modelPath, overwrite: true);
    }

    private static ValueInfoProto CreateTensorValueInfo(
        string name,
        TensorProto.Types.DataType dataType,
        params long[] dimensions
    )
    {
        var tensorShape = new TensorShapeProto();
        foreach (var dimension in dimensions)
        {
            tensorShape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = dimension });
        }

        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto
            {
                TensorType = new TypeProto.Types.Tensor
                {
                    ElemType = (int)dataType,
                    Shape = tensorShape,
                }
            }
        };
    }

    private static void AddFloatInitializer(
        GraphProto graph,
        string name,
        IEnumerable<long> shape,
        IEnumerable<float> values
    )
    {
        var tensor = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Float,
        };
        tensor.Dims.AddRange(shape);
        tensor.FloatData.AddRange(values);
        graph.Initializer.Add(tensor);
    }

    private static string GenerateSingleTorchModuleSource(string tempRoot, string modelPath)
    {
        var driver = CreateDriver(
            additionalFiles: [new BinaryAdditionalText(modelPath)],
            globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                ["build_property.RootNamespace"] = "Demo.App",
            },
            fileOptions: new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)
            {
                [modelPath] = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_metadata.additionalfiles.OnnxifyModelImportType"] = "TorchModule",
                }
            });

        var compilation = CreateCompilation();
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

        Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
        Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

        return GetGeneratedSource(driver);
    }

    private static CSharpCompilation CreateCompilation()
    {
        var syntaxTree = CSharpSyntaxTree.ParseText("""
            namespace Demo;

            public static class Marker
            {
            }
            """);

        var trustedPlatformAssemblies = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))?
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Distinct(StringComparer.Ordinal)
            .Select(static path => MetadataReference.CreateFromFile(path))
            .ToList() ?? [];

        trustedPlatformAssemblies.Add(MetadataReference.CreateFromFile(typeof(global::Microsoft.ML.OnnxRuntime.InferenceSession).Assembly.Location));
        trustedPlatformAssemblies.Add(MetadataReference.CreateFromFile(typeof(OnnxModel).Assembly.Location));
        trustedPlatformAssemblies.Add(MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch).Assembly.Location));

        return CSharpCompilation.Create(
            assemblyName: "GeneratedModelTests",
            syntaxTrees: [syntaxTree],
            references: trustedPlatformAssemblies,
            options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
    }

    private static GeneratorDriver CreateDriver(
        IReadOnlyList<AdditionalText> additionalFiles,
        IReadOnlyDictionary<string, string> globalOptions,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>? fileOptions = null)
    {
        return CSharpGeneratorDriver.Create(
            generators: [new OnnxModelGenerator().AsSourceGenerator()],
            additionalTexts: additionalFiles,
            parseOptions: (CSharpParseOptions)CSharpParseOptions.Default.WithLanguageVersion(LanguageVersion.Latest),
            optionsProvider: new TestAnalyzerConfigOptionsProvider(
                globalOptions,
                fileOptions ?? new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)));
    }

    private static string GetGeneratedSource(GeneratorDriver driver)
    {
        var generatedSources = driver.GetRunResult()
            .Results
            .SelectMany(static x => x.GeneratedSources)
            .ToArray();

        Assert.NotEmpty(generatedSources);

        var modelSource = generatedSources
            .Select(static x => x.SourceText.ToString())
            .First(static x => x.Contains("MODEL_PROJECT_RELATIVE_PATH", StringComparison.Ordinal));

        return modelSource;
    }

    private sealed class BinaryAdditionalText(string path) : AdditionalText
    {
        public override string Path { get; } = path;

        public override SourceText GetText(CancellationToken cancellationToken = default)
        {
            return SourceText.From(string.Empty, Encoding.UTF8);
        }
    }

    private sealed class TestAnalyzerConfigOptionsProvider(
        IReadOnlyDictionary<string, string> globalOptions,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> fileOptions) : AnalyzerConfigOptionsProvider
    {
        private readonly AnalyzerConfigOptions _global = new DictionaryAnalyzerConfigOptions(globalOptions);
        private readonly IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> _fileOptions = fileOptions;

        public override AnalyzerConfigOptions GlobalOptions => _global;

        public override AnalyzerConfigOptions GetOptions(SyntaxTree tree)
        {
            return EmptyAnalyzerConfigOptions.Instance;
        }

        public override AnalyzerConfigOptions GetOptions(AdditionalText textFile)
        {
            return _fileOptions.TryGetValue(textFile.Path, out var options)
                ? new DictionaryAnalyzerConfigOptions(options)
                : EmptyAnalyzerConfigOptions.Instance;
        }
    }

    private sealed class DictionaryAnalyzerConfigOptions(IReadOnlyDictionary<string, string> values) : AnalyzerConfigOptions
    {
        public override bool TryGetValue(string key, out string value)
        {
            if (values.TryGetValue(key, out var found))
            {
                value = found;
                return true;
            }

            value = string.Empty;
            return false;
        }
    }

    private sealed class EmptyAnalyzerConfigOptions : AnalyzerConfigOptions
    {
        public static EmptyAnalyzerConfigOptions Instance { get; } = new();

        public override bool TryGetValue(string key, out string value)
        {
            value = string.Empty;
            return false;
        }
    }
}
