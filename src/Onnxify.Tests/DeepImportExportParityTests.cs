using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Onnxify.Tests;

public sealed class DeepImportExportParityTests
{
    [Fact]
    public void RoundTrip_ForMobileNetClassifierHeadOperators_PreservesOutput()
    {
        var model = CreateClassifierHeadModel();
        var input = NamedOnnxValue.CreateFromTensor(
            "input",
            new DenseTensor<float>(
                Enumerable.Range(0, 8).Select(static x => (x - 3.5f) / 4f).ToArray(),
                [1, 2, 2, 2]
            )
        );

        var result = DeepImportExportParity.AssertRoundTripMse(
            model,
            [input],
            outputName: "logits",
            threshold: 1e-8f
        );

        Assert.Equal([1, 3], result.OutputShape);
    }

    [Fact]
    public void RoundTrip_ForInlineFunctionalConv2d_PreservesOutput()
    {
        var model = CreateInlineFunctionalConvModel();
        var input = NamedOnnxValue.CreateFromTensor(
            "input",
            new DenseTensor<float>(
                Enumerable.Range(0, 75).Select(static x => ((x % 13) - 6) / 6f).ToArray(),
                [1, 3, 5, 5]
            )
        );

        var result = DeepImportExportParity.AssertRoundTripMse(
            model,
            [input],
            outputName: "output",
            threshold: 1e-8f
        );

        Assert.Equal([1, 2, 3, 1], result.OutputShape);
    }

    [Fact]
    public void RoundTrip_ForAsymmetricPaddedConv2d_PreservesOutput()
    {
        var model = CreateAsymmetricPaddedConvModel();
        var input = NamedOnnxValue.CreateFromTensor(
            "input",
            new DenseTensor<float>(
                Enumerable.Range(0, 16).Select(static x => ((x % 9) - 4) / 4f).ToArray(),
                [1, 1, 4, 4]
            )
        );

        var result = DeepImportExportParity.AssertRoundTripMse(
            model,
            [input],
            outputName: "output",
            threshold: 1e-8f
        );

        Assert.Equal([1, 1, 2, 2], result.OutputShape);
    }

    [Fact]
    public void RoundTrip_ForAsymmetricPaddedConv2dAfterIntermediateEdge_PreservesOutput()
    {
        var model = CreateIntermediateAsymmetricPaddedConvModel();
        var input = NamedOnnxValue.CreateFromTensor(
            "input",
            new DenseTensor<float>(
                Enumerable.Range(0, 16).Select(static x => ((x % 9) - 4) / 4f).ToArray(),
                [1, 1, 4, 4]
            )
        );

        var result = DeepImportExportParity.AssertRoundTripMse(
            model,
            [input],
            outputName: "output",
            threshold: 1e-8f
        );

        Assert.Equal([1, 1, 2, 2], result.OutputShape);
    }

    [Fact]
    public void RoundTrip_ForMobileNetInvertedResidualOperators_PreservesOutput()
    {
        var model = CreateInvertedResidualModel();
        var input = NamedOnnxValue.CreateFromTensor(
            "input",
            new DenseTensor<float>(
                Enumerable.Range(0, 32).Select(static x => ((x % 11) - 5) / 5f).ToArray(),
                [1, 2, 4, 4]
            )
        );

        var result = DeepImportExportParity.AssertRoundTripMse(
            model,
            [input],
            outputName: "output",
            threshold: 1e-8f
        );

        Assert.Equal([1, 2, 4, 4], result.OutputShape);
    }

    [Fact]
    public void RoundTrip_ForYolo26sAsset_ExportsGeneratedTorchModule()
    {
        var modelPath = Path.Combine(AppContext.BaseDirectory, "Assets", "yolo26s.onnx");
        var model = OnnxModel.FromFile(modelPath);

        var roundTrippedModel = DeepImportExportParity.AssertRoundTripExports(
            model,
            modelFileName: "yolo26s.onnx"
        );

        Assert.Equal(model.Graph.Inputs.Count, roundTrippedModel.Graph.Inputs.Count);
        Assert.Equal(model.Graph.Outputs.Count, roundTrippedModel.Graph.Outputs.Count);
        Assert.Contains(roundTrippedModel.Graph.Nodes, node => node.OpType == "Sigmoid");
    }

    private static OnnxModel CreateClassifierHeadModel()
    {
        var model = CreateModel();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1L, 2L, 2L, 2L]));
        var logits = model.Graph.AddOutput("logits", OnnxTensorType.Create<float>([1L, 3L]));
        var weight = model.Graph.AddTensor("classifier.weight", [3L, 2L], [0.5f, -0.25f, 0.75f, 0.125f, -0.5f, 0.25f]);
        var bias = model.Graph.AddTensor("classifier.bias", [3L], [0.1f, -0.2f, 0.3f]);

        var pooled = model.Graph.GlobalAveragePool(
            name: "pool",
            options: new GlobalAveragePoolInputOptions
            {
                X = input,
            }
        );
        var flattened = model.Graph.Flatten(
            name: "flatten",
            options: new FlattenInputOptions
            {
                Input = pooled,
                Axis = 1L,
            }
        );
        model.Graph.Gemm(
            name: "classifier",
            options: new GemmInputOutputOptions
            {
                A = flattened,
                B = weight,
                C = bias,
                Y = logits,
                Alpha = 1f,
                Beta = 1f,
                TransB = 1L,
            }
        );

        return model;
    }

    private static OnnxModel CreateInlineFunctionalConvModel()
    {
        var model = CreateModel();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1L, 3L, 5L, 5L]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1L, 2L, 3L, 1L]));
        var weight = model.Graph.AddTensor("weight", [2L, 3L, 3L, 3L], CreateSequence(54, 0.015f));
        var bias = model.Graph.AddTensor("bias", [2L], [0.05f, -0.025f]);

        model.Graph.Conv(
            name: "inline_conv",
            options: new ConvInputOutputOptions
            {
                X = input,
                W = weight,
                B = bias,
                Y = output,
                Dilations = [1L, 2L],
                Group = 1L,
                KernelShape = [3L, 3L],
                Pads = [0L, 0L, 0L, 0L],
                Strides = [1L, 1L],
            }
        );

        return model;
    }

    private static OnnxModel CreateAsymmetricPaddedConvModel()
    {
        var model = CreateModel();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1L, 1L, 4L, 4L]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1L, 1L, 2L, 2L]));
        var weight = model.Graph.AddTensor("weight", [1L, 1L, 3L, 3L], CreateSequence(9, 0.125f));
        var bias = model.Graph.AddTensor("bias", [1L], [0.05f]);

        model.Graph.Conv(
            name: "asymmetric_conv",
            options: new ConvInputOutputOptions
            {
                X = input,
                W = weight,
                B = bias,
                Y = output,
                Dilations = [1L, 1L],
                Group = 1L,
                KernelShape = [3L, 3L],
                Pads = [0L, 0L, 1L, 1L],
                Strides = [2L, 2L],
            }
        );

        return model;
    }

    private static OnnxModel CreateIntermediateAsymmetricPaddedConvModel()
    {
        var model = CreateModel();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1L, 1L, 4L, 4L]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1L, 1L, 2L, 2L]));
        var preWeight = model.Graph.AddTensor("pre.weight", [1L, 1L, 1L, 1L], [1.25f]);
        var preBias = model.Graph.AddTensor("pre.bias", [1L], [-0.125f]);
        var weight = model.Graph.AddTensor("weight", [1L, 1L, 3L, 3L], CreateSequence(9, 0.125f));
        var bias = model.Graph.AddTensor("bias", [1L], [0.05f]);

        var intermediate = model.Graph.Conv(
            name: "pre_conv",
            options: new ConvInputOptions
            {
                X = input,
                W = preWeight,
                B = preBias,
                Dilations = [1L, 1L],
                Group = 1L,
                KernelShape = [1L, 1L],
                Pads = [0L, 0L, 0L, 0L],
                Strides = [1L, 1L],
            }
        );

        model.Graph.Conv(
            name: "asymmetric_conv",
            options: new ConvInputOutputOptions
            {
                X = intermediate,
                W = weight,
                B = bias,
                Y = output,
                Dilations = [1L, 1L],
                Group = 1L,
                KernelShape = [3L, 3L],
                Pads = [0L, 0L, 1L, 1L],
                Strides = [2L, 2L],
            }
        );

        return model;
    }

    private static OnnxModel CreateInvertedResidualModel()
    {
        var model = CreateModel();
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1L, 2L, 4L, 4L]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1L, 2L, 4L, 4L]));
        var clipMin = model.Graph.AddTensor("clip_min", [], [0f]);
        var clipMax = model.Graph.AddTensor("clip_max", [], [6f]);
        var expandWeight = model.Graph.AddTensor("expand.weight", [4L, 2L, 1L, 1L], CreateSequence(8, 0.05f));
        var expandBias = model.Graph.AddTensor("expand.bias", [4L], [0.1f, -0.2f, 0.05f, -0.05f]);
        var depthwiseWeight = model.Graph.AddTensor("depthwise.weight", [4L, 1L, 3L, 3L], CreateSequence(36, 0.025f));
        var depthwiseBias = model.Graph.AddTensor("depthwise.bias", [4L], [0.01f, 0.02f, -0.03f, 0.04f]);
        var projectWeight = model.Graph.AddTensor("project.weight", [2L, 4L, 1L, 1L], CreateSequence(8, -0.04f));
        var projectBias = model.Graph.AddTensor("project.bias", [2L], [0.03f, -0.02f]);

        var expanded = AddConvNode(
            model,
            "expand",
            input,
            expandWeight,
            expandBias,
            group: 1L,
            kernel: [1L, 1L],
            pads: [0L, 0L, 0L, 0L]
        );
        var expandedClipped = AddClipNode(model, "expand_clip", expanded, clipMin, clipMax);
        var depthwise = AddConvNode(
            model,
            "depthwise",
            expandedClipped,
            depthwiseWeight,
            depthwiseBias,
            group: 4L,
            kernel: [3L, 3L],
            pads: [1L, 1L, 1L, 1L]
        );
        var depthwiseClipped = AddClipNode(model, "depthwise_clip", depthwise, clipMin, clipMax);
        var projected = AddConvNode(
            model,
            "project",
            depthwiseClipped,
            projectWeight,
            projectBias,
            group: 1L,
            kernel: [1L, 1L],
            pads: [0L, 0L, 0L, 0L]
        );

        model.Graph.Add(
            name: "residual",
            options: new AddInputOutputOptions
            {
                A = input,
                B = projected,
                C = output,
            }
        );

        return model;
    }

    private static OnnxModel CreateModel()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "onnxify-tests",
            IrVersion = 9,
            Opset = 22,
        });
        model.Graph.Name = "deep_import_export_parity";
        return model;
    }

    private static IOnnxGraphEdge AddConvNode(
        OnnxModel model,
        string name,
        IOnnxGraphEdge input,
        IOnnxGraphEdge weight,
        IOnnxGraphEdge bias,
        long group,
        long[] kernel,
        long[] pads
    )
    {
        return model.Graph.Conv(
            name: name,
            options: new ConvInputOptions
            {
                X = input,
                W = weight,
                B = bias,
                Dilations = [1L, 1L],
                Group = group,
                KernelShape = kernel,
                Pads = pads,
                Strides = [1L, 1L],
            }
        );
    }

    private static IOnnxGraphEdge AddClipNode(
        OnnxModel model,
        string name,
        IOnnxGraphEdge input,
        IOnnxGraphEdge min,
        IOnnxGraphEdge max
    )
    {
        return model.Graph.Clip(
            name: name,
            options: new ClipInputOptions
            {
                Input = input,
                Min = min,
                Max = max,
            }
        );
    }

    private static float[] CreateSequence(int count, float scale)
    {
        return Enumerable.Range(0, count)
            .Select(index => ((index % 7) - 3) * scale)
            .ToArray();
    }
}
