using Onnxify.Data.Numerics;
using Onnxify.ProjectGenerator;

namespace Onnxify.Tests;

public sealed class OnnxProjectGeneratorTests
{
    [Fact]
    public void Generate_InlinesSmallTensorValuesIntoProgram()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions
            {
                ProducerName = "generator-tests",
                IrVersion = 9,
                Opset = 13,
            });
            model.ProducerVersion = "2.3.4";
            model.ModelVersion = 12;
            model.Domain = "ai.onnxify.generator.tests";
            model.Document = "Generated project";
            model.AddMetadataProps("generator-key", "generator-value");

            var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 2], "input-denotation"));
            var hidden = model.Graph.AddValue("hidden", OnnxTensorType.Create<float>([1, 2]));
            var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 2]));
            var weights = model.Graph.AddTensor("weights", [1, 2], [1.0f, 2.0f]);

            model.Graph.AddNode(
                name: "custom_node",
                opType: "CustomOp",
                domain: "",
                docString: "Moves data around",
                inputs: [input, weights],
                outputs: [hidden],
                attributes:
                [
                    new OnnxAttribute<long[]>("axes", [0L, 1L]),
                    new OnnxAttribute<string>("note", "hello"),
                ]);

            model.Graph.AddNode(
                name: "output_node",
                opType: "Identity",
                domain: "",
                docString: "",
                inputs: [hidden],
                outputs: [output],
                attributes: []);

            model.Save(modelPath);

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(new ProjectGeneratorOptions
            {
                InputModelPath = modelPath,
                OutputDirectoryPath = outputDirectoryPath,
                ProjectName = "SampleGeneratedProject",
                Namespace = "Sample.Generated.Project",
                Overwrite = true,
                OnnxifyPackageVersion = "9.9.9-test",
            });

            Assert.Equal(Path.Combine(outputDirectoryPath, "Program.cs"), result.ProgramFilePath);
            Assert.Equal(Path.Combine(outputDirectoryPath, "SampleGeneratedProject.csproj"), result.ProjectFilePath);
            Assert.Empty(result.TensorFilePaths);
            Assert.Empty(result.Warnings);

            var programText = File.ReadAllText(result.ProgramFilePath);
            Assert.Contains("namespace Sample_Generated_Project;", programText);
            Assert.Contains("model.Graph.AddInput(", programText);
            Assert.Contains("model.Graph.AddValue(", programText);
            Assert.Contains("model.Graph.AddOutput(", programText);
            Assert.Contains("model.Graph.AddTensor(", programText);
            Assert.DoesNotContain("internal static class TensorDataLoader", programText);
            Assert.DoesNotContain("using System.Linq;", programText);
            Assert.DoesNotContain("using System.Runtime.InteropServices;", programText);
            Assert.DoesNotContain("using System.Text.Json;", programText);
            Assert.Contains("var model = OnnxModel.Create(new OnnxModelCreationOptions", programText);
            Assert.Contains("ProducerName = \"generator-tests\"", programText);
            Assert.Contains("IrVersion = 9L", programText);
            Assert.Contains("Opset = 13", programText);
            Assert.Contains("model.ClearOpsetImports();", programText);
            Assert.Contains("model.SetOpsetImport(\"\", 13L);", programText);
            Assert.Contains("value: [1F, 2F]", programText);
            Assert.DoesNotContain("OnnxExternalDataProvider.Instance.ReadTensorValue<float>(ResolveAssetPath(\"Assets/weights.bin\"), offset: 0, length: -1)", programText);
            Assert.Contains("new OnnxAttribute<long[]>(\"axes\", [0L, 1L])", programText);
            Assert.Contains("new OnnxAttribute<string>(\"note\", \"hello\")", programText);
            Assert.Contains("model.AddMetadataProps(\"generator-key\", \"generator-value\");", programText);
            Assert.DoesNotContain("StringStringEntryProto", programText);
            Assert.DoesNotContain("OperatorSetIdProto", programText);
            Assert.DoesNotContain("MetadataProps.Clear()", programText);
            Assert.DoesNotContain("OpsetImport.Clear()", programText);

            var projectText = File.ReadAllText(result.ProjectFilePath!);
            Assert.Contains("<PackageReference Include=\"Onnxify\" Version=\"9.9.9-test\" />", projectText);
            Assert.Contains("<None Include=\"Assets\\**\\*\">", projectText);

        }
        finally
        {
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
            }

            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_WritesTensorAssetsForLargeTensors()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions
            {
                ProducerName = "large-tensor-generator-tests",
                IrVersion = 9,
                Opset = 13,
            });

            model.Graph.AddTensor("weights", [4, 5], Enumerable.Range(1, 20).Select(static x => (float)x).ToArray());

            model.Save(modelPath);

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(new ProjectGeneratorOptions
            {
                InputModelPath = modelPath,
                OutputDirectoryPath = outputDirectoryPath,
                Overwrite = true,
            });

            Assert.Single(result.TensorFilePaths);

            var programText = File.ReadAllText(result.ProgramFilePath);
            Assert.Contains("OnnxExternalDataProvider.Instance.ReadTensorValue<float>(ResolveAssetPath(\"Assets/weights.bin\"), offset: 0, length: -1)", programText);

            var tensorFilePath = Assert.Single(result.TensorFilePaths);
            Assert.True(File.Exists(tensorFilePath));
            Assert.Equal(sizeof(float) * 20, new FileInfo(tensorFilePath).Length);
        }
        finally
        {
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
            }

            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_EmitsGraphNameAndCustomOpsetImports()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions
            {
                ProducerName = "custom-opset-tests",
                IrVersion = 9,
                Opset = 13,
            });

            model.Graph.Name = "bvlc_alexnet";
            model.ClearOpsetImports();
            model.SetOpsetImport("ai.onnx.ml", 2);
            model.SetOpsetImport("ai.onnx.preview.training", 1);
            model.SetOpsetImport("com.microsoft", 1);
            model.SetOpsetImport("com.microsoft.experimental", 1);
            model.SetOpsetImport("com.microsoft.nchwc", 1);
            model.SetOpsetImport("ai.onnx.training", 1);
            model.SetOpsetImport("ai.onnx.contrib", 1000);

            model.Save(modelPath);

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(new ProjectGeneratorOptions
            {
                InputModelPath = modelPath,
                OutputDirectoryPath = outputDirectoryPath,
                Overwrite = true,
            });

            Assert.Empty(result.Warnings);

            var programText = File.ReadAllText(result.ProgramFilePath);
            Assert.Contains("model.ClearOpsetImports();", programText);
            Assert.DoesNotContain("model.SetOpsetImport(\"\",", programText);
            Assert.Contains("model.SetOpsetImport(\"ai.onnx.ml\", 2L);", programText);
            Assert.Contains("model.SetOpsetImport(\"ai.onnx.preview.training\", 1L);", programText);
            Assert.Contains("model.SetOpsetImport(\"com.microsoft\", 1L);", programText);
            Assert.Contains("model.SetOpsetImport(\"com.microsoft.experimental\", 1L);", programText);
            Assert.Contains("model.SetOpsetImport(\"com.microsoft.nchwc\", 1L);", programText);
            Assert.Contains("model.SetOpsetImport(\"ai.onnx.training\", 1L);", programText);
            Assert.Contains("model.SetOpsetImport(\"ai.onnx.contrib\", 1000L);", programText);
            Assert.Contains("model.Graph.Name = \"bvlc_alexnet\";", programText);
        }
        finally
        {
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
            }

            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_InlinesCurrentNumericTensorTypesWhenSmall()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions
            {
                ProducerName = "numeric-generator-tests",
                IrVersion = 9,
                Opset = 13,
            });

            model.Graph.AddTensor("float8", [2], [new Float8E4M3FN(1.0f), new Float8E4M3FN(-2.0f)]);
            model.Graph.AddTensor("float4", [3], [new Float4E2M1(0.5f), new Float4E2M1(1.0f), new Float4E2M1(2.0f)]);
            model.Graph.AddTensor("uint4", [3], [new UInt4(1), new UInt4(2), new UInt4(15)]);
            model.Graph.AddTensor("int2", [5], [new Int2(-1), new Int2(0), new Int2(1), new Int2(-2), new Int2(1)]);
            model.Graph.AddTensor("float8e8m0", [2], [new Float8E8M0(1.0f), new Float8E8M0(2.0f)]);

            model.Save(modelPath);

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(new ProjectGeneratorOptions
            {
                InputModelPath = modelPath,
                OutputDirectoryPath = outputDirectoryPath,
                Overwrite = true,
                OnnxifyPackageVersion = "9.9.9-test",
            });

            Assert.Empty(result.TensorFilePaths);

            var programText = File.ReadAllText(result.ProgramFilePath);
            Assert.Contains("using Onnxify.Data.Numerics;", programText);
            Assert.DoesNotContain("internal static class TensorDataLoader", programText);
            Assert.Contains("value: [new Float8E4M3FN(", programText);
            Assert.Contains("value: [new Float4E2M1(", programText);
            Assert.Contains("value: [new UInt4(", programText);
            Assert.Contains("value: [new Int2(", programText);
            Assert.Contains("value: [new Float8E8M0(", programText);
            Assert.DoesNotContain("ReadTensorValue<Float8E4M3FN>", programText);
        }
        finally
        {
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
            }

            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_UsesLowerCamelCaseForGeneratedLocals()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            var model = OnnxModel.Create();

            var input = model.Graph.AddInput("OC2_DUMMY_0", OnnxTensorType.Create<float>([1]));
            var output = model.Graph.AddOutput("OC2_DUMMY_0_quantized", OnnxTensorType.Create<float>([1]));

            model.Graph.AddNode(
                name: "IdentityNode",
                opType: "Identity",
                domain: "",
                docString: "",
                inputs: [input],
                outputs: [output],
                attributes: []);

            model.Save(modelPath);

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(new ProjectGeneratorOptions
            {
                InputModelPath = modelPath,
                OutputDirectoryPath = outputDirectoryPath,
                Overwrite = true,
            });

            var programText = File.ReadAllText(result.ProgramFilePath);
            Assert.Contains("var oc2_dummy_0 = model.Graph.AddInput(", programText);
            Assert.Contains("var oc2_dummy_0_quantized = model.Graph.AddOutput(", programText);
        }
        finally
        {
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
            }

            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_UsesTypedOperatorWrappersWhenAvailable()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            var model = OnnxModel.Create();

            var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 1, 3, 3]));
            var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 1, 3, 3]));
            var scale = model.Graph.AddTensor("scale", [1], [1.0f]);
            var zeroPoint = model.Graph.AddTensor("zero_point", [1], [(byte)0]);
            var weights = model.Graph.AddTensor("weights", [1, 1, 3, 3], new float[9]);
            var bias = model.Graph.AddTensor("bias", [1], [0.0f]);
            var quantized = model.Graph.AddEdge("quantized");
            var dequantized = model.Graph.AddEdge("dequantized");

            model.Graph.QuantizeLinear(
                name: "quantize",
                options: new QuantizeLinearInputOutputOptions
                {
                    X = input,
                    YScale = scale,
                    YZeroPoint = zeroPoint,
                    Y = quantized,
                    Axis = 1,
                    BlockSize = 0,
                    OutputDtype = 0,
                    Precision = 0,
                    Saturate = 1,
                });

            model.Graph.DequantizeLinear(
                name: "dequantize",
                options: new DequantizeLinearInputOutputOptions
                {
                    X = quantized,
                    XScale = scale,
                    XZeroPoint = zeroPoint,
                    Y = dequantized,
                    Axis = 1,
                    BlockSize = 0,
                    OutputDtype = 0,
                });

            model.Graph.Conv(
                name: "conv",
                options: new ConvInputOutputOptions
                {
                    X = dequantized,
                    W = weights,
                    B = bias,
                    Y = output,
                    AutoPad = "NOTSET",
                    Group = 1,
                    KernelShape = [3, 3],
                    Pads = [1, 1, 1, 1],
                    Strides = [1, 1],
                });

            model.Save(modelPath);

            var generator = new OnnxProjectGenerator();
            var result = generator.Generate(new ProjectGeneratorOptions
            {
                InputModelPath = modelPath,
                OutputDirectoryPath = outputDirectoryPath,
                Overwrite = true,
            });

            var programText = File.ReadAllText(result.ProgramFilePath);
            Assert.Contains("model.Graph.QuantizeLinear(", programText);
            Assert.Contains("options: new global::Onnxify.QuantizeLinearInputOutputOptions", programText);
            Assert.Contains("Y = quantized", programText);
            Assert.Contains("model.Graph.DequantizeLinear(", programText);
            Assert.Contains("options: new global::Onnxify.DequantizeLinearInputOutputOptions", programText);
            Assert.Contains("model.Graph.Conv(", programText);
            Assert.Contains("options: new global::Onnxify.ConvInputOutputOptions", programText);
            Assert.Contains("AutoPad = \"NOTSET\"", programText);
            Assert.DoesNotContain("model.Graph.AddNode(", programText);
        }
        finally
        {
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
            }

            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }
}
