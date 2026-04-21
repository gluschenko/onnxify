using Onnxify.CLI;
using Onnxify.Safetensors;
using SafeTensorsModel = Onnxify.Safetensors.SafeTensors;

namespace Onnxify.Tests;

public sealed class OnnxifyCliTests
{
    [Fact]
    public void Run_OnnxShow_WritesModelToString()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            var model = OnnxModel.Create(new OnnxModelCreationOptions
            {
                ProducerName = "cli-tests",
                IrVersion = 9,
                Opset = 13,
            });

            model.Graph.Name = "demo";
            var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 2]));
            var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 2]));
            var weights = model.Graph.AddTensor("weights", [1, 2], [1.0f, 2.0f]);

            model.Graph.AddNode(
                name: "node",
                opType: "Add",
                domain: "",
                docString: "",
                inputs: [input, weights],
                outputs: [output],
                attributes: []);

            model.Save(modelPath);

            var stdout = new StringWriter();
            var stderr = new StringWriter();

            var exitCode = App.Run(["onnx", "show", modelPath], stdout, stderr);

            Assert.Equal(0, exitCode);
            Assert.Equal(string.Empty, stderr.ToString());

            var outputText = stdout.ToString();
            Assert.Contains("OnnxModel(", outputText);
            Assert.Contains("Graph=OnnxGraph(", outputText);
            Assert.Contains("Initializers=", outputText);
            Assert.Contains("Nodes=", outputText);
            Assert.Contains("weights: Single[1, 2] = [1, 2]", outputText);
        }
        finally
        {
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
            }
        }
    }

    [Fact]
    public void Run_OnnxIo_WritesOnlyInputsAndOutputs()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            var model = OnnxModel.Create();
            model.Graph.Name = "io-only";
            model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 3]));
            model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 3]));
            model.Graph.AddTensor("weights", [1], [1.0f]);
            model.Save(modelPath);

            var stdout = new StringWriter();
            var stderr = new StringWriter();

            var exitCode = App.Run(["onnx", "io", modelPath], stdout, stderr);

            Assert.Equal(0, exitCode);
            Assert.Equal(string.Empty, stderr.ToString());

            var outputText = stdout.ToString();
            Assert.Contains("OnnxModelInputsOutputs(", outputText);
            Assert.Contains("GraphName=io-only", outputText);
            Assert.Contains("Inputs=", outputText);
            Assert.Contains("Outputs=", outputText);
            Assert.DoesNotContain("Initializers=", outputText);
            Assert.DoesNotContain("Nodes=", outputText);
            Assert.DoesNotContain("weights", outputText);
        }
        finally
        {
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
            }
        }
    }

    [Fact]
    public void Run_SafetensorsShow_WritesArchiveToString()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.safetensors");

        try
        {
            SafeTensorsModel.SerializeToFile(
                new Dictionary<string, TensorView>
                {
                    ["embedding"] = new(DataType.F32, [2], FloatsToBytes(1.0f, 2.0f)),
                },
                new Dictionary<string, string>
                {
                    ["framework"] = "pt",
                },
                path);

            var stdout = new StringWriter();
            var stderr = new StringWriter();

            var exitCode = App.Run(["safetensors", "show", path], stdout, stderr);

            Assert.Equal(0, exitCode);
            Assert.Equal(string.Empty, stderr.ToString());

            var outputText = stdout.ToString();
            Assert.Contains("Safetensors(", outputText);
            Assert.Contains("Metadata=", outputText);
            Assert.Contains("framework=pt", outputText);
            Assert.Contains("embedding: Single[2] = [1, 2]", outputText);
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
    public void Run_ProjectGenerate_CreatesOutputFiles()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            var model = OnnxModel.Create();
            model.Graph.AddTensor("weights", [1], [1.0f]);
            model.Save(modelPath);

            var stdout = new StringWriter();
            var stderr = new StringWriter();

            var exitCode = App.Run(
                ["project", "generate", modelPath, outputDirectoryPath, "--project-name", "CliGenerated", "--overwrite"],
                stdout,
                stderr);

            Assert.Equal(0, exitCode);
            Assert.Equal(string.Empty, stderr.ToString());

            var outputText = stdout.ToString();
            Assert.Contains("ProjectGenerationResult(", outputText);
            Assert.Contains($"OutputDirectory={Path.GetFullPath(outputDirectoryPath)}", outputText);
            Assert.Contains($"ProgramFile={Path.Combine(Path.GetFullPath(outputDirectoryPath), "Program.cs")}", outputText);

            Assert.True(File.Exists(Path.Combine(outputDirectoryPath, "Program.cs")));
            Assert.True(File.Exists(Path.Combine(outputDirectoryPath, "CliGenerated.csproj")));
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

    private static byte[] FloatsToBytes(params float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
