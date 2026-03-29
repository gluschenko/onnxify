using Onnxify.ProjectGenerator;

namespace Onnxify.Tests;

public sealed class OnnxProjectGeneratorTests
{
    [Fact]
    public void Generate_WritesProgramProjectAndTensorAssets()
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
            Assert.Single(result.TensorFilePaths);
            Assert.Empty(result.Warnings);

            var programText = File.ReadAllText(result.ProgramFilePath);
            Assert.Contains("namespace Sample_Generated_Project;", programText);
            Assert.Contains("model.Graph.AddInput(", programText);
            Assert.Contains("model.Graph.AddValue(", programText);
            Assert.Contains("model.Graph.AddOutput(", programText);
            Assert.Contains("model.Graph.AddTensor(", programText);
            Assert.Contains("TensorDataLoader.LoadArray<float>(\"Assets/weights.bin\")", programText);
            Assert.Contains("new OnnxAttribute<long[]>(\"axes\", [0L, 1L])", programText);
            Assert.Contains("new OnnxAttribute<string>(\"note\", \"hello\")", programText);
            Assert.Contains("model.MetadataProps.Add(new StringStringEntryProto", programText);
            Assert.Contains("model.OpsetImport.Add(new OperatorSetIdProto", programText);

            var projectText = File.ReadAllText(result.ProjectFilePath!);
            Assert.Contains("<PackageReference Include=\"Onnxify\" Version=\"9.9.9-test\" />", projectText);
            Assert.Contains("<None Include=\"Assets\\**\\*\">", projectText);

            var tensorFilePath = Assert.Single(result.TensorFilePaths);
            Assert.True(File.Exists(tensorFilePath));
            Assert.Equal(sizeof(float) * 2, new FileInfo(tensorFilePath).Length);
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
