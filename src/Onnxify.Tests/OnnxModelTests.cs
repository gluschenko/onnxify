using System.IO;

namespace Onnxify.Tests;

public sealed class OnnxModelTests
{
    [Fact]
    public void Create_UsesProvidedOptions()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "tests",
            IrVersion = 99,
            Opset = 21,
        });

        Assert.Equal("tests", model.ProducerName);
        Assert.Equal(99, model.IrVersion);
        Assert.Single(model.OpsetImport);
        Assert.Equal("", model.OpsetImport[0].Domain);
        Assert.Equal(21, model.OpsetImport[0].Version);
        Assert.NotNull(model.Graph);
        Assert.Empty(model.Graph.Nodes);
    }

    [Fact]
    public void FromFile_MissingPath_ThrowsFileNotFoundException()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        var exception = Assert.Throws<FileNotFoundException>(() => OnnxModel.FromFile(path));

        Assert.Equal(path, exception.FileName);
    }

    [Fact]
    public void Save_ExistingFileWithoutOverwrite_ThrowsIOException()
    {
        var model = OnnxModel.Create();
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            model.Save(path);

            var exception = Assert.Throws<IOException>(() => model.Save(path));
            Assert.Contains(path, exception.Message);
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
    public void Save_AndLoad_RoundTripsGraphStructureAndTensorValues()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "roundtrip-tests",
            IrVersion = 8,
            Opset = 13,
        });
        model.ProducerVersion = "1.2.3";
        model.ModelVersion = 7;
        model.Domain = "ai.onnxify.tests";
        model.Document = "Roundtrip document";

        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 3]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 3], "scores"));
        var hidden = model.Graph.AddValue("hidden", OnnxTensorType.Create<float>([1, 3]));
        var weights = model.Graph.AddTensor("weights", [1, 3], [1.0f, 2.0f, 3.0f]);

        model.Graph.AddNode(
            name: "add_node",
            opType: "CustomAdd",
            domain: "",
            docString: "Adds input and weights",
            inputs: [input, weights],
            outputs: [hidden],
            attributes: []);

        model.Graph.AddNode(
            name: "identity_node",
            opType: "CustomIdentity",
            domain: "",
            docString: "Forwards hidden to output",
            inputs: [hidden],
            outputs: [output],
            attributes: [new OnnxAttribute<string>("label", "final")]);

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            model.Save(path);
            var loaded = OnnxModel.FromFile(path);

            Assert.Equal("roundtrip-tests", loaded.ProducerName);
            Assert.Equal("1.2.3", loaded.ProducerVersion);
            Assert.Equal(7, loaded.ModelVersion);
            Assert.Equal(8, loaded.IrVersion);
            Assert.Equal("ai.onnxify.tests", loaded.Domain);
            Assert.Equal("Roundtrip document", loaded.Document);
            Assert.Equal(string.Empty, loaded.Graph.Name);

            Assert.Single(loaded.Graph.Inputs);
            var loadedInput = Assert.IsType<OnnxValue<OnnxTensorType>>(loaded.Graph.Inputs[0]);
            Assert.Equal("input", loadedInput.Name);
            Assert.Equal(typeof(float), loadedInput.Type.Type);
            Assert.Equal(2, loadedInput.Type.Shape.Dimensions.Length);
            Assert.Equal(1L, Assert.IsType<OnnxDimension<long>>(loadedInput.Type.Shape.Dimensions[0]).Value);
            Assert.Equal(3L, Assert.IsType<OnnxDimension<long>>(loadedInput.Type.Shape.Dimensions[1]).Value);

            Assert.Single(loaded.Graph.Outputs);
            var loadedOutput = Assert.IsType<OnnxValue<OnnxTensorType>>(loaded.Graph.Outputs[0]);
            Assert.Equal("scores", loadedOutput.Type.Denotation);

            var loadedTensor = Assert.Single(loaded.Graph.Initializers);
            var typedTensor = Assert.IsType<OnnxTensor<float>>(loadedTensor);
            Assert.Equal("weights", typedTensor.Name);
            Assert.Equal(OnnxTensor.TensorDataLocation.Default, typedTensor.DataLocation);
            Assert.Equal([1L, 3L], typedTensor.Shape);
            Assert.Equal([1.0f, 2.0f, 3.0f], typedTensor.Value.ToArray());

            Assert.Single(loaded.Graph.Placeholders);
            Assert.Equal("hidden", loaded.Graph.Placeholders[0].Name);

            Assert.Equal(2, loaded.Graph.Nodes.Count);
            Assert.Equal("add_node", loaded.Graph.Nodes[0].Name);
            Assert.Equal(["input", "weights"], loaded.Graph.Nodes[0].Inputs.Select(x => x.Name).ToArray());
            Assert.Equal(["hidden"], loaded.Graph.Nodes[0].Outputs.Select(x => x.Name).ToArray());

            Assert.Equal("identity_node", loaded.Graph.Nodes[1].Name);
            Assert.Equal("CustomIdentity", loaded.Graph.Nodes[1].OpType);
            Assert.Equal("Forwards hidden to output", loaded.Graph.Nodes[1].DocString);
            var labelAttribute = Assert.IsType<OnnxAttribute<string>>(Assert.Single(loaded.Graph.Nodes[1].Attributes));
            Assert.Equal("label", labelAttribute.Name);
            Assert.Equal("final", labelAttribute.Value);
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

public sealed class OnnxGraphTests
{
    [Fact]
    public void AddMembers_WithDuplicateNames_ThrowsInvalidOperationException()
    {
        var graph = OnnxModel.Create().Graph;

        graph.AddInput("value", OnnxTensorType.Create<float>([1]));
        graph.AddTensor("weights", [1], [42.0f]);
        graph.AddEdge("edge");
        graph.AddNode(
            name: "node",
            opType: "Identity",
            domain: "",
            docString: "",
            inputs: [graph.GetValue("value")!],
            outputs: [graph.GetValue("edge")!],
            attributes: []);

        Assert.Throws<InvalidOperationException>(() => graph.AddInput("value", OnnxTensorType.Create<float>([1])));
        Assert.Throws<InvalidOperationException>(() => graph.AddTensor("weights", [1], [13.0f]));
        Assert.Throws<InvalidOperationException>(() => graph.AddEdge("edge"));
        Assert.Throws<InvalidOperationException>(() => graph.AddNode(
            name: "node",
            opType: "Relu",
            domain: "",
            docString: "",
            inputs: [],
            outputs: [],
            attributes: []));
    }
}
