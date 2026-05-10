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
    public void Save_AndLoad_RoundTripsGraphNameAndCustomOpsetImports()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "graph-name-tests",
            IrVersion = 9,
            Opset = 13,
        });

        model.Graph.Name = "bvlc_alexnet";
        model.ClearOpsetImports();
        model.SetOpsetImport("", 13);
        model.SetOpsetImport("ai.onnx.ml", 2);
        model.SetOpsetImport("com.microsoft", 1);

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            model.Save(path);
            var loaded = OnnxModel.FromFile(path);

            Assert.Equal("bvlc_alexnet", loaded.Graph.Name);
            Assert.Collection(
                loaded.OpsetImport,
                x =>
                {
                    Assert.Equal(string.Empty, x.Domain);
                    Assert.Equal(13, x.Version);
                },
                x =>
                {
                    Assert.Equal("ai.onnx.ml", x.Domain);
                    Assert.Equal(2, x.Version);
                },
                x =>
                {
                    Assert.Equal("com.microsoft", x.Domain);
                    Assert.Equal(1, x.Version);
                });
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
            Assert.NotNull(loadedInput.Type.Shape);
            var loadedInputShape = loadedInput.Type.Shape!;
            Assert.Equal(2, loadedInputShape.Dimensions.Length);
            Assert.Equal(1L, Assert.IsType<OnnxDimension<long>>(loadedInputShape.Dimensions[0]).Value);
            Assert.Equal(3L, Assert.IsType<OnnxDimension<long>>(loadedInputShape.Dimensions[1]).Value);

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

    [Fact]
    public void Save_AndLoad_RoundTripsVariadicInputsAndOutputs()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            Opset = 13,
        });

        var left = model.Graph.AddInput("left", OnnxTensorType.Create<float>([1, 2]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<float>([1, 2]));
        var concatEdge = model.Graph.AddEdge("concat_edge");

        var concat = new Concat("concat", new ConcatInputOutputOptions
        {
            In = [left, right],
            Axis = 0,
            ConcatResult = concatEdge,
        });

        model.Graph.AddNode(concat);

        var splitInput = model.Graph.AddInput("split_input", OnnxTensorType.Create<float>([2, 2]));
        var splitLeft = model.Graph.AddEdge("split_left");
        var splitRight = model.Graph.AddEdge("split_right");

        var splitOutputs = model.Graph.Split("split", new SplitInputOutputOptions
        {
            Input = splitInput,
            Axis = 0,
            Out = [splitLeft, splitRight],
        });

        Assert.Equal(["left", "right"], concat.In.Select(x => x.Name).ToArray());
        Assert.Equal("concat_edge", concat.ConcatResult.Name);
        Assert.Equal(["split_left", "split_right"], splitOutputs.Select(x => x.Name).ToArray());

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            model.Save(path);
            var loaded = OnnxModel.FromFile(path);

            Assert.Equal(2, loaded.Graph.Nodes.Count);

            var loadedConcat = Assert.IsType<Concat>(loaded.Graph.Nodes[0]);
            Assert.Equal(["left", "right"], loadedConcat.In.Select(x => x.Name).ToArray());
            Assert.Equal("concat_edge", loadedConcat.ConcatResult.Name);

            var loadedSplit = Assert.IsType<Split>(loaded.Graph.Nodes[1]);
            Assert.Equal("split_input", loadedSplit.Input.Name);
            Assert.Equal(["split_left", "split_right"], loadedSplit.Out.Select(x => x.Name).ToArray());
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
    public void Save_AndLoad_RoundTripsNonTensorGraphValueTypes()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "non-tensor-type-tests",
            IrVersion = 9,
            Opset = 13,
        });

        model.Graph.AddInput(
            "sequence_input",
            new OnnxSequenceType(
                OnnxTensorType.Create<long>(new OnnxDimension[] { "tokens" }),
                "sequence-items"));
        model.Graph.AddInput(
            "map_input",
            new OnnxMapType(
                typeof(string),
                OnnxTensorType.Create<float>(new OnnxDimension[] { 1L }),
                "lookup"));
        model.Graph.AddInput(
            "opaque_input",
            new OnnxOpaqueType(
                "ai.onnx.ml",
                "Vocabulary",
                "opaque-metadata"));
        model.Graph.AddValue(
            "sparse_hidden",
            new OnnxSparseTensorType(
                typeof(float),
                OnnxTensorShape.Create(new OnnxDimension[] { 2L, "nnz" }),
                "sparse-values"));
        model.Graph.AddOutput(
            "optional_output",
            new OnnxOptionalType(
                OnnxTensorType.Create<bool>(new OnnxDimension[] { 1L }),
                "optional-flag"));

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");

        try
        {
            model.Save(path);
            var loaded = OnnxModel.FromFile(path);

            Assert.Equal(3, loaded.Graph.Inputs.Count);

            var loadedSequence = Assert.IsType<OnnxValue<OnnxSequenceType>>(loaded.Graph.Inputs[0]);
            Assert.Equal("sequence_input", loadedSequence.Name);
            Assert.Equal("sequence-items", loadedSequence.Type.Denotation);
            var sequenceElementType = Assert.IsType<OnnxTensorType>(loadedSequence.Type.ElementType);
            Assert.Equal(typeof(long), sequenceElementType.Type);
            Assert.NotNull(sequenceElementType.Shape);
            Assert.Equal("tokens", Assert.IsType<OnnxDimension<string>>(sequenceElementType.Shape!.Dimensions[0]).Value);

            var loadedMap = Assert.IsType<OnnxValue<OnnxMapType>>(loaded.Graph.Inputs[1]);
            Assert.Equal("lookup", loadedMap.Type.Denotation);
            Assert.Equal(typeof(string), loadedMap.Type.KeyType);
            var mapValueType = Assert.IsType<OnnxTensorType>(loadedMap.Type.ValueType);
            Assert.Equal(typeof(float), mapValueType.Type);

            var loadedOpaque = Assert.IsType<OnnxValue<OnnxOpaqueType>>(loaded.Graph.Inputs[2]);
            Assert.Equal("opaque-metadata", loadedOpaque.Type.Denotation);
            Assert.Equal("ai.onnx.ml", loadedOpaque.Type.Domain);
            Assert.Equal("Vocabulary", loadedOpaque.Type.Name);

            var loadedSparse = Assert.IsType<OnnxValue<OnnxSparseTensorType>>(Assert.Single(loaded.Graph.Placeholders));
            Assert.Equal("sparse-values", loadedSparse.Type.Denotation);
            Assert.Equal(typeof(float), loadedSparse.Type.Type);
            Assert.NotNull(loadedSparse.Type.Shape);
            Assert.Equal(2L, Assert.IsType<OnnxDimension<long>>(loadedSparse.Type.Shape!.Dimensions[0]).Value);
            Assert.Equal("nnz", Assert.IsType<OnnxDimension<string>>(loadedSparse.Type.Shape!.Dimensions[1]).Value);

            var loadedOptional = Assert.IsType<OnnxValue<OnnxOptionalType>>(Assert.Single(loaded.Graph.Outputs));
            Assert.Equal("optional-flag", loadedOptional.Type.Denotation);
            var optionalElementType = Assert.IsType<OnnxTensorType>(loadedOptional.Type.ElementType);
            Assert.Equal(typeof(bool), optionalElementType.Type);
            Assert.NotNull(optionalElementType.Shape);
            Assert.Equal(1L, Assert.IsType<OnnxDimension<long>>(optionalElementType.Shape!.Dimensions[0]).Value);
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
