using Google.Protobuf;
using Onnx;
using Onnxify.ModelGenerator;

namespace Onnxify.Tests;

public sealed class OnnxModelMetadataReaderTests
{
    [Fact]
    public void ReadModel_ParsesTensorInputsOutputsAndInitializerMetadata()
    {
        var bytes = CreateModelBytes(model =>
        {
            model.Graph.AddInput(
                "input_ids",
                OnnxTensorType.Create<long>(new OnnxDimension[] { 1L, "sequence_length" }, "tokens"));
            model.Graph.AddOutput(
                "logits",
                OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, "sequence_length", 128L }, "scores"));
            model.Graph.AddTensor("weights", [128L], Enumerable.Range(0, 128).Select(static x => (float)x).ToArray());
        });

        var parsed = OnnxModelMetadataReader.ReadModel(bytes);

        Assert.Single(parsed.Graph.Initializers);
        Assert.Equal("weights", parsed.Graph.Initializers[0].Name);
        Assert.False(parsed.Graph.Initializers[0].HasExternalData);

        var input = Assert.Single(parsed.Graph.Inputs);
        Assert.Equal("input_ids", input.Name);
        Assert.Equal(OnnxValueKind.Tensor, input.Type.Kind);
        Assert.Equal("tokens", input.Type.Denotation);
        var inputTensorType = Assert.IsType<ParsedOnnxTensorType>(input.Type.TensorType);
        Assert.Equal(OnnxTensorDataType.Int64, inputTensorType.ElementType);
        Assert.Collection(
            inputTensorType.Shape,
            dimension =>
            {
                Assert.Equal(1L, dimension.NumericValue);
                Assert.Null(dimension.SymbolicName);
            },
            dimension =>
            {
                Assert.Null(dimension.NumericValue);
                Assert.Equal("sequence_length", dimension.SymbolicName);
            });

        var output = Assert.Single(parsed.Graph.Outputs);
        Assert.Equal("logits", output.Name);
        Assert.Equal(OnnxValueKind.Tensor, output.Type.Kind);
        Assert.Equal("scores", output.Type.Denotation);
        var outputTensorType = Assert.IsType<ParsedOnnxTensorType>(output.Type.TensorType);
        Assert.Equal(OnnxTensorDataType.Float, outputTensorType.ElementType);
        Assert.Collection(
            outputTensorType.Shape,
            dimension =>
            {
                Assert.Equal(1L, dimension.NumericValue);
                Assert.Null(dimension.SymbolicName);
            },
            dimension =>
            {
                Assert.Null(dimension.NumericValue);
                Assert.Equal("sequence_length", dimension.SymbolicName);
            },
            dimension =>
            {
                Assert.Equal(128L, dimension.NumericValue);
                Assert.Null(dimension.SymbolicName);
            });
    }

    [Fact]
    public void ReadModel_ParsesUnknownDimensionsAndExternalInitializerFlags()
    {
        var bytes = CreateMutatedModelBytes(proto =>
        {
            var input = proto.Graph.Input[0];
            input.Type = new TypeProto
            {
                Denotation = "dynamic",
                TensorType = new TypeProto.Types.Tensor
                {
                    ElemType = (int)TensorProto.Types.DataType.Float,
                    Shape = new TensorShapeProto
                    {
                        Dim =
                        {
                            new TensorShapeProto.Types.Dimension(),
                            new TensorShapeProto.Types.Dimension { DimParam = "batch_size" },
                            new TensorShapeProto.Types.Dimension { DimValue = 224L },
                        }
                    }
                }
            };

            var initializer = proto.Graph.Initializer[0];
            initializer.DataLocation = TensorProto.Types.DataLocation.External;
            initializer.ExternalData.Add(new StringStringEntryProto
            {
                Key = "location",
                Value = "weights.bin",
            });
        });

        var parsed = OnnxModelMetadataReader.ReadModel(bytes);

        var input = Assert.Single(parsed.Graph.Inputs);
        Assert.Equal("dynamic", input.Type.Denotation);
        var tensorType = Assert.IsType<ParsedOnnxTensorType>(input.Type.TensorType);
        Assert.Collection(
            tensorType.Shape,
            dimension =>
            {
                Assert.Null(dimension.NumericValue);
                Assert.Null(dimension.SymbolicName);
            },
            dimension =>
            {
                Assert.Null(dimension.NumericValue);
                Assert.Equal("batch_size", dimension.SymbolicName);
            },
            dimension =>
            {
                Assert.Equal(224L, dimension.NumericValue);
                Assert.Null(dimension.SymbolicName);
            });

        var initializer = Assert.Single(parsed.Graph.Initializers);
        Assert.Equal("weights", initializer.Name);
        Assert.True(initializer.HasExternalData);
    }

    [Fact]
    public void ReadModel_NullData_ThrowsArgumentNullException()
    {
        var exception = Assert.Throws<ArgumentNullException>(() => OnnxModelMetadataReader.ReadModel(null!));

        Assert.Equal("data", exception.ParamName);
    }

    [Fact]
    public void ReadModel_InvalidProtobuf_ThrowsInvalidDataException()
    {
        var bytes = new byte[] { 0x3A, 0x02, 0x0A };

        Assert.Throws<InvalidDataException>(() => OnnxModelMetadataReader.ReadModel(bytes));
    }

    [Fact]
    public void ReadModel_ModelWithoutGraph_ReturnsEmptyGraph()
    {
        var bytes = new ModelProto().ToByteArray();

        var parsed = OnnxModelMetadataReader.ReadModel(bytes);

        Assert.Same(ParsedOnnxGraph.Empty, parsed.Graph);
        Assert.Empty(parsed.Graph.Initializers);
        Assert.Empty(parsed.Graph.Inputs);
        Assert.Empty(parsed.Graph.Outputs);
    }

    private static byte[] CreateModelBytes(Action<OnnxModel> configure)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "metadata-reader-tests",
            IrVersion = 9,
            Opset = 13,
        });

        configure(model);
        return model.ToProto().ToByteArray();
    }

    private static byte[] CreateMutatedModelBytes(Action<ModelProto> mutate)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "metadata-reader-tests",
            IrVersion = 9,
            Opset = 13,
        });

        model.Graph.AddInput("input", OnnxTensorType.Create<float>([1L, 3L, 224L, 224L]));
        model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1L, 1000L]));
        model.Graph.AddTensor("weights", [1L], [42.0f]);

        var proto = model.ToProto();
        mutate(proto);
        return proto.ToByteArray();
    }
}
