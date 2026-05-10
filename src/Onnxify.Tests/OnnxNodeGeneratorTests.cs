extern alias SourceGen;

using System.Text.Json;
using SourceGen::Onnxify.SourceGenerator;
using SourceGen::Onnxify.SourceGenerator.Models;

namespace Onnxify.Tests;

public class OnnxNodeGeneratorTests
{
    [Fact]
    public void GetParameterComment_UsesCodeFormattingForClosedGenericTypes()
    {
        var parameter = new OperatorParameter
        {
            Name = "X",
            Description = "Input tensor.",
            Option = FormalParameterOption.Single,
            MinArity = 0,
            Type = "T",
            Types = ["tensor(double)", "tensor(float)", "tensor(int32)"],
        };

        var comment = OnnxNodeGenerator.GetParameterComment(parameter);

        Assert.Contains("<c>OnnxTensor&lt;double&gt;</c>", comment);
        Assert.Contains("<c>OnnxTensor&lt;float&gt;</c>", comment);
        Assert.Contains("<c>OnnxTensor&lt;int&gt;</c>", comment);
        Assert.DoesNotContain("cref=\"", comment);
    }

    [Fact]
    public void GetAttributeComment_UsesCodeFormattingForSimpleTypes()
    {
        var attribute = new OperatorAttribute
        {
            Name = "alpha",
            Description = "Scalar attribute.",
            Required = false,
            Type = (int)AttributeType.Float,
            Default = null,
        };

        var comment = OnnxNodeGenerator.GetAttributeComment(attribute);

        Assert.Contains("<c>float</c>", comment);
        Assert.DoesNotContain("cref=\"", comment);
    }

    [Theory]
    [InlineData("sparse_tensor(double)", "OnnxValue<OnnxSparseTensorType>")]
    [InlineData("optional(tensor(float))", "OnnxValue<OnnxOptionalType>")]
    [InlineData("optional(sparse_tensor(int64))", "OnnxValue<OnnxOptionalType>")]
    [InlineData("seq(tensor(float))", "OnnxValue<OnnxSequenceType>")]
    [InlineData("map(int64,tensor(double))", "OnnxValue<OnnxMapType>")]
    [InlineData("optional(seq(tensor(float)))", "OnnxValue<OnnxOptionalType>")]
    [InlineData("seq(map(string,tensor(float)))", "OnnxValue<OnnxSequenceType>")]
    public void MapType_MapsExpandedConcreteTypes(string type, string expected)
    {
        var mappedType = OnnxNodeGenerator.MapType(type);

        Assert.Equal(expected, mappedType);
    }

    [Fact]
    public void MapType_CoversAllConcreteParameterTypesInBundledSchema()
    {
        var assetPath = Path.Combine(AppContext.BaseDirectory, "Assets", "onnx_operators.json");
        Assert.True(File.Exists(assetPath), $"Expected schema asset at '{assetPath}'.");

        var root = JsonSerializer.Deserialize<OperatorSchemaRoot>(File.ReadAllText(assetPath))
            ?? throw new InvalidOperationException("Failed to load bundled operator schema.");

        var unsupportedTypes = root.Operators
            .SelectMany(x => x.Inputs.Concat(x.Outputs))
            .SelectMany(x => x.Types)
            .Where(x => x.Contains('('))
            .Distinct(StringComparer.Ordinal)
            .Where(x => OnnxNodeGenerator.MapType(x) == x)
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToArray();

        Assert.True(
            unsupportedTypes.Length == 0,
            $"Unmapped concrete ONNX types: {string.Join(", ", unsupportedTypes)}");
    }
}
