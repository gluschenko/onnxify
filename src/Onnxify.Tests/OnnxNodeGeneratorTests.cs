extern alias SourceGen;

using System.Reflection;
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

    [Fact]
    public void GetLatestOperatorSchemas_PrefersHighestSinceVersionPerDomainAndName()
    {
        var latest = OnnxNodeGenerator.GetLatestOperatorSchemas(
        [
            new OperatorSchema
            {
                Name = "Add",
                Domain = "",
                SinceVersion = 7,
                Doc = "older",
                Attributes = [],
                Inputs = [],
                Outputs = [],
            },
            new OperatorSchema
            {
                Name = "Add",
                Domain = "",
                SinceVersion = 14,
                Doc = "newer",
                Attributes = [],
                Inputs = [],
                Outputs = [],
            },
            new OperatorSchema
            {
                Name = "Normalizer",
                Domain = "ai.onnx.ml",
                SinceVersion = 1,
                Doc = null,
                Attributes = [],
                Inputs = [],
                Outputs = [],
            },
        ]);

        Assert.Equal(2, latest.Count);
        Assert.Contains(latest, x => x.Domain == "" && x.Name == "Add" && x.SinceVersion == 14 && x.Doc == "newer");
        Assert.DoesNotContain(latest, x => x.Domain == "" && x.Name == "Add" && x.SinceVersion == 7);
        Assert.Contains(latest, x => x.Domain == "ai.onnx.ml" && x.Name == "Normalizer" && x.SinceVersion == 1);
    }

    [Fact]
    public void CompatibilityMetadataGenerator_PreservesHistoryAndDerivesLatestFromSameSchemaSet()
    {
        var schemas =
        new[]
        {
            new OperatorSchema
            {
                Name = "Add",
                Domain = "",
                SinceVersion = 1,
                Doc = null,
                Attributes = [],
                Inputs = [],
                Outputs = [],
            },
            new OperatorSchema
            {
                Name = "Add",
                Domain = "",
                SinceVersion = 7,
                Doc = null,
                Attributes = [],
                Inputs = [],
                Outputs = [],
            },
            new OperatorSchema
            {
                Name = "Add",
                Domain = "",
                SinceVersion = 14,
                Doc = null,
                Attributes = [],
                Inputs = [],
                Outputs = [],
            },
            new OperatorSchema
            {
                Name = "TreeEnsemble",
                Domain = "ai.onnx.ml",
                SinceVersion = 3,
                Doc = null,
                Attributes = [],
                Inputs = [],
                Outputs = [],
            },
        };

        var historical = OnnxCompatibilityMetadataGenerator.GetHistoricalOperatorSchemas(schemas);
        var latest = OnnxCompatibilityMetadataGenerator.GetLatestOperatorSchemas(schemas);

        Assert.Equal([1, 7, 14], historical.Where(x => x.Domain == "" && x.Name == "Add").Select(x => x.SinceVersion).ToArray());
        Assert.Equal(2, latest.Count);
        Assert.Contains(latest, x => x.Domain == "" && x.Name == "Add" && x.SinceVersion == 14);
        Assert.DoesNotContain(latest, x => x.Domain == "" && x.Name == "Add" && x.SinceVersion == 7);
        Assert.Contains(latest, x => x.Domain == "ai.onnx.ml" && x.Name == "TreeEnsemble" && x.SinceVersion == 3);
    }

    [Fact]
    public void CompatibilityMetadataGenerator_ComputesStructuralCompatibilityBoundariesFromBundledSchemaHistory()
    {
        var assetPath = Path.Combine(AppContext.BaseDirectory, "Assets", "onnx_operators.json");
        Assert.True(File.Exists(assetPath), $"Expected schema asset at '{assetPath}'.");

        var root = JsonSerializer.Deserialize<OperatorSchemaRoot>(File.ReadAllText(assetPath))
            ?? throw new InvalidOperationException("Failed to load bundled operator schema.");

        var structuralCompatibility = OnnxCompatibilityMetadataGenerator.GetCurrentStructuralCompatibility(root.Operators);

        Assert.Equal(7, structuralCompatibility[("", "Add")]);
        Assert.Equal(4, structuralCompatibility[("", "Concat")]);
        Assert.Equal(18, structuralCompatibility[("", "Split")]);
        Assert.Equal(23, structuralCompatibility[("", "QuantizeLinear")]);
    }

    [Fact]
    public void CompatibilityMetadataGenerator_ToGeneratedSchema_DerivesInputAndOutputAritySeparately()
    {
        var schema = new OperatorSchema
        {
            Name = "ArityCheck",
            Domain = "",
            SinceVersion = 1,
            Doc = null,
            Attributes = [],
            Inputs =
            [
                new OperatorParameter
                {
                    Name = "X",
                    Description = null,
                    Option = FormalParameterOption.Single,
                    MinArity = 1,
                    Type = "T",
                    Types = ["tensor(float)"],
                },
                new OperatorParameter
                {
                    Name = "Y",
                    Description = null,
                    Option = FormalParameterOption.Optional,
                    MinArity = 0,
                    Type = "T",
                    Types = ["tensor(float)"],
                },
            ],
            Outputs =
            [
                new OperatorParameter
                {
                    Name = "Z",
                    Description = null,
                    Option = FormalParameterOption.Single,
                    MinArity = 1,
                    Type = "T",
                    Types = ["tensor(float)"],
                },
            ],
        };

        var toGeneratedSchema = typeof(OnnxCompatibilityMetadataGenerator).GetMethod(
            "ToGeneratedSchema",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(toGeneratedSchema);

        var generatedSchema = toGeneratedSchema!.Invoke(null, [schema]);
        Assert.NotNull(generatedSchema);

        var generatedSchemaType = generatedSchema!.GetType();

        Assert.Equal(1, generatedSchemaType.GetProperty("MinimumInputs")!.GetValue(generatedSchema));
        Assert.Equal(2, generatedSchemaType.GetProperty("MaximumInputs")!.GetValue(generatedSchema));
        Assert.Equal(1, generatedSchemaType.GetProperty("MinimumOutputs")!.GetValue(generatedSchema));
        Assert.Equal(1, generatedSchemaType.GetProperty("MaximumOutputs")!.GetValue(generatedSchema));
    }
}
