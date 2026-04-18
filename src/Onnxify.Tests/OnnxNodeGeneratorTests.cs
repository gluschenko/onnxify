extern alias SourceGen;

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
            Type = (int)SourceGen::AttributeType.Float,
            Default = null,
        };

        var comment = OnnxNodeGenerator.GetAttributeComment(attribute);

        Assert.Contains("<c>float</c>", comment);
        Assert.DoesNotContain("cref=\"", comment);
    }
}
