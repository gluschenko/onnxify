using Onnxify.AgentSkillGenerator;

namespace Onnxify.Tests;

public sealed class AgentSkillGeneratorTests
{
    [Fact]
    public void BuildGeneratedFiles_IncludesTorchSharpConverterSignatures()
    {
        var files = TorchSharpConverterSkillGenerator.BuildGeneratedFiles();

        Assert.Contains("index.md", files.Keys);
        Assert.Contains(Path.Combine("entry-points", "TorchModule.md"), files.Keys);
        Assert.Contains(Path.Combine("composites", "Sequential.md"), files.Keys);
        Assert.Contains(Path.Combine("torch-ops", "Conv2d.md"), files.Keys);
        Assert.Contains(Path.Combine("torch-ops", "LSTM.md"), files.Keys);

        var indexMarkdown = files["index.md"];
        Assert.Contains("# Onnxify TorchSharp Converter Instructions", indexMarkdown);
        Assert.Contains("Dispatch Entry Points", indexMarkdown);
        Assert.Contains("Torch-Op-Backed Converters", indexMarkdown);
        Assert.Contains("aten::conv2d", indexMarkdown);

        var conv2dMarkdown = files[Path.Combine("torch-ops", "Conv2d.md")];
        Assert.Contains("Conv2d Converter", conv2dMarkdown);
        Assert.Contains("TorchModuleExtensions.Export(this Conv2d module, OnnxGraph graph, IOnnxGraphEdge input) -> IOnnxGraphEdge", conv2dMarkdown);
        Assert.Contains("aten::conv2d", conv2dMarkdown);

        var lstmMarkdown = files[Path.Combine("torch-ops", "LSTM.md")];
        Assert.Contains("LSTM Converter", lstmMarkdown);
        Assert.Contains("LSTMOutput", lstmMarkdown);
        Assert.Contains("aten::lstm.input", lstmMarkdown);
    }
}
