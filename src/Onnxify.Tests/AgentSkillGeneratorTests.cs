using System.Text.RegularExpressions;
using Onnxify.AgentSkillGenerator;

namespace Onnxify.Tests;

public sealed class AgentSkillGeneratorTests
{
    [Fact]
    public void BuildGeneratedFiles_IncludesTorchSharpConverterSignatures()
    {
        var files = TorchSharpConverterSkillGenerator.BuildGeneratedFiles();

        Assert.Contains("index.md", files.Keys);
        Assert.Contains(Path.Combine("composites", "Sequential.md"), files.Keys);
        Assert.Contains(Path.Combine("composites", "torch.nn.Module_torch.Tensor__torch.Tensor_.md"), files.Keys);
        Assert.Contains(Path.Combine("torch-ops", "Conv2d.md"), files.Keys);
        Assert.Contains(Path.Combine("torch-ops", "LSTM.md"), files.Keys);
        Assert.Contains(Path.Combine("torch-ops", "OnnxGraph__aten__bmm.md"), files.Keys);

        var indexMarkdown = files["index.md"];
        Assert.Contains("# Onnxify TorchSharp Converter Instructions", indexMarkdown);
        Assert.Contains("Composite Converters", indexMarkdown);
        Assert.Contains("Torch-Op-Backed Converters", indexMarkdown);
        Assert.Contains("aten::conv2d", indexMarkdown);
        Assert.Contains("src/Onnxify.TorchSharp/TorchTensorOperatorExtensions.cs", indexMarkdown);

        var moduleMarkdown = files[Path.Combine("composites", "torch.nn.Module_torch.Tensor__torch.Tensor_.md")];
        Assert.Contains("torch.nn.Module<torch.Tensor, torch.Tensor> Converter", moduleMarkdown);
        Assert.Contains("Onnxify.TorchSharp.TorchModuleExtensions.Export(this torch.nn.Module<torch.Tensor, torch.Tensor> module, OnnxGraph graph, IOnnxGraphEdge input) -> IOnnxGraphEdge", moduleMarkdown);

        var conv2dMarkdown = files[Path.Combine("torch-ops", "Conv2d.md")];
        Assert.Contains("Conv2d Converter", conv2dMarkdown);
        Assert.Contains("TorchModuleExtensions.Export(this Conv2d module, OnnxGraph graph, IOnnxGraphEdge input) -> IOnnxGraphEdge", conv2dMarkdown);
        Assert.Contains("aten::conv2d", conv2dMarkdown);

        var matmulMarkdown = files[Path.Combine("torch-ops", "OnnxGraph__aten__bmm.md")];
        Assert.Contains("OnnxGraph Converter", matmulMarkdown);
        Assert.Contains("TorchTensorOperatorExtensions.ExportMatMul(this OnnxGraph graph, IOnnxGraphEdge input, IOnnxGraphEdge other) -> IOnnxGraphEdge", matmulMarkdown);
        Assert.Contains("aten::matmul", matmulMarkdown);
        Assert.Contains("src/Onnxify.TorchSharp/TorchTensorOperatorExtensions.cs", matmulMarkdown);

        var lstmMarkdown = files[Path.Combine("torch-ops", "LSTM.md")];
        Assert.Contains("LSTM Converter", lstmMarkdown);
        Assert.Contains("LSTMOutput", lstmMarkdown);
        Assert.Contains("aten::lstm.input", lstmMarkdown);
    }

    [Fact]
    public void BuildGeneratedFiles_ForTorchSharpConverters_PreservesExistingTorchOpSlug()
    {
        var existingRelativePaths = new HashSet<string>(StringComparer.Ordinal)
        {
            Path.Combine("torch-ops", "OnnxGraph___operator__add.md"),
            Path.Combine("torch-ops", "OnnxGraph__aten__bitwise_and.Tensor.md"),
        };

        var files = TorchSharpConverterSkillGenerator.BuildGeneratedFiles(existingRelativePaths);

        Assert.Contains(Path.Combine("torch-ops", "OnnxGraph___operator__add.md"), files.Keys);
        Assert.DoesNotContain(Path.Combine("torch-ops", "OnnxGraph__aten__add.Tensor.md"), files.Keys);
        Assert.Contains(Path.Combine("torch-ops", "OnnxGraph__aten__bitwise_and.Tensor.md"), files.Keys);
        Assert.DoesNotContain(Path.Combine("torch-ops", "OnnxGraph___operator__and_.md"), files.Keys);
    }

    [Fact]
    public void BuildGeneratedFiles_ForOperators_EmitsSharedReferencesAndTableOfContents()
    {
        var files = OperatorSkillGenerator.BuildGeneratedFiles();

        Assert.Contains("index.md", files.Keys);
        Assert.Contains(Path.Combine("common", "Broadcasting.md"), files.Keys);
        Assert.Contains(Path.Combine("common", "IR.md"), files.Keys);
        Assert.Contains(Path.Combine("ai.onnx", "Add.md"), files.Keys);

        var indexMarkdown = files["index.md"];
        Assert.Contains("# Onnxify Operator Instructions", indexMarkdown);
        Assert.Contains("## Table of Contents", indexMarkdown);
        Assert.Contains("- `ai.onnx` - ", indexMarkdown);
        Assert.Contains("ModelGenerator TorchModule", indexMarkdown);
        Assert.Contains("- Operators with at least one Onnxify.ModelGenerator TorchModule path: `", indexMarkdown);

        var addMarkdown = files[Path.Combine("ai.onnx", "Add.md")];
        Assert.Contains("- Onnxify.ModelGenerator TorchModule coverage: `not detected`", addMarkdown);
        Assert.Contains("(../common/Broadcasting.md)", addMarkdown);
        Assert.DoesNotContain("(Broadcasting.md)", addMarkdown);

        var convMarkdown = files[Path.Combine("ai.onnx", "Conv.md")];
        Assert.Contains("- Onnxify.ModelGenerator TorchModule coverage: `available`", convMarkdown);
        Assert.Contains("Conv2dTorchModuleOperator", convMarkdown);

        var batchNormalizationMarkdown = files[Path.Combine("ai.onnx", "BatchNormalization.md")];
        Assert.Contains("(../common/IR.md)", batchNormalizationMarkdown);
        Assert.DoesNotContain("(IR.md)", batchNormalizationMarkdown);
    }

    [Fact]
    public void BuildGeneratedFiles_ForOperators_DoesNotContainBrokenRelativeMarkdownLinks()
    {
        var files = OperatorSkillGenerator.BuildGeneratedFiles();
        string tempRoot = Path.Combine(AppContext.BaseDirectory, "AgentSkillGeneratorTestOutput", Guid.NewGuid().ToString("N"));

        try
        {
            foreach ((string relativePath, string content) in files)
            {
                string fullPath = Path.Combine(tempRoot, relativePath);
                Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
                File.WriteAllText(fullPath, content);
            }

            foreach ((string relativePath, string content) in files.Where(static x => x.Key.EndsWith(".md", StringComparison.Ordinal)))
            {
                string fullPath = Path.Combine(tempRoot, relativePath);
                foreach (string linkTarget in ExtractMarkdownTargets(content))
                {
                    if (Uri.TryCreate(linkTarget, UriKind.Absolute, out _))
                    {
                        continue;
                    }

                    string resolvedPath = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(fullPath)!, linkTarget));
                    Assert.True(File.Exists(resolvedPath), $"Missing markdown link target '{linkTarget}' from '{relativePath}'.");
                }
            }
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    private static IReadOnlyList<string> ExtractMarkdownTargets(string markdown)
    {
        var targets = new List<string>();
        bool inCodeFence = false;

        foreach (string line in markdown.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n'))
        {
            if (line.StartsWith("```", StringComparison.Ordinal))
            {
                inCodeFence = !inCodeFence;
                continue;
            }

            if (inCodeFence)
            {
                continue;
            }

            if (line.StartsWith("    ", StringComparison.Ordinal) || line.StartsWith('\t'))
            {
                continue;
            }

            foreach (Match match in Regex.Matches(line, @"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)"))
            {
                targets.Add(match.Groups[1].Value.Trim());
            }
        }

        return targets;
    }
}
