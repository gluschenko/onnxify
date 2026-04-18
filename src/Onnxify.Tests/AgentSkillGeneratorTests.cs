using Onnxify.AgentSkillGenerator;
using System.Text.RegularExpressions;

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

        var addMarkdown = files[Path.Combine("ai.onnx", "Add.md")];
        Assert.Contains("(../common/Broadcasting.md)", addMarkdown);
        Assert.DoesNotContain("(Broadcasting.md)", addMarkdown);

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

            foreach (Match match in Regex.Matches(line, @"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)"))
            {
                targets.Add(match.Groups[1].Value.Trim());
            }
        }

        return targets;
    }
}
