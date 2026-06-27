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
        Assert.Contains("## Package Versions", indexMarkdown);
        Assert.Contains("[Full package versions and dependencies](../packages.md)", indexMarkdown);
        Assert.DoesNotContain("`ICSharpCode.Decompiler` `10.0.1.8346`", indexMarkdown);

        var moduleMarkdown = files[Path.Combine("composites", "torch.nn.Module_torch.Tensor__torch.Tensor_.md")];
        Assert.Contains("torch.nn.Module<torch.Tensor, torch.Tensor> Converter", moduleMarkdown);
        Assert.Contains("Onnxify.TorchSharp.TorchModuleExtensions.Export(this torch.nn.Module<torch.Tensor, torch.Tensor> module, OnnxGraph graph, IOnnxGraphEdge input) -> IOnnxGraphEdge", moduleMarkdown);

        var conv2dMarkdown = files[Path.Combine("torch-ops", "Conv2d.md")];
        Assert.Contains("Conv2d Converter", conv2dMarkdown);
        Assert.Contains("TorchModuleExtensions.Export(this Conv2d module, OnnxGraph graph, IOnnxGraphEdge input) -> IOnnxGraphEdge", conv2dMarkdown);
        Assert.Contains("aten::conv2d", conv2dMarkdown);
        Assert.Contains("## Package Versions", conv2dMarkdown);
        Assert.Contains("[Full package versions and dependencies](../../packages.md)", conv2dMarkdown);
        Assert.Contains("| `Onnxify` | `0.3.6.2` |", conv2dMarkdown);
        Assert.Contains("| `Onnxify.TorchSharp` | `0.3.6.2` |", conv2dMarkdown);
        Assert.DoesNotContain("`TorchSharp` `0.107.0`", conv2dMarkdown);

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
        Assert.Contains("## Package Versions", indexMarkdown);
        Assert.Contains("[Full package versions and dependencies](../packages.md)", indexMarkdown);
        Assert.DoesNotContain("`Google.Protobuf` `3.34.0`", indexMarkdown);

        var addMarkdown = files[Path.Combine("ai.onnx", "Add.md")];
        Assert.Contains("- Onnxify.ModelGenerator TorchModule coverage: `available`", addMarkdown);
        Assert.Contains("AddTorchModuleInlineOperator", addMarkdown);
        Assert.Contains("(../common/Broadcasting.md)", addMarkdown);
        Assert.DoesNotContain("(Broadcasting.md)", addMarkdown);
        Assert.Contains("## Package Versions", addMarkdown);
        Assert.Contains("[Full package versions and dependencies](../../packages.md)", addMarkdown);
        Assert.Contains("| `Onnxify` | `0.3.6.2` |", addMarkdown);
        Assert.Contains("| `Onnxify.TorchSharp` | `0.3.6.2` |", addMarkdown);
        Assert.DoesNotContain("| `Onnxify.ModelGenerator` | `0.3.6.2` |", addMarkdown);
        Assert.DoesNotContain("`Microsoft.CodeAnalysis.CSharp` `4.11.0`", addMarkdown);

        var convMarkdown = files[Path.Combine("ai.onnx", "Conv.md")];
        Assert.Contains("- Onnxify.ModelGenerator TorchModule coverage: `available`", convMarkdown);
        Assert.Contains("Conv2dTorchModuleOperator", convMarkdown);

        var mlOperatorMarkdown = files[Path.Combine("ai.onnx.ml", "ArrayFeatureExtractor.md")];
        Assert.Contains("| `Onnxify` | `0.3.6.2` |", mlOperatorMarkdown);
        Assert.DoesNotContain("| `Onnxify.TorchSharp` | `0.3.6.2` |", mlOperatorMarkdown);

        var batchNormalizationMarkdown = files[Path.Combine("ai.onnx", "BatchNormalization.md")];
        Assert.Contains("(../common/IR.md)", batchNormalizationMarkdown);
        Assert.DoesNotContain("(IR.md)", batchNormalizationMarkdown);
    }

    [Fact]
    public void BuildGeneratedFiles_ForOperators_DoesNotContainBrokenRelativeMarkdownLinks()
    {
        var files = OperatorSkillGenerator.BuildGeneratedFiles();
        string repoRoot = FindRepositoryRoot();
        var packageInventory = PackageInventory.Load(repoRoot);
        string tempParent = Path.Combine(AppContext.BaseDirectory, "AgentSkillGeneratorTestOutput", Guid.NewGuid().ToString("N"));
        string tempRoot = Path.Combine(tempParent, "operators");

        try
        {
            string packagePath = Path.Combine(tempParent, "packages.md");
            Directory.CreateDirectory(Path.GetDirectoryName(packagePath)!);
            File.WriteAllText(packagePath, packageInventory.BuildFullMarkdown());

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
            if (Directory.Exists(tempParent))
            {
                Directory.Delete(tempParent, recursive: true);
            }
        }
    }

    [Fact]
    public void PackageInventory_BuildFullMarkdown_IncludesVersionsAndThirdPartyDependencies()
    {
        string repoRoot = FindRepositoryRoot();
        var packageInventory = PackageInventory.Load(repoRoot);

        string markdown = packageInventory.BuildFullMarkdown();

        Assert.Contains("# Onnxify Package Versions And Dependencies", markdown);
        Assert.Contains("| `Onnxify` | `0.3.6.2` |", markdown);
        Assert.Contains("| `Onnxify.TorchSharp` | `0.3.6.2` |", markdown);
        Assert.Contains("`Google.Protobuf` `3.34.0`", markdown);
        Assert.Contains("`ICSharpCode.Decompiler` `10.0.1.8346`", markdown);
        Assert.Contains("`TorchSharp` `0.107.0`", markdown);
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

    private static string FindRepositoryRoot()
    {
        return SkillGeneratorPaths.FindRepositoryRoot(Directory.GetCurrentDirectory())
            ?? SkillGeneratorPaths.FindRepositoryRoot(AppContext.BaseDirectory)
            ?? throw new DirectoryNotFoundException("Repository root was not found.");
    }
}
