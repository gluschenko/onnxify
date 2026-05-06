using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;
using Onnxify.ModelGenerator;

namespace Onnxify.Tests;

public sealed class OnnxModelGeneratorTests
{
    [Fact]
    public void Generate_ForAdditionalOnnxFile_ProducesTypedWrapper()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "Models", "sample-classifier.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

        try
        {
            CreateTensorModel(
                modelPath: modelPath,
                inputName: "input_ids",
                inputType: OnnxTensorType.Create<long>(new OnnxDimension[] { 1L, "sequence_length" }, "token_ids"),
                outputName: "logits",
                outputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, "sequence_length", 128L }, "class_scores"));

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                });

            var compilation = CreateCompilation();
            driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

            Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
            Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("namespace Demo.App", generatedSource);
            Assert.Contains("public sealed class SampleClassifierModel", generatedSource);
            Assert.Contains("/// <summary>", generatedSource);
            Assert.Contains("Provides a typed ONNX Runtime wrapper for the model file 'sample-classifier.onnx'.", generatedSource);
            Assert.Contains("public sealed class SampleClassifierModelInputs", generatedSource);
            Assert.Contains("Input property <c>InputIds</c> maps to ONNX name <c>input_ids</c>; tensor type <c>Tensor&lt;long&gt;</c>; shape <c>[1, sequence_length]</c>; denotation <c>token_ids</c>", generatedSource);
            Assert.Contains("public sealed class SampleClassifierModelOutputs", generatedSource);
            Assert.Contains("Output property <c>Logits</c> maps to ONNX name <c>logits</c>; tensor type <c>Tensor&lt;float&gt;</c>; shape <c>[1, sequence_length, 128]</c>; denotation <c>class_scores</c>", generatedSource);
            Assert.Contains("public Tensor<long>? InputIds", generatedSource);
            Assert.Contains("Gets or sets the tensor supplied for model input 'input_ids'.", generatedSource);
            Assert.Contains("Tensor type: <c>Tensor&lt;long&gt;</c>", generatedSource);
            Assert.Contains("Element type: <c>long</c>", generatedSource);
            Assert.Contains("Shape: <c>[1, sequence_length]</c>", generatedSource);
            Assert.Contains("Denotation: <c>token_ids</c>", generatedSource);
            Assert.Contains("public Tensor<float> Logits => GetTensor<float>(\"logits\")", generatedSource);
            Assert.Contains("Gets the tensor returned for model output 'logits'.", generatedSource);
            Assert.Contains("Tensor type: <c>Tensor&lt;float&gt;</c>", generatedSource);
            Assert.Contains("Shape: <c>[1, sequence_length, 128]</c>", generatedSource);
            Assert.Contains("Denotation: <c>class_scores</c>", generatedSource);
            Assert.Contains("NamedOnnxValue.CreateFromTensor(\"input_ids\"", generatedSource);
            Assert.Contains("ModelProjectRelativePath = @\"Models\\sample-classifier.onnx\"", generatedSource);
            Assert.Contains("public static IReadOnlyList<Onnxify.OnnxValue> Inputs { get; } = CreateInputs();", generatedSource);
            Assert.Contains("public static IReadOnlyList<Onnxify.OnnxValue> Outputs { get; } = CreateOutputs();", generatedSource);
            Assert.Contains("<c>input_ids</c>: <c>Tensor&lt;long&gt;</c>, shape <c>[1, sequence_length]</c>, denotation <c>token_ids</c>", generatedSource);
            Assert.Contains("<c>logits</c>: <c>Tensor&lt;float&gt;</c>, shape <c>[1, sequence_length, 128]</c>, denotation <c>class_scores</c>", generatedSource);
            Assert.Contains("/// <param name=\"inputIds\">Tensor value for model input <c>input_ids</c>; parameter type <c>Tensor&lt;long&gt;</c>; shape <c>[1, sequence_length]</c>; denotation <c>token_ids</c></param>", generatedSource);
            Assert.Contains("public SampleClassifierModel()", generatedSource);
            Assert.Contains("public SampleClassifierModel(SessionOptions? sessionOptions)", generatedSource);
            Assert.Contains("new Onnxify.OnnxValue<Onnxify.OnnxTensorType>(", generatedSource);
            Assert.Contains("Onnxify.OnnxTensorType.Create<long>(", generatedSource);
            Assert.Contains("\"sequence_length\"", generatedSource);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_RespectsAdditionalFileMetadataOverrides()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelPath = Path.Combine(tempRoot, "override-me.onnx");

        try
        {
            CreateTensorModel(
                modelPath: modelPath,
                inputName: "pixel_values",
                inputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 3L, 224L, 224L }),
                outputName: "scores",
                outputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 1000L }));

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Ignored.Root.Namespace",
                },
                fileOptions: new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)
                {
                    [modelPath] = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["build_metadata.AdditionalFiles.OnnxifyModelClassName"] = "VisionWrapper",
                        ["build_metadata.AdditionalFiles.OnnxifyModelNamespace"] = "Demo.Custom.Models",
                    }
                });

            var compilation = CreateCompilation();
            driver = driver.RunGenerators(compilation);

            var generatedSource = GetGeneratedSource(driver);
            Assert.Contains("namespace Demo.Custom.Models", generatedSource);
            Assert.Contains("public sealed class VisionWrapperModel", generatedSource);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void Generate_ReportsDuplicateTypeNames()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "OnnxModelGeneratorTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        var modelAPath = Path.Combine(tempRoot, "A", "duplicate.onnx");
        var modelBPath = Path.Combine(tempRoot, "B", "duplicate.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(modelAPath)!);
        Directory.CreateDirectory(Path.GetDirectoryName(modelBPath)!);

        try
        {
            CreateTensorModel(
                modelPath: modelAPath,
                inputName: "input",
                inputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 4L }),
                outputName: "output",
                outputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 4L }));

            CreateTensorModel(
                modelPath: modelBPath,
                inputName: "input",
                inputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 4L }),
                outputName: "output",
                outputType: OnnxTensorType.Create<float>(new OnnxDimension[] { 1L, 4L }));

            var driver = CreateDriver(
                additionalFiles: [new BinaryAdditionalText(modelAPath), new BinaryAdditionalText(modelBPath)],
                globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                    ["build_property.RootNamespace"] = "Demo.App",
                });

            var compilation = CreateCompilation();
            driver = driver.RunGenerators(compilation);

            var diagnostics = driver.GetRunResult().Diagnostics;
            Assert.Contains(diagnostics, static x => x.Id == "OMG003");
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    private static void CreateTensorModel(
        string modelPath,
        string inputName,
        OnnxTensorType inputType,
        string outputName,
        OnnxTensorType outputType)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions
        {
            ProducerName = "generator-tests",
            IrVersion = 9,
            Opset = 13,
        });

        model.Graph.AddInput(inputName, inputType);
        model.Graph.AddOutput(outputName, outputType);
        model.Save(modelPath, overwrite: true);
    }

    private static CSharpCompilation CreateCompilation()
    {
        var syntaxTree = CSharpSyntaxTree.ParseText("""
            namespace Demo;

            public static class Marker
            {
            }
            """);

        var trustedPlatformAssemblies = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))?
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Distinct(StringComparer.Ordinal)
            .Select(static path => MetadataReference.CreateFromFile(path))
            .ToList() ?? [];

        trustedPlatformAssemblies.Add(MetadataReference.CreateFromFile(typeof(global::Microsoft.ML.OnnxRuntime.InferenceSession).Assembly.Location));
        trustedPlatformAssemblies.Add(MetadataReference.CreateFromFile(typeof(OnnxModel).Assembly.Location));

        return CSharpCompilation.Create(
            assemblyName: "GeneratedModelTests",
            syntaxTrees: [syntaxTree],
            references: trustedPlatformAssemblies,
            options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
    }

    private static GeneratorDriver CreateDriver(
        IReadOnlyList<AdditionalText> additionalFiles,
        IReadOnlyDictionary<string, string> globalOptions,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>? fileOptions = null)
    {
        return CSharpGeneratorDriver.Create(
            generators: [new OnnxModelGenerator().AsSourceGenerator()],
            additionalTexts: additionalFiles,
            parseOptions: (CSharpParseOptions)CSharpParseOptions.Default.WithLanguageVersion(LanguageVersion.Latest),
            optionsProvider: new TestAnalyzerConfigOptionsProvider(
                globalOptions,
                fileOptions ?? new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)));
    }

    private static string GetGeneratedSource(GeneratorDriver driver)
    {
        var generatedSources = driver.GetRunResult()
            .Results
            .SelectMany(static x => x.GeneratedSources)
            .ToArray();

        Assert.NotEmpty(generatedSources);

        var modelSource = generatedSources
            .Select(static x => x.SourceText.ToString())
            .First(static x => x.Contains("ModelProjectRelativePath", StringComparison.Ordinal));

        return modelSource;
    }

    private sealed class BinaryAdditionalText(string path) : AdditionalText
    {
        public override string Path { get; } = path;

        public override SourceText GetText(CancellationToken cancellationToken = default)
        {
            return SourceText.From(string.Empty, Encoding.UTF8);
        }
    }

    private sealed class TestAnalyzerConfigOptionsProvider(
        IReadOnlyDictionary<string, string> globalOptions,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> fileOptions) : AnalyzerConfigOptionsProvider
    {
        private readonly AnalyzerConfigOptions _global = new DictionaryAnalyzerConfigOptions(globalOptions);
        private readonly IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> _fileOptions = fileOptions;

        public override AnalyzerConfigOptions GlobalOptions => _global;

        public override AnalyzerConfigOptions GetOptions(SyntaxTree tree)
        {
            return EmptyAnalyzerConfigOptions.Instance;
        }

        public override AnalyzerConfigOptions GetOptions(AdditionalText textFile)
        {
            return _fileOptions.TryGetValue(textFile.Path, out var options)
                ? new DictionaryAnalyzerConfigOptions(options)
                : EmptyAnalyzerConfigOptions.Instance;
        }
    }

    private sealed class DictionaryAnalyzerConfigOptions(IReadOnlyDictionary<string, string> values) : AnalyzerConfigOptions
    {
        public override bool TryGetValue(string key, out string value)
        {
            if (values.TryGetValue(key, out var found))
            {
                value = found;
                return true;
            }

            value = string.Empty;
            return false;
        }
    }

    private sealed class EmptyAnalyzerConfigOptions : AnalyzerConfigOptions
    {
        public static EmptyAnalyzerConfigOptions Instance { get; } = new();

        public override bool TryGetValue(string key, out string value)
        {
            value = string.Empty;
            return false;
        }
    }
}
