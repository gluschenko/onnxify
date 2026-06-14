extern alias ModelGen;

using System.Reflection;
using System.Runtime.Loader;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.TorchSharp;
using OnnxModelGenerator = ModelGen::Onnxify.ModelGenerator.OnnxModelGenerator;
using TorchModule = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>;

namespace Onnxify.Tests;

internal static class DeepImportExportParity
{
    public static DeepImportExportParityResult AssertRoundTripMse(
        OnnxModel model,
        IReadOnlyList<NamedOnnxValue> inputs,
        string outputName,
        float threshold,
        string modelFileName = "round-trip.onnx"
    )
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputName);

        if (inputs.Count != 1)
        {
            throw new NotSupportedException("Deep import/export parity currently supports a single ONNX input.");
        }

        if (inputs[0].Value is not DenseTensor<float> inputTensor)
        {
            throw new NotSupportedException("Deep import/export parity currently supports DenseTensor<float> inputs.");
        }

        string tempRoot = Path.Combine(Path.GetTempPath(), "DeepImportExportParity", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        var modelPath = Path.Combine(tempRoot, modelFileName);

        try
        {
            model.Save(modelPath, overwrite: true);

            var originalOutput = RunSingleFloatOutput(modelPath, inputs, outputName);
            using var imported = CompileAndLoadTorchModule(tempRoot, modelPath, model);
            using var eagerOutput = RunTorchModule(imported.Module, inputTensor);
            var eagerTensor = new DenseTensor<float>(
                eagerOutput.data<float>().ToArray(),
                eagerOutput.shape.Select(static dimension => checked((int)dimension)).ToArray()
            );
            var eagerMse = MeanSquaredError(originalOutput, eagerTensor);
            var roundTrippedModel = imported.Module.ExportOnnxModel(
                inputName: inputs[0].Name,
                outputName: outputName,
                input: OnnxTensorType.Create<float>(ToOnnxShape(inputTensor.Dimensions)),
                output: OnnxTensorType.Create<float>(ToOnnxShape(originalOutput.Dimensions)),
                options: new OnnxModelCreationOptions
                {
                    Opset = 22,
                    ProducerName = "onnxify-tests",
                }
            );

            var roundTrippedPath = Path.Combine(tempRoot, "round-tripped.onnx");
            roundTrippedModel.Save(roundTrippedPath, overwrite: true);
            var roundTrippedOutput = RunSingleFloatOutput(roundTrippedPath, inputs, outputName);
            var mse = MeanSquaredError(originalOutput, roundTrippedOutput);

            Assert.True(
                mse <= threshold,
                $"Deep import/export MSE {mse} exceeded threshold {threshold} for output '{outputName}'. "
                + $"Deep import eager MSE was {eagerMse}."
            );

            return new DeepImportExportParityResult(eagerMse, mse, originalOutput.Dimensions.ToArray());
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                DeleteTempRoot(tempRoot);
            }
        }
    }

    private static ImportedTorchModule CompileAndLoadTorchModule(
        string tempRoot,
        string modelPath,
        OnnxModel model
    )
    {
        var driver = CreateDriver(
            additionalFiles: [new BinaryAdditionalText(modelPath)],
            globalOptions: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["build_property.ProjectDir"] = tempRoot + Path.DirectorySeparatorChar,
                ["build_property.RootNamespace"] = "Onnxify.Tests.Generated",
            },
            fileOptions: new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal)
            {
                [modelPath] = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["build_metadata.additionalfiles.OnnxifyModelImportType"] = "TorchModule",
                }
            }
        );

        var compilation = CreateCompilation();
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var updatedCompilation, out var generatorDiagnostics);

        Assert.DoesNotContain(generatorDiagnostics, static x => x.Severity == DiagnosticSeverity.Error);
        Assert.DoesNotContain(updatedCompilation.GetDiagnostics(), static x => x.Severity == DiagnosticSeverity.Error);

        var assemblyName = $"GeneratedTorchModule{Guid.NewGuid():N}";
        updatedCompilation = updatedCompilation.WithAssemblyName(assemblyName);
        var assemblyPath = Path.Combine(tempRoot, $"{assemblyName}.dll");
        var emitResult = updatedCompilation.Emit(assemblyPath);
        Assert.True(
            emitResult.Success,
            string.Join(Environment.NewLine, emitResult.Diagnostics)
        );

        var loadContext = new AssemblyLoadContext(assemblyName, isCollectible: true);
        loadContext.Resolving += ResolveFromCurrentAppDomain;
        var assembly = loadContext.LoadFromAssemblyPath(assemblyPath);
        var moduleType = assembly.GetTypes()
            .Single(type =>
                type.IsPublic
                && type.Name.EndsWith("TorchModule", StringComparison.Ordinal)
                && typeof(TorchModule).IsAssignableFrom(type)
            );

        var module = (TorchModule)Activator.CreateInstance(moduleType)!;
        moduleType.GetMethod("LoadWeightsFromOnnx", [typeof(OnnxModel)])!
            .Invoke(module, [model]);
        module.eval();

        return new ImportedTorchModule(module, loadContext);
    }

    private static Assembly? ResolveFromCurrentAppDomain(
        AssemblyLoadContext loadContext,
        AssemblyName assemblyName
    )
    {
        return AppDomain.CurrentDomain
            .GetAssemblies()
            .FirstOrDefault(assembly =>
                string.Equals(assembly.GetName().Name, assemblyName.Name, StringComparison.Ordinal)
            );
    }

    private static GeneratorDriver CreateDriver(
        IReadOnlyList<AdditionalText> additionalFiles,
        IReadOnlyDictionary<string, string> globalOptions,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> fileOptions
    )
    {
        return CSharpGeneratorDriver.Create(
            generators: [new OnnxModelGenerator().AsSourceGenerator()],
            additionalTexts: additionalFiles,
            parseOptions: (CSharpParseOptions)CSharpParseOptions.Default.WithLanguageVersion(LanguageVersion.Latest),
            optionsProvider: new TestAnalyzerConfigOptionsProvider(globalOptions, fileOptions)
        );
    }

    private static CSharpCompilation CreateCompilation()
    {
        var syntaxTree = CSharpSyntaxTree.ParseText("""
            namespace Onnxify.Tests.Generated;

            public static class Marker
            {
            }
            """);

        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))?
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Distinct(StringComparer.Ordinal)
            .Select(static path => MetadataReference.CreateFromFile(path))
            .ToList() ?? [];

        references.Add(MetadataReference.CreateFromFile(typeof(InferenceSession).Assembly.Location));
        references.Add(MetadataReference.CreateFromFile(typeof(OnnxModel).Assembly.Location));
        references.Add(MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch).Assembly.Location));

        return CSharpCompilation.Create(
            assemblyName: "GeneratedTorchModule",
            syntaxTrees: [syntaxTree],
            references: references,
            options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
        );
    }

    private static DenseTensor<float> RunSingleFloatOutput(
        string modelPath,
        IReadOnlyCollection<NamedOnnxValue> inputs,
        string outputName
    )
    {
        using var session = new InferenceSession(modelPath);
        using var results = session.Run(inputs);
        var output = results.Single(x => string.Equals(x.Name, outputName, StringComparison.Ordinal));
        var tensor = output.AsTensor<float>();
        return new DenseTensor<float>(tensor.ToArray(), tensor.Dimensions.ToArray());
    }

    private static global::TorchSharp.torch.Tensor RunTorchModule(
        TorchModule module,
        DenseTensor<float> input
    )
    {
        using var tensor = global::TorchSharp.torch.tensor(
            input.Buffer.ToArray(),
            input.Dimensions.ToArray().Select(static dimension => (long)dimension).ToArray(),
            dtype: global::TorchSharp.torch.ScalarType.Float32
        );
        using var output = module.forward(tensor);
        return output.detach().cpu();
    }

    private static float MeanSquaredError(DenseTensor<float> expected, DenseTensor<float> actual)
    {
        Assert.Equal(expected.Dimensions.ToArray(), actual.Dimensions.ToArray());
        Assert.Equal(expected.Length, actual.Length);

        var expectedValues = expected.Buffer.Span;
        var actualValues = actual.Buffer.Span;
        double sum = 0d;

        for (var index = 0; index < expectedValues.Length; index++)
        {
            var difference = expectedValues[index] - actualValues[index];
            sum += difference * difference;
        }

        return (float)(sum / expectedValues.Length);
    }

    private static OnnxDimension[] ToOnnxShape(ReadOnlySpan<int> shape)
    {
        var result = new OnnxDimension[shape.Length];
        for (var index = 0; index < shape.Length; index++)
        {
            result[index] = shape[index];
        }

        return result;
    }

    private static void DeleteTempRoot(string tempRoot)
    {
        for (var attempt = 0; attempt < 5; attempt++)
        {
            try
            {
                Directory.Delete(tempRoot, recursive: true);
                return;
            }
            catch (IOException)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                Thread.Sleep(50);
            }
            catch (UnauthorizedAccessException)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                Thread.Sleep(50);
            }
        }
    }

    private sealed class ImportedTorchModule(
        TorchModule module,
        AssemblyLoadContext loadContext
    ) : IDisposable
    {
        public TorchModule Module { get; } = module;

        public void Dispose()
        {
            Module.Dispose();
            loadContext.Unload();
        }
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
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> fileOptions
    ) : AnalyzerConfigOptionsProvider
    {
        private readonly AnalyzerConfigOptions _global = new DictionaryAnalyzerConfigOptions(globalOptions);

        public override AnalyzerConfigOptions GlobalOptions => _global;

        public override AnalyzerConfigOptions GetOptions(SyntaxTree tree)
        {
            return EmptyAnalyzerConfigOptions.Instance;
        }

        public override AnalyzerConfigOptions GetOptions(AdditionalText textFile)
        {
            return fileOptions.TryGetValue(textFile.Path, out var options)
                ? new DictionaryAnalyzerConfigOptions(options)
                : EmptyAnalyzerConfigOptions.Instance;
        }
    }

    private sealed class DictionaryAnalyzerConfigOptions(
        IReadOnlyDictionary<string, string> values
    ) : AnalyzerConfigOptions
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

internal sealed record DeepImportExportParityResult(float EagerMse, float Mse, int[] OutputShape);
