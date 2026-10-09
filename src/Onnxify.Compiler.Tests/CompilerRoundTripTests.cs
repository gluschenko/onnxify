using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class CompilerRoundTripTests
{
    [Fact]
    public void TorchSharpInputRoundTripsThroughOnnxAndGeneratedTorchSharp()
    {
        // Source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (nn.functional.relu).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/activation/activation_op_test.cc (Relu).
        var sourceText = """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                return torch.nn.functional.relu(input);
            }
            """;
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(sourceText));
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var sourceOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal(CompilerOperationCapability.Bidirectional, sourceOperation.Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal("Relu", Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        var onnxImported = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(onnxImported.IsSuccess, FormatDiagnostics(onnxImported.Diagnostics));
        var onnxOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(onnxImported.Value!.Operations));
        Assert.Equal(sourceOperation.Descriptor, onnxOperation.Descriptor);

        var generatedSource = GenerateCSharp(onnxImported.Value);
        var inputValues = new[] { -2f, -0.5f, 0f, 3f };
        var generatedOutput = ExecuteGeneratedModule(generatedSource, inputValues);
        var onnxOutput = ExecuteOnnx(emitted.Value, inputValues);
        Assert.Equal(onnxOutput, generatedOutput);

        using var input = global::TorchSharp.torch.tensor(
            inputValues,
            [4L],
            dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var eager = global::TorchSharp.torch.nn.functional.relu(input);
        Assert.Equal(eager.data<float>().ToArray(), onnxOutput);
    }

    private static string GenerateCSharp(CompilerComputationTree tree)
    {
        var suffix = Guid.NewGuid().ToString("N");
        var generated = Compiler.GenerateCSharp(tree, new CompilerCSharpGenerationOptions
        {
            Namespace = $"Onnxify.GeneratedRoundTrip{suffix}",
            ClassName = $"GeneratedTorchModule{suffix}",
            ModuleName = $"GeneratedTorchModule{suffix}",
        });
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        return generated.Value!;
    }

    private static float[] ExecuteGeneratedModule(string source, float[] inputValues)
    {
        var assemblyName = $"Onnxify.GeneratedRoundTrip{Guid.NewGuid():N}";
        var compilation = CSharpCompilation.Create(
            assemblyName,
            [CSharpSyntaxTree.ParseText(source)],
            GetCompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine,
            emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));

        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.StartsWith("GeneratedTorchModule", StringComparison.Ordinal));
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "round-trip")!;
        using var input = global::TorchSharp.torch.tensor(
            inputValues,
            [inputValues.Length],
            dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<float>().ToArray();
    }

    private static float[] ExecuteOnnx(OnnxModel model, float[] inputValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-round-trip-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(inputValues, [inputValues.Length])),
            ]);
            return results.Single().AsTensor<float>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static IEnumerable<MetadataReference> GetCompilationReferences()
    {
        var trustedPlatformAssemblies = (string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES");
        if (!string.IsNullOrWhiteSpace(trustedPlatformAssemblies))
        {
            foreach (var path in trustedPlatformAssemblies.Split(Path.PathSeparator))
            {
                yield return MetadataReference.CreateFromFile(path);
            }
        }

        yield return MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location);
    }

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics)
    {
        return string.Join(Environment.NewLine, diagnostics.Select(static diagnostic =>
            $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));
    }

}
