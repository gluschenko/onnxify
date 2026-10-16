using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Compiler;
using Onnxify.Data.Numerics;
using Onnxify.Safetensors;
using Onnxify;

namespace Onnxify.Compiler.Tests;

public sealed class CompilerWeightArtifactTests
{
    [Fact]
    public void OnnxWeightModeReturnsOnlySourceAndLoadsByName()
    {
        var generated = Compiler.GenerateSourceCode(CreateTree(), new CompilerCSharpGenerationOptions
        {
            ClassName = "WeightLoadingModule",
            WeightStorage = CompilerWeightStorageKind.Onnx,
        });

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(static item => item.Message)));
        var source = Assert.IsType<SourceCodeFile>(Assert.Single(generated.Value!));
        Assert.Equal("GeneratedTorchModule.cs", source.Name);
        Assert.Contains("string weightsPath", source.Content, StringComparison.Ordinal);
        Assert.Contains("OnnxModel.FromFile(weightsPath, new OnnxModelBaseOptions", source.Content, StringComparison.Ordinal);
        Assert.Contains("Missing ONNX initializer 'weight'", source.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("1.25f", source.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void SafetensorsWeightModeReturnsSourceAndTypedBinaryArtifact()
    {
        var generated = Compiler.GenerateSourceCode(CreateTree(), new CompilerCSharpGenerationOptions
        {
            ClassName = "WeightLoadingModule",
            WeightStorage = CompilerWeightStorageKind.Safetensors,
        });

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(static item => item.Message)));
        var files = generated.Value!;
        var source = Assert.Single(files.OfType<SourceCodeFile>());
        var binary = Assert.Single(files.OfType<BinaryFile>());
        Assert.Equal("WeightLoadingModule.safetensors", binary.Name);
        Assert.Contains("SafeTensors.Deserialize(File.ReadAllBytes(weightsPath))", source.Content, StringComparison.Ordinal);
        Assert.Contains("Get<global::System.Single>(\"weight\")", source.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("MemoryMarshal", source.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("1.25f", source.Content, StringComparison.Ordinal);

        var archive = SafeTensors.Deserialize(binary.Data.ToArray());
        var tensor = archive.Tensor("weight");
        Assert.Equal(DataType.F32, tensor.DataType);
        Assert.Equal([2UL, 2UL], tensor.Shape);
        Assert.Equal(
            BitConverter.GetBytes(1.25f)
                .Concat(BitConverter.GetBytes(-3.5f))
                .Concat(BitConverter.GetBytes(0.5f))
                .Concat(BitConverter.GetBytes(2f)),
            tensor.Data.ToArray());
    }

    [Fact]
    public void SafetensorsWeightModePreservesFloat16InitializerBits()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<Half>([1, 2]));
        var weight = model.Graph.AddTensor("weight", [1, 2], [HalfFromBits(0x3c00), HalfFromBits(0xc000)]);
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>([1, 2]));
        model.Graph.AddNode("add", "Add", string.Empty, string.Empty, [input, weight], [output], []);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, string.Join(Environment.NewLine, imported.Diagnostics.Select(static item => item.Message)));

        var generated = Compiler.GenerateSourceCode(imported.Value!, new CompilerCSharpGenerationOptions
        {
            ClassName = "Float16WeightsModule",
            WeightStorage = CompilerWeightStorageKind.Safetensors,
        });

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(static item => item.Message)));
        var binary = Assert.Single(generated.Value!.OfType<BinaryFile>());
        var weights = SafeTensors.Deserialize(binary.Data.ToArray());
        Assert.Equal(DataType.F16, weights.Tensor("weight").DataType);
        Assert.Equal(
            new ushort[] { 0x3c00, 0xc000 },
            weights.Get<Half>("weight").Select(BitConverter.HalfToUInt16Bits));
    }

    [Theory]
    [InlineData(CompilerWeightStorageKind.Onnx)]
    [InlineData(CompilerWeightStorageKind.Safetensors)]
    public void InitializersUsedByNonWeightOperatorsAreStoredExternally(CompilerWeightStorageKind storage)
    {
        var tree = Compiler.CreateTreeFromOnnx(CreateAddModel());
        Assert.True(tree.IsSuccess, string.Join(Environment.NewLine, tree.Diagnostics.Select(static item => item.Message)));

        var generated = Compiler.GenerateSourceCode(tree.Value!, new CompilerCSharpGenerationOptions
        {
            ClassName = "AddConstantModule",
            WeightStorage = storage,
        });

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(static item => item.Message)));
        var generatedFiles = generated.Value!;
        var source = Assert.Single(generatedFiles.OfType<SourceCodeFile>());
        Assert.DoesNotContain("1.25f", source.Content, StringComparison.Ordinal);
        if (storage == CompilerWeightStorageKind.Onnx)
        {
            Assert.Contains("Missing ONNX initializer 'bias'", source.Content, StringComparison.Ordinal);
        }
        else
        {
            var binary = Assert.Single(generatedFiles.OfType<BinaryFile>());
            Assert.Equal(DataType.F32, SafeTensors.Deserialize(binary.Data.ToArray()).Tensor("bias").DataType);
        }
    }

    [Theory]
    [InlineData(CompilerWeightStorageKind.Onnx)]
    [InlineData(CompilerWeightStorageKind.Safetensors)]
    public void GeneratedModuleLoadsExternalWeightsAndMatchesOnnxRuntime(CompilerWeightStorageKind storage)
    {
        var model = CreateModel();
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, string.Join(Environment.NewLine, imported.Diagnostics.Select(static item => item.Message)));
        var generated = Compiler.GenerateSourceCode(imported.Value!, new CompilerCSharpGenerationOptions
        {
            Namespace = $"Onnxify.GeneratedWeights{Guid.NewGuid():N}",
            ClassName = "WeightLoadingModule",
            WeightStorage = storage,
        });
        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(static item => item.Message)));

        var tempDirectory = Path.Combine(Path.GetTempPath(), $"oxy043-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDirectory);
        var onnxPath = Path.Combine(tempDirectory, "model.onnx");
        model.Save(onnxPath, overwrite: true);
        var weightsPath = onnxPath;
        if (storage == CompilerWeightStorageKind.Safetensors)
        {
            weightsPath = Path.Combine(tempDirectory, Assert.Single(generated.Value!.OfType<BinaryFile>()).Name);
            File.WriteAllBytes(weightsPath, Assert.Single(generated.Value!.OfType<BinaryFile>()).Data.ToArray());
        }

        try
        {
            using var sourceSession = new InferenceSession(onnxPath);
            var inputValues = new[] { 2f, -1f };
            using var expectedResult = sourceSession.Run(
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(inputValues, [1, 2])),
            ]);
            var expected = expectedResult.Single().AsTensor<float>().ToArray();
            var assemblyName = $"Onnxify.GeneratedWeights{Guid.NewGuid():N}";
            var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))!
                .Split(Path.PathSeparator)
                .Select(static path => MetadataReference.CreateFromFile(path));
            var compilation = CSharpCompilation.Create(
                assemblyName,
                [CSharpSyntaxTree.ParseText(generated.SourceCode())],
                references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
            using var assemblyBytes = new MemoryStream();
            var emit = compilation.Emit(assemblyBytes);
            Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static item => item.Severity == DiagnosticSeverity.Error)));
            assemblyBytes.Position = 0;
            var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
            var type = assembly.GetTypes().Single(static candidate => candidate.Name == "WeightLoadingModule");
            using var module = (IDisposable)Activator.CreateInstance(type, weightsPath, "weight-check")!;
            using var input = global::TorchSharp.torch.tensor(inputValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float32);
            using var actualResult = (global::TorchSharp.torch.Tensor)type.GetMethod("forward")!.Invoke(module, [input])!;
            var actual = actualResult.data<float>().ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < expected.Length; index++)
            {
                Assert.InRange(Math.Abs(expected[index] - actual[index]), 0f, 1e-6f);
            }
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    private static CompilerComputationTree CreateTree()
    {
        var imported = Compiler.CreateTreeFromOnnx(CreateModel());
        Assert.True(imported.IsSuccess, string.Join(Environment.NewLine, imported.Diagnostics.Select(static item => item.Message)));
        return imported.Value!;
    }

    private static OnnxModel CreateModel()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 2]));
        var weight = model.Graph.AddTensor("weight", [2, 2], [1.25f, -3.5f, 0.5f, 2f]);
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 2]));
        model.Graph.AddNode("matmul", "MatMul", string.Empty, string.Empty, [input, weight], [output], []);
        return model;
    }

    private static OnnxModel CreateAddModel()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 2]));
        var bias = model.Graph.AddTensor("bias", [1, 2], [1.25f, -3.5f]);
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([1, 2]));
        model.Graph.AddNode("add", "Add", string.Empty, string.Empty, [input, bias], [output], []);
        return model;
    }

    private static Half HalfFromBits(ushort bits) => BitConverter.UInt16BitsToHalf(bits);
}
