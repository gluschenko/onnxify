using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class ShapeCompilerTests
{
    [Fact]
    public void FlattenAxisOneRoundTripsAndPreservesRuntimeValues()
    {
        // ONNXScript source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_flatten).
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/nn/flatten_op_test.cc (FlattenOpTest.Flatten_default_axis).
        var model = CreateFlattenModel();
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Flatten>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateCSharp(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("torch.flatten(input, start_dim: 1)", generated.Value);

        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values);
        Assert.Equal(expected, ExecuteGenerated(generated.Value!, values));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.flatten(1); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        Assert.IsType<Onnxify.Flatten>(torchStep.Node);
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);
        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Flatten>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, values));
    }

    [Fact]
    public void FlattenRejectsUnsupportedAxisAndDynamicTorchSharpArguments()
    {
        var invalidModel = CreateFlattenModel(axis: 4);
        var imported = Compiler.CreateTreeFromOnnx(invalidModel);
        Assert.False(imported.IsSuccess);
        Assert.Contains(imported.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error
            && diagnostic.Message.Contains("rank", StringComparison.Ordinal));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, int start) { return input.flatten(start); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.False(scanned.IsSuccess);
        Assert.Contains(scanned.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error);
    }

    [Fact]
    public void FlattenSupportsAdditionalStaticAxesInBothDirections()
    {
        foreach (var axis in new long[] { 0, 2, -1 })
        {
            var model = CreateFlattenModel(axis);
            var imported = Compiler.CreateTreeFromOnnx(model);
            Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
            var generated = Compiler.GenerateCSharp(imported.Value!);
            Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));

            var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
            var expected = ExecuteOnnx(model, values);
            Assert.Equal(expected, ExecuteGenerated(generated.Value!, values));

            var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) {{ return input.flatten({axis}); }}";
            var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
            Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
            var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
            Assert.Equal((long?)axis, Assert.IsType<Onnxify.Flatten>(torchStep.Node).Axis);
            var emitted = Compiler.GenerateOnnx(scanned.Value);
            Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
            Assert.Equal(expected, ExecuteOnnx(emitted.Value!, values));
        }
    }

    [Fact]
    public void TransposePermutationRoundTripsAndPreservesRuntimeValues()
    {
        // ONNXScript source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_permute).
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/transpose_test.cc (TransposeOpTest.TwoDim).
        var model = CreateTransposeModel([1, 2, 0]);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Transpose>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateCSharp(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("input.permute(new long[] { 1, 2, 0 })", generated.Value);

        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values);
        Assert.Equal(expected, ExecuteGenerated(generated.Value!, values));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.permute(new long[] { 1, 2, 0 }); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        Assert.IsType<Onnxify.Transpose>(torchStep.Node);
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);
        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Transpose>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, values));
    }

    [Fact]
    public void TransposeRejectsInvalidOrDynamicPermutations()
    {
        var invalidModel = CreateTransposeModel([0, 0, 2]);
        var imported = Compiler.CreateTreeFromOnnx(invalidModel);
        Assert.False(imported.IsSuccess);
        Assert.Contains(imported.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error);

        var rankMismatch = Compiler.CreateTreeFromOnnx(CreateTransposeModel([1, 0]));
        Assert.False(rankMismatch.IsSuccess);
        Assert.Contains(rankMismatch.Diagnostics, static diagnostic => diagnostic.Message.Contains("input tensor rank", StringComparison.Ordinal));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, long axis) { return input.permute(new long[] { 0, axis, 2 }); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.False(scanned.IsSuccess);
        Assert.Contains(scanned.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error);
    }

    private static OnnxModel CreateFlattenModel(long? axis = null)
    {
        var normalizedAxis = axis ?? 1;
        if (normalizedAxis < 0)
        {
            normalizedAxis += 3;
        }

        var dimensions = new long[] { 2, 3, 4 };
        var outputShape = normalizedAxis is >= 0 and <= 3
            ? new long[]
            {
                dimensions.Take((int)normalizedAxis).Aggregate(1L, static (product, dimension) => product * dimension),
                dimensions.Skip((int)normalizedAxis).Aggregate(1L, static (product, dimension) => product * dimension),
            }
            : new long[] { 2, 12 };
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([2, 3, 4]));
        var output = model.Graph.AddOutput(
            "output",
            OnnxTensorType.Create<float>(outputShape.Select(static dimension => new OnnxDimension<long>(dimension))));
        model.Graph.AddNode(new Onnxify.Flatten(
            "flatten",
            new Onnxify.FlattenInputOutputOptions
            {
                Input = input,
                Axis = axis,
                Output = output,
            }));
        return model;
    }

    private static OnnxModel CreateTransposeModel(long[] permutation)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([2, 3, 4]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([3, 4, 2]));
        model.Graph.AddNode(new Onnxify.Transpose(
            "transpose",
            new Onnxify.TransposeInputOutputOptions
            {
                Data = input,
                Perm = permutation,
                Transposed = output,
            }));
        return model;
    }

    private static float[] ExecuteGenerated(string source, float[] values)
    {
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Shape{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "flatten")!;
        using var input = global::TorchSharp.torch.tensor(values, [2L, 3L, 4L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<float>().ToArray();
    }

    private static float[] ExecuteOnnx(OnnxModel model, float[] values)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-flatten-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(values, [2, 3, 4])),
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

    private static IEnumerable<MetadataReference> CompilationReferences()
    {
        var paths = (string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES");
        if (!string.IsNullOrWhiteSpace(paths))
        {
            foreach (var path in paths.Split(Path.PathSeparator))
            {
                yield return MetadataReference.CreateFromFile(path);
            }
        }

        yield return MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location);
    }

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics)
    {
        return string.Join(Environment.NewLine, diagnostics.Select(static diagnostic => $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));
    }
}
