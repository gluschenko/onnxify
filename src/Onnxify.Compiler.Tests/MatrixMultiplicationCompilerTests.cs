using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class MatrixMultiplicationCompilerTests
{
    [Theory]
    [InlineData("matmul", new long[] { 2, 2, 3 }, new long[] { 1, 3, 2 })]
    [InlineData("mm", new long[] { 2, 3 }, new long[] { 3, 2 })]
    [InlineData("bmm", new long[] { 2, 2, 3 }, new long[] { 2, 3, 2 })]
    [InlineData("matmul", new long[] { 3 }, new long[] { 3 })]
    [InlineData("matmul", new long[] { 2 }, new long[] { 2, 3 })]
    [InlineData("matmul", new long[] { 3, 2, 2 }, new long[] { 2 })]
    public void MatMulMappingsRoundTripBatchedAndMatrixForms(string methodName, long[] leftShape, long[] rightShape)
    {
        // TorchLib upstream: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_matmul, aten_mm, aten_bmm).
        // ONNX Runtime upstream: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/matmul_test.cc (MathOpTest.MatMulFloatType).
        var leftValues = Enumerable.Range(1, (int)leftShape.Aggregate(1L, static (size, dim) => size * dim)).Select(static value => (float)value).ToArray();
        var rightValues = Enumerable.Range(1, (int)rightShape.Aggregate(1L, static (size, dim) => size * dim)).Select(static value => (float)(value * 0.5)).ToArray();
        var model = CreateMatMulModel(leftShape, rightShape);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxOperation = Assert.IsType<CompilerOperation>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("MatMul", onnxOperation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxOperation.Descriptor.Capability);
        var generated = Compiler.GenerateCSharp(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("torch.matmul(left, right)", generated.Value);
        var expected = ExecuteOnnx(model, leftValues, leftShape, rightValues, rightShape);
        AssertClose(expected, ExecuteGenerated(generated.Value!, leftValues, leftShape, rightValues, rightShape));

        var torchSource = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) {{ return left.{methodName}(right); }}";
        var sourceTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(torchSource));
        Assert.True(sourceTree.IsSuccess, FormatDiagnostics(sourceTree.Diagnostics));
        var torchOperation = Assert.IsType<CompilerOperation>(Assert.Single(sourceTree.Value!.Operations));
        Assert.Equal(onnxOperation.Descriptor, torchOperation.Descriptor);
        var emitted = Compiler.GenerateOnnx(sourceTree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal("MatMul", Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        AssertClose(expected, ExecuteOnnx(emitted.Value, leftValues, leftShape, rightValues, rightShape));
    }

    [Fact]
    public void GemmAndAddmmPreserveTransposeBiasAndAlphaBeta()
    {
        // TorchLib upstream: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_addmm).
        // ONNX Runtime upstream: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/gemm_test.cc (GemmOpTest.GemmTransB_1, GemmOpTest.GemmAlpha, GemmOpTest.GemmBeta).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor bias, global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor weight) { return torch.addmm(bias, input, weight, alpha: 0.5f, beta: 2f); }";
        var sourceTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(sourceTree.IsSuccess, FormatDiagnostics(sourceTree.Diagnostics));
        var torchOperation = Assert.IsType<CompilerOperation>(Assert.Single(sourceTree.Value!.Operations));
        Assert.Equal("Gemm", torchOperation.Descriptor.Name);
        Assert.Equal(["input", "weight", "bias"], torchOperation.Inputs.Select(static reference => reference.Name));
        Assert.Equal(0.5f, Assert.IsType<CompilerFloatingPointLiteral>(torchOperation.Attributes.Single(static attribute => attribute.Name == "alpha").Value).Value);
        Assert.Equal(2f, Assert.IsType<CompilerFloatingPointLiteral>(torchOperation.Attributes.Single(static attribute => attribute.Name == "beta").Value).Value);
        var emitted = Compiler.GenerateOnnx(sourceTree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal("Gemm", Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        Assert.Equal(["input", "weight", "bias"], emitted.Value.Graph.Nodes.Single().Inputs.Select(static input => input.Name));
        Assert.Equal(0.5f, (float)emitted.Value.Graph.Nodes.Single().Attributes.Single(static attribute => attribute.Name == "alpha").GetValue());
        Assert.Equal(2f, (float)emitted.Value.Graph.Nodes.Single().Attributes.Single(static attribute => attribute.Name == "beta").GetValue());

        var onnxModel = CreateGemmModel();
        var imported = Compiler.CreateTreeFromOnnx(onnxModel);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var importedOperation = Assert.IsType<CompilerOperation>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("Gemm", importedOperation.Descriptor.Name);
        var generated = Compiler.GenerateCSharp(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("torch.matmul(input.transpose(0, 1), weight.transpose(0, 1))", generated.Value);

        var bias = new[] { 1f, -1f, 2f, 0.5f };
        var input = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var weight = new[] { 1f, 0f, 1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f, 0f, 1f };
        var transposedWeight = new[] { 1f, 0f, 0f, 1f, 0f, 1f, 1f, 0f, 1f, 0f, 0f, 1f };
        var transposedInput = new[] { 1f, 4f, 2f, 5f, 3f, 6f };
        var expected = ExecuteOnnx(onnxModel, transposedInput, [3, 2], weight, [4, 3], bias, [1, 4]);
        AssertClose(expected, ExecuteGeneratedGemm(generated.Value!, transposedInput, [3, 2], weight, [4, 3], bias, [1, 4]));
        AssertClose(expected, ExecuteOnnx(emitted.Value, input, [2, 3], transposedWeight, [3, 4], bias, [1, 4]));
    }

    [Fact]
    public void FunctionalLinearMapsToGemmWithTransposedWeightAndOptionalBias()
    {
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor weight, global::TorchSharp.torch.Tensor bias) { return torch.nn.functional.linear(input, weight, bias); }";
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(result.IsSuccess, FormatDiagnostics(result.Diagnostics));
        var operation = Assert.IsType<CompilerOperation>(Assert.Single(result.Value!.Operations));
        Assert.Equal("Gemm", operation.Descriptor.Name);
        Assert.Equal(["input", "weight", "bias"], operation.Inputs.Select(static reference => reference.Name));
        Assert.Equal(1L, Assert.IsType<CompilerSignedIntegerLiteral>(Assert.Single(operation.Attributes).Value).Value);
        Assert.Equal("transB", Assert.Single(operation.Attributes).Name);
        var emitted = Compiler.GenerateOnnx(result.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal("Gemm", Assert.Single(emitted.Value!.Graph.Nodes).OpType);

        var inputValues = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var weightValues = new[] { 1f, 0f, 1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f, 0f, 1f };
        var biasValues = new[] { 0.25f, -0.5f, 1f, 2f };
        var expected = ExecuteOnnx(emitted.Value, inputValues, [2, 3], weightValues, [4, 3], biasValues, [4]);
        using var module = CompileModule(Compiler.GenerateCSharp(result.Value).Value!, "linear");
        using var input = global::TorchSharp.torch.tensor(inputValues, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var weight = global::TorchSharp.torch.tensor(weightValues, [4L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var bias = global::TorchSharp.torch.tensor(biasValues, [4L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var torchOutput = (global::TorchSharp.torch.Tensor)module.Type.GetMethod("forward")!.Invoke(module.Instance, [input, weight, bias])!;
        AssertClose(expected, torchOutput.data<float>().ToArray());
    }

    [Fact]
    public void FunctionalLinearSupportsMissingBias()
    {
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor weight) { return torch.nn.functional.linear(input, weight); }";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        var operation = Assert.IsType<CompilerOperation>(Assert.Single(tree.Value!.Operations));
        Assert.Equal("Gemm", operation.Descriptor.Name);
        Assert.Equal(2, operation.Inputs.Count);
        Assert.Single(operation.Attributes);
        var onnx = Compiler.GenerateOnnx(tree.Value);
        Assert.True(onnx.IsSuccess, FormatDiagnostics(onnx.Diagnostics));
        var node = Assert.Single(onnx.Value!.Graph.Nodes);
        Assert.Equal("Gemm", node.OpType);
        Assert.Equal(2, node.Inputs.Count);

        var values = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var weights = new[] { 1f, 0f, 1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f, 0f, 1f };
        var expected = ExecuteOnnxWithoutBias(onnx.Value, values, weights);
        using var module = CompileModule(Compiler.GenerateCSharp(tree.Value).Value!, "linear-no-bias");
        using var input = global::TorchSharp.torch.tensor(values, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var weight = global::TorchSharp.torch.tensor(weights, [4L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)module.Type.GetMethod("forward")!.Invoke(module.Instance, [input, weight])!;
        AssertClose(expected, output.data<float>().ToArray());

        var imported = Compiler.CreateTreeFromOnnx(onnx.Value);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateCSharp(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        using var importedModule = CompileModule(generated.Value!, "gemm-no-bias");
        using var importedInput = global::TorchSharp.torch.tensor(values, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var importedWeight = global::TorchSharp.torch.tensor(weights, [4L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var importedOutput = (global::TorchSharp.torch.Tensor)importedModule.Type.GetMethod("forward")!.Invoke(importedModule.Instance, [importedInput, importedWeight])!;
        AssertClose(expected, importedOutput.data<float>().ToArray());
    }

    [Fact]
    public void MatMulRejectsIncompatibleKnownInnerDimensions()
    {
        var result = Compiler.CreateTreeFromOnnx(CreateMatMulModel([2, 3], [4, 2]));
        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error
            && diagnostic.Code == CompilerDiagnosticCodes.Unsupported);
    }

    [Fact]
    public void MatMulRejectsIncompatibleKnownBatchDimensions()
    {
        var result = Compiler.CreateTreeFromOnnx(CreateMatMulModel([2, 2, 3], [3, 3, 2]));
        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error
            && diagnostic.Message.Contains("batch dimensions", StringComparison.Ordinal));
    }

    [Fact]
    public void StaticTorchMatMulCallUsesTheSharedMapping()
    {
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) { return torch.matmul(left, right); }";
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(result.IsSuccess, FormatDiagnostics(result.Diagnostics));
        var operation = Assert.IsType<CompilerOperation>(Assert.Single(result.Value!.Operations));
        Assert.Equal("MatMul", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);
        Assert.True(Compiler.GenerateOnnx(result.Value).IsSuccess);
    }

    [Fact]
    public void MatMulRejectsDtypesOutsideTheTorchSharpAndCpuRuntimeIntersection()
    {
        var result = Compiler.CreateTreeFromOnnx(CreateTypedMatMulModel<uint>());
        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error
            && diagnostic.Message.Contains("runtime-verified mapping", StringComparison.Ordinal));
    }

    [Fact]
    public void GemmRejectsInvalidTransposeFlags()
    {
        var model = CreateGemmModel(transA: 2);
        var result = Compiler.CreateTreeFromOnnx(model);
        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error
            && diagnostic.Message.Contains("transA", StringComparison.Ordinal));
    }

    [Fact]
    public void MatMulSupportsAuditedFloatingAndIntegerRuntimeTypes()
    {
        // ONNXScript aten_matmul and the TorchSharp/CPU-ORT intersection of MathOpTest.MatMulFloatType,
        // MatMulDoubleType, MatMulInt32Type, and MatMulInt64Type. TorchSharp has no UInt32/UInt64 ScalarType.
        AssertTypedMatMul([1f, 2f, 3f, 4f], [2f, 0f, 0f, 2f], [2f, 4f, 6f, 8f], global::TorchSharp.torch.ScalarType.Float32);
        AssertTypedMatMul([1d, 2d, 3d, 4d], [2d, 0d, 0d, 2d], [2d, 4d, 6d, 8d], global::TorchSharp.torch.ScalarType.Float64);
        AssertTypedMatMul([1, 2, 3, 4], [2, 0, 0, 2], [2, 4, 6, 8], global::TorchSharp.torch.ScalarType.Int32);
        AssertTypedMatMul([1L, 2L, 3L, 4L], [2L, 0L, 0L, 2L], [2L, 4L, 6L, 8L], global::TorchSharp.torch.ScalarType.Int64);
    }

    private static OnnxModel CreateMatMulModel(long[] leftShape, long[] rightShape)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<float>(leftShape.Select(static dimension => new OnnxDimension<long>(dimension))));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<float>(rightShape.Select(static dimension => new OnnxDimension<long>(dimension))));
        var batchRank = Math.Max(Math.Max(0, leftShape.Length - 2), Math.Max(0, rightShape.Length - 2));
        var outputShape = new List<long>();
        for (var dimensionIndex = 0; dimensionIndex < batchRank; dimensionIndex++)
        {
            var leftBatchIndex = leftShape.Length - batchRank + dimensionIndex;
            var rightBatchIndex = rightShape.Length - batchRank + dimensionIndex;
            var leftDimension = leftBatchIndex < 0 ? 1 : leftShape[leftBatchIndex];
            var rightDimension = rightBatchIndex < 0 ? 1 : rightShape[rightBatchIndex];
            outputShape.Add(Math.Max(leftDimension, rightDimension));
        }

        if (leftShape.Length > 1)
        {
            outputShape.Add(leftShape[leftShape.Length - 2]);
        }

        if (rightShape.Length > 1)
        {
            outputShape.Add(rightShape[rightShape.Length - 1]);
        }

        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>(outputShape.Select(static dimension => new OnnxDimension<long>(dimension))));
        model.Graph.AddNode("matmul", "MatMul", string.Empty, string.Empty, [left, right], [output], []);
        return model;
    }

    private static OnnxModel CreateTypedMatMulModel<T>() where T : struct
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<T>([new OnnxDimension<long>(2), new OnnxDimension<long>(2)]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<T>([new OnnxDimension<long>(2), new OnnxDimension<long>(2)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<T>([new OnnxDimension<long>(2), new OnnxDimension<long>(2)]));
        model.Graph.AddNode("matmul", "MatMul", string.Empty, string.Empty, [left, right], [output], []);
        return model;
    }

    private static void AssertTypedMatMul<T>(T[] leftValues, T[] rightValues, T[] expected, global::TorchSharp.torch.ScalarType elementType) where T : struct
    {
        var model = CreateTypedMatMulModel<T>();
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateCSharp(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        using var module = CompileModule(generated.Value!, "typed-matmul");
        var torchValues = ExecuteTypedTorchMatMul(module, leftValues, rightValues, elementType);
        var onnxValues = ExecuteTypedOnnxMatMul(model, leftValues, rightValues);
        Assert.Equal(expected, torchValues);
        Assert.Equal(expected, onnxValues);
    }

    private static T[] ExecuteTypedTorchMatMul<T>(CompiledModule module, T[] leftValues, T[] rightValues, global::TorchSharp.torch.ScalarType elementType) where T : struct
    {
        using var left = CreateTorchTensor(leftValues, elementType);
        using var right = CreateTorchTensor(rightValues, elementType);
        using var output = (global::TorchSharp.torch.Tensor)module.Type.GetMethod("forward")!.Invoke(module.Instance, [left, right])!;
        return ReadTorchTensor<T>(output);
    }

    private static global::TorchSharp.torch.Tensor CreateTorchTensor<T>(T[] values, global::TorchSharp.torch.ScalarType elementType) where T : struct
    {
        return values switch
        {
            float[] typed => global::TorchSharp.torch.tensor(typed, [2L, 2L], dtype: elementType),
            double[] typed => global::TorchSharp.torch.tensor(typed, [2L, 2L], dtype: elementType),
            int[] typed => global::TorchSharp.torch.tensor(typed, [2L, 2L], dtype: elementType),
            long[] typed => global::TorchSharp.torch.tensor(typed, [2L, 2L], dtype: elementType),
            _ => throw new NotSupportedException($"Test tensor type '{typeof(T)}' is unsupported."),
        };
    }

    private static T[] ReadTorchTensor<T>(global::TorchSharp.torch.Tensor tensor) where T : struct
    {
        object values = typeof(T) == typeof(float) ? tensor.data<float>().ToArray()
            : typeof(T) == typeof(double) ? tensor.data<double>().ToArray()
            : typeof(T) == typeof(int) ? tensor.data<int>().ToArray()
            : typeof(T) == typeof(long) ? tensor.data<long>().ToArray()
            : throw new NotSupportedException($"Test tensor type '{typeof(T)}' is unsupported.");
        return (T[])values;
    }

    private static T[] ExecuteTypedOnnxMatMul<T>(OnnxModel model, T[] leftValues, T[] rightValues) where T : struct
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-typed-matmul-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<T>(leftValues, [2, 2])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<T>(rightValues, [2, 2])),
            ]);
            return results.Single().AsTensor<T>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static OnnxModel CreateGemmModel(long transA = 1)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([new OnnxDimension<long>(3), new OnnxDimension<long>(2)]));
        var weight = model.Graph.AddInput("weight", OnnxTensorType.Create<float>([new OnnxDimension<long>(4), new OnnxDimension<long>(3)]));
        var bias = model.Graph.AddInput("bias", OnnxTensorType.Create<float>([new OnnxDimension<long>(1), new OnnxDimension<long>(4)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([new OnnxDimension<long>(2), new OnnxDimension<long>(4)]));
        model.Graph.AddNode("gemm", "Gemm", string.Empty, string.Empty, [input, weight, bias], [output],
        [
            new OnnxAttribute<float>("alpha", 0.5f),
            new OnnxAttribute<float>("beta", 2f),
            new OnnxAttribute<long>("transA", transA),
            new OnnxAttribute<long>("transB", 1),
        ]);
        return model;
    }

    private static float[] ExecuteGenerated(string source, float[] leftValues, long[] leftShape, float[] rightValues, long[] rightShape)
    {
        using var module = CompileModule(source, "matmul");
        using var left = global::TorchSharp.torch.tensor(leftValues, leftShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var right = global::TorchSharp.torch.tensor(rightValues, rightShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)module.Type.GetMethod("forward")!.Invoke(module.Instance, [left, right])!;
        return output.data<float>().ToArray();
    }

    private static float[] ExecuteGeneratedGemm(string source, float[] inputValues, long[] inputShape, float[] weightValues, long[] weightShape, float[] biasValues, long[] biasShape)
    {
        using var module = CompileModule(source, "gemm");
        using var bias = global::TorchSharp.torch.tensor(biasValues, biasShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var input = global::TorchSharp.torch.tensor(inputValues, inputShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var weight = global::TorchSharp.torch.tensor(weightValues, weightShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)module.Type.GetMethod("forward")!.Invoke(module.Instance, [input, weight, bias])!;
        return output.data<float>().ToArray();
    }

    private static CompiledModule CompileModule(string source, string name)
    {
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Matrix{name}{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var type = assembly.GetTypes().Single(static candidate => candidate.Name.Contains("TorchModule", StringComparison.Ordinal));
        return new CompiledModule(type, (IDisposable)Activator.CreateInstance(type, name)!);
    }

    private static float[] ExecuteOnnx(OnnxModel model, float[] leftValues, long[] leftShape, float[] rightValues, long[] rightShape)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-matmul-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<float>(leftValues, leftShape.Select(static dimension => (int)dimension).ToArray())),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<float>(rightValues, rightShape.Select(static dimension => (int)dimension).ToArray())),
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

    private static float[] ExecuteOnnx(OnnxModel model, float[] inputValues, long[] inputShape, float[] weightValues, long[] weightShape, float[] biasValues, long[] biasShape)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-gemm-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(inputValues, inputShape.Select(static dimension => (int)dimension).ToArray())),
                NamedOnnxValue.CreateFromTensor("weight", new DenseTensor<float>(weightValues, weightShape.Select(static dimension => (int)dimension).ToArray())),
                NamedOnnxValue.CreateFromTensor("bias", new DenseTensor<float>(biasValues, biasShape.Select(static dimension => (int)dimension).ToArray())),
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

    private static float[] ExecuteOnnxWithoutBias(OnnxModel model, float[] inputValues, float[] weightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-gemm-no-bias-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(inputValues, [2, 3])),
                NamedOnnxValue.CreateFromTensor("weight", new DenseTensor<float>(weightValues, [4, 3])),
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

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics) =>
        string.Join(Environment.NewLine, diagnostics.Select(static diagnostic => $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
        {
            Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0f, 1e-5f * MathF.Max(1f, MathF.Abs(expected[index])));
        }
    }

    private sealed class CompiledModule(Type type, IDisposable instance) : IDisposable
    {
        public Type Type { get; } = type;

        public IDisposable Instance { get; } = instance;

        public void Dispose() => Instance.Dispose();
    }
}
