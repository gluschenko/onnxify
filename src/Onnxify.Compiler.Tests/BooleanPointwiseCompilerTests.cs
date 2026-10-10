using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class BooleanPointwiseCompilerTests
{
    public static IEnumerable<object[]> ComparisonCases =>
    [
        ["Equal", "eq", (Func<float, float, bool>)((left, right) => left == right)],
        ["Greater", "gt", (Func<float, float, bool>)((left, right) => left > right)],
        ["GreaterOrEqual", "ge", (Func<float, float, bool>)((left, right) => left >= right)],
        ["Less", "lt", (Func<float, float, bool>)((left, right) => left < right)],
        ["LessOrEqual", "le", (Func<float, float, bool>)((left, right) => left <= right)],
    ];

    public static IEnumerable<object[]> LogicalCases =>
    [
        ["And", "logical_and", (Func<bool, bool, bool>)((left, right) => left && right)],
        ["Or", "logical_or", (Func<bool, bool, bool>)((left, right) => left || right)],
        ["Xor", "logical_xor", (Func<bool, bool, bool>)((left, right) => left ^ right)],
    ];

    [Theory]
    [MemberData(nameof(ComparisonCases))]
    public void ComparisonMappingsAreBidirectionalAndMatchOnnxRuntime(
        string onnxName,
        string torchName,
        Func<float, float, bool> comparison
    )
    {
        // Torch converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (eq, ge, gt, le, lt), exercised by ops_test.py::test_output_match_opinfo_.
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Equal_float, Greater_9_float, GreaterOrEqual_12_float, Less, LessOrEqual).
        var leftValues = new[] { 1f, 4f, 3f, 5f, 2f, 6f };
        var rightValues = new[] { 1f, 3f, 7f };
        var expected = Broadcast(leftValues, rightValues, comparison);
        var model = CreateModel(onnxName, isLogical: false);

        var onnxTree = Compiler.CreateTreeFromOnnx(model);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(onnxTree.Value!.Operations));
        Assert.Equal(onnxName, onnxStep.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generatedSource = Compiler.GenerateCSharp(onnxTree.Value);
        Assert.True(generatedSource.IsSuccess, FormatDiagnostics(generatedSource.Diagnostics));
        Assert.True(expected.SequenceEqual(ExecuteGeneratedModule(generatedSource.Value!, leftValues, rightValues)));

        var torchSource = CreateTorchSource(torchName);
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(torchSource));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations));
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);

        var emittedModel = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emittedModel.IsSuccess, FormatDiagnostics(emittedModel.Diagnostics));
        Assert.Equal(onnxName, Assert.Single(emittedModel.Value!.Graph.Nodes).OpType);
        Assert.True(expected.SequenceEqual(ExecuteOnnx(emittedModel.Value, leftValues, rightValues)));
    }

    [Theory]
    [MemberData(nameof(LogicalCases))]
    public void LogicalMappingsAreBidirectionalAndMatchOnnxRuntime(
        string onnxName,
        string torchName,
        Func<bool, bool, bool> operation
    )
    {
        // Torch converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (logical_and, logical_or, logical_xor), exercised by ops_test.py::test_output_match_opinfo_.
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (And, Or, Xor, Xor_bcast3v2d).
        var leftValues = new[] { true, false, true, false, true, false };
        var rightValues = new[] { false, true, true };
        var expected = Broadcast(leftValues, rightValues, operation);
        var model = CreateModel(onnxName, isLogical: true);

        var onnxTree = Compiler.CreateTreeFromOnnx(model);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(onnxTree.Value!.Operations));
        Assert.Equal(onnxName, onnxStep.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generatedSource = Compiler.GenerateCSharp(onnxTree.Value);
        Assert.True(generatedSource.IsSuccess, FormatDiagnostics(generatedSource.Diagnostics));
        Assert.True(expected.SequenceEqual(ExecuteGeneratedModule(generatedSource.Value!, leftValues, rightValues)));

        var torchSource = CreateTorchSource(torchName);
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(torchSource));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations));
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);

        var emittedModel = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emittedModel.IsSuccess, FormatDiagnostics(emittedModel.Diagnostics));
        Assert.Equal(onnxName, Assert.Single(emittedModel.Value!.Graph.Nodes).OpType);
        Assert.True(expected.SequenceEqual(ExecuteOnnx(emittedModel.Value, leftValues, rightValues)));
    }

    [Fact]
    public void LogicalNotMappingIsBidirectionalAndMatchesOnnxRuntime()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (logical_not), exercised by ops_test.py::test_output_match_opinfo_.
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Not).
        var inputValues = new[] { true, false, true, false };
        var expected = new[] { false, true, false, true };
        var model = CreateUnaryModel(
            inputType: OnnxTensorType.Create<bool>([new OnnxDimension<long>(4)]),
            node: new Onnxify.Not(
                "logical_not",
                new Onnxify.NotInputOutputOptions
                {
                    X = new OnnxEdge("input"),
                    Y = new OnnxEdge("output"),
                }),
            outputType: OnnxTensorType.Create<bool>([new OnnxDimension<long>(4)]));

        VerifyUnaryBooleanMapping(
            model,
            "torch.logical_not(input)",
            "Not",
            inputValues,
            expected);
    }

    [Fact]
    public void IsNaNMappingIsBidirectionalAndMatchesOnnxRuntime()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (isnan), exercised by ops_test.py::test_output_match_opinfo_.
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/isnan_test.cc (IsNaNFloat9, IsNaNDouble9).
        var inputValues = new[] { float.NaN, 0f, float.PositiveInfinity, -1f };
        var expected = new[] { true, false, false, false };
        var model = CreateUnaryModel(
            inputType: OnnxTensorType.Create<float>([new OnnxDimension<long>(4)]),
            node: new Onnxify.IsNaN(
                "isnan",
                new Onnxify.IsNaNInputOutputOptions
                {
                    X = new OnnxEdge("input"),
                    Y = new OnnxEdge("output"),
                }),
            outputType: OnnxTensorType.Create<bool>([new OnnxDimension<long>(4)]));

        VerifyUnaryBooleanMapping(
            model,
            "torch.isnan(input)",
            "IsNaN",
            inputValues,
            expected);
    }

    private static OnnxModel CreateUnaryModel(OnnxTensorType inputType, OnnxNode node, OnnxTensorType outputType)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        model.Graph.AddInput("input", inputType);
        model.Graph.AddOutput("output", outputType);
        model.Graph.AddNode(node);
        return model;
    }

    private static void VerifyUnaryBooleanMapping<TInput>(
        OnnxModel model,
        string torchExpression,
        string onnxName,
        TInput[] inputValues,
        bool[] expected)
    {
        var onnxTree = Compiler.CreateTreeFromOnnx(model);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(onnxTree.Value!.Operations));
        Assert.Equal(onnxName, onnxStep.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);
        var generatedSource = Compiler.GenerateCSharp(onnxTree.Value);
        Assert.True(generatedSource.IsSuccess, FormatDiagnostics(generatedSource.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedUnaryModule(generatedSource.Value!, inputValues));

        var torchSource = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) {{ return {torchExpression}; }}";
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(torchSource));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations));
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);
        var emittedModel = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emittedModel.IsSuccess, FormatDiagnostics(emittedModel.Diagnostics));
        Assert.Equal(onnxName, Assert.Single(emittedModel.Value!.Graph.Nodes).OpType);
        Assert.Equal(expected, ExecuteOnnxUnary(emittedModel.Value, inputValues));
    }

    private static OnnxModel CreateModel(string opType, bool isLogical)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var leftShape = new OnnxDimension[] { new OnnxDimension<long>(2), new OnnxDimension<long>(3) };
        var rightShape = new OnnxDimension[] { new OnnxDimension<long>(3) };
        var leftType = isLogical ? OnnxTensorType.Create<bool>(leftShape) : OnnxTensorType.Create<float>(leftShape);
        var rightType = isLogical ? OnnxTensorType.Create<bool>(rightShape) : OnnxTensorType.Create<float>(rightShape);
        var left = model.Graph.AddInput("left", leftType);
        var right = model.Graph.AddInput("right", rightType);
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<bool>(leftShape));
        var leftEdge = new OnnxEdge(left.Name);
        var rightEdge = new OnnxEdge(right.Name);
        var outputEdge = new OnnxEdge(output.Name);
        var node = CreateNode(opType, leftEdge, rightEdge, outputEdge);
        model.Graph.AddNode(node);
        return model;
    }

    private static OnnxNode CreateNode(string opType, OnnxEdge left, OnnxEdge right, OnnxEdge output)
    {
        var result = opType switch
        {
            "Equal" => (OnnxNode)new Onnxify.Equal("pointwise", new Onnxify.EqualInputOutputOptions { A = left, B = right, C = output }),
            "Greater" => new Onnxify.Greater("pointwise", new Onnxify.GreaterInputOutputOptions { A = left, B = right, C = output }),
            "GreaterOrEqual" => new Onnxify.GreaterOrEqual("pointwise", new Onnxify.GreaterOrEqualInputOutputOptions { A = left, B = right, C = output }),
            "Less" => new Onnxify.Less("pointwise", new Onnxify.LessInputOutputOptions { A = left, B = right, C = output }),
            "LessOrEqual" => new Onnxify.LessOrEqual("pointwise", new Onnxify.LessOrEqualInputOutputOptions { A = left, B = right, C = output }),
            "And" => new Onnxify.And("pointwise", new Onnxify.AndInputOutputOptions { A = left, B = right, C = output }),
            "Or" => new Onnxify.Or("pointwise", new Onnxify.OrInputOutputOptions { A = left, B = right, C = output }),
            "Xor" => new Onnxify.Xor("pointwise", new Onnxify.XorInputOutputOptions { A = left, B = right, C = output }),
            _ => throw new ArgumentOutOfRangeException(nameof(opType), opType, "No typed test node is registered for this operator."),
        };
        return result;
    }

    private static string CreateTorchSource(string torchName)
    {
        var result = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) {{ return torch.{torchName}(left, right); }}";
        return result;
    }

    private static TOutput[] Broadcast<TInput, TOutput>(
        TInput[] left,
        TInput[] right,
        Func<TInput, TInput, TOutput> operation)
    {
        var result = new TOutput[left.Length];
        for (var index = 0; index < result.Length; index++)
        {
            result[index] = operation(left[index], right[index % right.Length]);
        }

        return result;
    }

    private static bool[] ExecuteGeneratedModule(string source, float[] left, float[] right)
    {
        var compilation = CSharpCompilation.Create(
            $"Onnxify.BooleanPointwise{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "pointwise")!;
        using var leftTensor = global::TorchSharp.torch.tensor(left, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var rightTensor = global::TorchSharp.torch.tensor(right, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [leftTensor, rightTensor])!;
        return output.data<bool>().ToArray();
    }

    private static bool[] ExecuteGeneratedModule(string source, bool[] left, bool[] right)
    {
        var compilation = CSharpCompilation.Create(
            $"Onnxify.BooleanPointwise{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "pointwise")!;
        using var leftTensor = global::TorchSharp.torch.tensor(left, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Bool);
        using var rightTensor = global::TorchSharp.torch.tensor(right, [3L], dtype: global::TorchSharp.torch.ScalarType.Bool);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [leftTensor, rightTensor])!;
        return output.data<bool>().ToArray();
    }

    private static bool[] ExecuteOnnx(OnnxModel model, float[] left, float[] right)
    {
        return ExecuteOnnx(model, new DenseTensor<float>(left, [2, 3]), new DenseTensor<float>(right, [3]));
    }

    private static bool[] ExecuteOnnx(OnnxModel model, bool[] left, bool[] right)
    {
        return ExecuteOnnx(model, new DenseTensor<bool>(left, [2, 3]), new DenseTensor<bool>(right, [3]));
    }

    private static bool[] ExecuteOnnx<T>(OnnxModel model, DenseTensor<T> left, DenseTensor<T> right)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-boolean-pointwise-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", left),
                NamedOnnxValue.CreateFromTensor("right", right),
            ]);
            return results.Single().AsTensor<bool>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static bool[] ExecuteGeneratedUnaryModule<TInput>(string source, TInput[] inputValues)
    {
        var compilation = CSharpCompilation.Create(
            $"Onnxify.BooleanUnary{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "boolean_unary")!;
        using var input = CreateTorchTensor(inputValues);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<bool>().ToArray();
    }

    private static global::TorchSharp.torch.Tensor CreateTorchTensor<TInput>(TInput[] inputValues)
    {
        var result = inputValues switch
        {
            bool[] booleanValues => global::TorchSharp.torch.tensor(
                booleanValues,
                [booleanValues.Length],
                dtype: global::TorchSharp.torch.ScalarType.Bool),
            float[] floatValues => global::TorchSharp.torch.tensor(
                floatValues,
                [floatValues.Length],
                dtype: global::TorchSharp.torch.ScalarType.Float32),
            _ => throw new ArgumentException($"Unsupported unary test input type '{typeof(TInput)}'.", nameof(inputValues)),
        };
        return result;
    }

    private static bool[] ExecuteOnnxUnary<TInput>(OnnxModel model, TInput[] inputValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-boolean-unary-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            NamedOnnxValue input;
            if (inputValues is bool[] booleanValues)
            {
                input = NamedOnnxValue.CreateFromTensor("input", new DenseTensor<bool>(booleanValues, [booleanValues.Length]));
            }
            else if (inputValues is float[] floatValues)
            {
                input = NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(floatValues, [floatValues.Length]));
            }
            else
            {
                throw new ArgumentException($"Unsupported unary test input type '{typeof(TInput)}'.", nameof(inputValues));
            }

            using var results = session.Run([input]);
            return results.Single().AsTensor<bool>().ToArray();
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
        var result = string.Join(Environment.NewLine, diagnostics.Select(static diagnostic => $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));
        return result;
    }
}
