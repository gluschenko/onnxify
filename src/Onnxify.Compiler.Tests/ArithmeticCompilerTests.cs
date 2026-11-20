using System.Runtime.Loader;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class ArithmeticCompilerTests
{
    public static IEnumerable<object[]> BinaryArithmeticCases =>
    [
        ["Add", "+", (Func<float, float, float>)((left, right) => left + right)],
        ["Sub", "-", (Func<float, float, float>)((left, right) => left - right)],
        ["Mul", "*", (Func<float, float, float>)((left, right) => left * right)],
        ["Div", "/", (Func<float, float, float>)((left, right) => left / right)],
        ["Pow", "pow", (Func<float, float, float>)MathF.Pow],
        ["Max", "maximum", (Func<float, float, float>)MathF.Max],
        ["Min", "minimum", (Func<float, float, float>)MathF.Min],
    ];

    public static IEnumerable<object[]> UnaryPointwiseCases =>
    [
        ["Abs", "abs", (Func<float, float>)MathF.Abs],
        ["Neg", "neg", (Func<float, float>)(value => -value)],
        ["Exp", "exp", (Func<float, float>)MathF.Exp],
        ["Log", "log", (Func<float, float>)MathF.Log],
        ["Sqrt", "sqrt", (Func<float, float>)MathF.Sqrt],
        ["Floor", "floor", (Func<float, float>)MathF.Floor],
        ["Ceil", "ceil", (Func<float, float>)MathF.Ceiling],
        ["Sin", "sin", (Func<float, float>)MathF.Sin],
        ["Cos", "cos", (Func<float, float>)MathF.Cos],
        ["Tan", "tan", (Func<float, float>)MathF.Tan],
        ["Asin", "asin", (Func<float, float>)MathF.Asin],
        ["Acos", "acos", (Func<float, float>)MathF.Acos],
        ["Atan", "atan", (Func<float, float>)MathF.Atan],
        ["Sinh", "sinh", (Func<float, float>)MathF.Sinh],
        ["Cosh", "cosh", (Func<float, float>)MathF.Cosh],
        ["Asinh", "asinh", (Func<float, float>)MathF.Asinh],
        ["Acosh", "acosh", (Func<float, float>)MathF.Acosh],
        ["Atanh", "atanh", (Func<float, float>)MathF.Atanh],
        ["Erf", "erf", (Func<float, float>)ErrorFunction],
        ["Reciprocal", "reciprocal", (Func<float, float>)(value => 1f / value)],
        ["Round", "round", (Func<float, float>)(value => MathF.Round(value, MidpointRounding.ToEven))],
        ["Sign", "sign", (Func<float, float>)(value => MathF.Sign(value))],
    ];

    [Theory]
    [MemberData(nameof(BinaryArithmeticCases))]
    public void BinaryArithmeticMappingsAreBidirectionalAndMatchOnnxRuntime(
        string onnxName,
        string csharpOperator,
        Func<float, float, float> expected
    )
    {
        // Torch converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_add, aten_sub, aten_mul, aten_div, aten_pow, aten_maximum, aten_minimum).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Add_Broadcast_MultidirectionalAB, Max_12_Float, Min_12_Float).
        var leftValues = new[] { 2f, 4f, 6f, 8f, 10f, 12f };
        var rightValues = new[] { 2f, 2f, 3f };
        var expectedValues = new float[6];
        for (var row = 0; row < 2; row++)
        {
            for (var column = 0; column < 3; column++)
            {
                expectedValues[row * 3 + column] = expected(leftValues[row * 3 + column], rightValues[column]);
            }
        }

        var onnxModel = CreateBinaryModel(onnxName, [2, 3], [3]);
        var onnxTree = Compiler.CreateTreeFromOnnx(onnxModel);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));
        var onnxOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(onnxTree.Value!.Operations));
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxOperation.Descriptor.Capability);
        Assert.Equal(onnxName, onnxOperation.Descriptor.Name);
        var generatedCSharp = Compiler.GenerateSourceCode(onnxTree.Value);
        Assert.True(generatedCSharp.IsSuccess, FormatDiagnostics(generatedCSharp.Diagnostics));
        var expectedGeneratedForm = onnxName switch
        {
            "Add" => " + ",
            "Sub" => " - ",
            "Mul" => " * ",
            "Div" => " / ",
            "Pow" => ".pow(",
            "Max" => ".maximum(",
            _ => ".minimum(",
        };
        Assert.Contains(expectedGeneratedForm, generatedCSharp.SourceCode());
        AssertClose(expectedValues, ExecuteGeneratedModule(generatedCSharp, leftValues, rightValues));

        var torchSource = onnxName switch
        {
            "Pow" => "return left.pow(right);",
            "Max" or "Min" => $"return left.{csharpOperator}(right);",
            _ => $"return left {csharpOperator} right;",
        };
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(
            $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) {{ {torchSource} }}"));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var torchOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations));
        Assert.Equal(onnxOperation.Descriptor, torchOperation.Descriptor);
        var emitted = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(onnxName, Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        AssertClose(expectedValues, ExecuteOnnx(emitted.Value, leftValues, [2, 3], rightValues, [3]));
    }

    [Fact]
    public void ScalarArithmeticCreatesAnOnnxScalarInitializer()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_add).
        // ONNX runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Add_Broadcast_0x1).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input + 2f; }";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Single(imported.Value!.Initializers);
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal([3f, 4f, 5f], ExecuteOnnxScalar(emitted.Value!, [1f, 2f, 3f]));

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal([3f, 4f, 5f], ExecuteGeneratedModule(generated, [1f, 2f, 3f], null, emitted.Value));
    }

    [Fact]
    public void ScalarArithmeticTorchSharpCallCreatesAnOnnxScalarInitializer()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_add).
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Add_Broadcast_0x1).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.add(2f); }";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("Add", operation.Descriptor.Name);
        Assert.Single(imported.Value.Initializers);
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        AssertClose([3f, 4f, 5f], ExecuteOnnxScalar(emitted.Value!, [1f, 2f, 3f]));
    }

    [Theory]
    [InlineData("return input - 2f;", new float[] { -1f, 0f, 1f })]
    [InlineData("return input * 2f;", new float[] { 2f, 4f, 6f })]
    [InlineData("return input / 2f;", new float[] { 0.5f, 1f, 1.5f })]
    [InlineData("return input.pow(2f);", new float[] { 1f, 4f, 9f })]
    [InlineData("return 2f - input;", new float[] { 1f, 0f, -1f })]
    [InlineData("return 2f / input;", new float[] { 2f, 1f, 2f / 3f })]
    public void ScalarArithmeticOperandsPreserveOrderAndExecute(string returnStatement, float[] expected)
    {
        // Torch converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_sub, aten_mul, aten_div, aten_pow).
        // ONNX Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Sub, Mul, Div, Pow scalar broadcasting cases).
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) {{ {returnStatement} }}";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Single(imported.Value!.Initializers);

        var emitted = Compiler.GenerateOnnx(imported.Value);

        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        AssertClose(expected, ExecuteOnnxScalar(emitted.Value!, [1f, 2f, 3f]));

        var generated = Compiler.GenerateSourceCode(imported.Value);

        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertClose(expected, ExecuteGeneratedModule(generated, [1f, 2f, 3f], null, emitted.Value));
    }

    [Fact]
    public void DynamicScalarArithmeticIsRejectedWithACompilerDiagnostic()
    {
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, float scale) { return input * scale; }";
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.NotNull(diagnostic.Span);
    }

    [Theory]
    [InlineData("torch.add(left, right, alpha: 2f)")]
    [InlineData("torch.sub(left, right, alpha: 2f)")]
    [InlineData("torch.add(left, right, alpha: scale)")]
    [InlineData("torch.sub(left, right, alpha: scale)")]
    public void AddAndSubRejectUnsupportedAlphaOverloadsWithSourceDiagnostics(string expression)
    {
        // TorchLib overloads: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py::aten_add and ::aten_sub.
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right, float scale) {{ return {expression}; }}";
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.NotNull(diagnostic.Span);
    }

    [Fact]
    public void KnownIncompatibleBroadcastShapesProduceAnExplicitDiagnostic()
    {
        // Runtime validation case: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Add_Invalid_Broadcast).
        var model = CreateBinaryModel("Add", [2, 3], [4]);
        var result = Compiler.CreateTreeFromOnnx(model);
        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics, static item => item.Severity == CompilerDiagnosticSeverity.Error);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Contains("cannot be broadcast", diagnostic.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BinaryMappingsSupportMultidirectionalBroadcastingAcrossSeveralRanks()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (TorchLibOpInfo("add", core_ops.aten_add)).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Add_Broadcast_2x1x4_1x3x1 and MathOpTest.Add_Invalid_Broadcast).
        var leftShape = new long[] { 2, 1, 4 };
        var rightShape = new long[] { 1, 3, 1 };
        var outputShape = new long[] { 2, 3, 4 };
        var leftValues = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
        var rightValues = new[] { 10f, 20f, 30f };
        var expected = new float[24];
        for (var batch = 0; batch < 2; batch++)
        {
            for (var row = 0; row < 3; row++)
            {
                for (var column = 0; column < 4; column++)
                {
                    expected[(batch * 3 + row) * 4 + column] = leftValues[batch * 4 + column] + rightValues[row];
                }
            }
        }

        var onnxModel = CreateBinaryModel("Add", leftShape, rightShape, outputShape);
        var onnxTree = Compiler.CreateTreeFromOnnx(onnxModel);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));
        var generated = Compiler.GenerateSourceCode(onnxTree.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedBroadcastModule(generated, leftValues, leftShape, rightValues, rightShape));
        Assert.Equal(expected, ExecuteOnnx(onnxModel, leftValues, leftShape, rightValues, rightShape));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) { return left + right; }";
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        Assert.Equal(
            Assert.IsType<CompilerOnnxStep>(onnxTree.Value!.Operations.Single()).Descriptor,
            Assert.IsType<CompilerOnnxStep>(torchTree.Value!.Operations.Single()).Descriptor);
        var emitted = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value!, leftValues, leftShape, rightValues, rightShape));
    }

    [Fact]
    public void AddInt32ImportAndExportPreserveDtypeAcrossBroadcasting()
    {
        // ONNX Runtime CPU kernel: third_party/onnxruntime/onnxruntime/core/providers/cpu/math/element_wise_ops.cc (Add, opset 14, int32_t).
        // TorchLib candidate: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py::aten_add.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<int>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<int>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<int>([2, 2]));
        model.Graph.AddNode(new Onnxify.Add("add", new Onnxify.AddInputOutputOptions
        {
            A = left, B = right, C = output,
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("Add", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var leftValues = new[] { -3, 4 };
        var rightValues = new[] { 10, -2 };
        using var eagerLeft = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var eagerRight = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var eagerOutput = eagerLeft + eagerRight;
        var expected = eagerOutput.data<int>().ToArray();
        Assert.Equal([7, -5, 14, 2], expected);
        Assert.Equal(expected, ExecuteGeneratedInt32BroadcastModule(generated, leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxInt32Broadcast(emitted.Value!, leftValues, rightValues));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) { return left + right; }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        Assert.Equal("Add", Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations)).Descriptor.Name);
        var scannedOnnx = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(scannedOnnx.IsSuccess, FormatDiagnostics(scannedOnnx.Diagnostics));
        Assert.IsType<Onnxify.Add>(Assert.Single(scannedOnnx.Value!.Graph.Nodes));
    }

    [Theory]
    [InlineData("Boolean")]
    [InlineData("Complex64")]
    [InlineData("Complex128")]
    [InlineData("UInt16")]
    [InlineData("UInt32")]
    [InlineData("UInt64")]
    public void AddRejectsCpuUnsupportedElementTypes(string elementType)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var type = elementType switch
        {
            "Boolean" => OnnxTensorType.Create<bool>([2]),
            "Complex64" => OnnxTensorType.Create<Onnxify.Data.Numerics.Complex64>([2]),
            "Complex128" => OnnxTensorType.Create<Onnxify.Data.Numerics.Complex128>([2]),
            "UInt16" => OnnxTensorType.Create<ushort>([2]),
            "UInt32" => OnnxTensorType.Create<uint>([2]),
            _ => OnnxTensorType.Create<ulong>([2]),
        };
        var left = model.Graph.AddInput("left", type);
        var right = model.Graph.AddInput("right", type);
        var output = model.Graph.AddOutput("output", type);
        model.Graph.AddNode(new Onnxify.Add("add", new Onnxify.AddInputOutputOptions
        {
            A = left, B = right, C = output,
        }));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics, static item => item.Severity == CompilerDiagnosticSeverity.Error);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Contains(elementType, diagnostic.Message, StringComparison.Ordinal);
        Assert.Contains("ONNX Runtime CPU", diagnostic.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("Pow")]
    [InlineData("Sub")]
    [InlineData("Mul")]
    [InlineData("Max")]
    [InlineData("Min")]
    public void IntegerBinaryPointwiseMappingsImportAndExportPreserveDtypeAcrossBroadcasting(string opType)
    {
        // ONNX Runtime CPU opset-14 int32_t registrations: third_party/onnxruntime/onnxruntime/core/providers/cpu/math/element_wise_ops.cc.
        // TorchLib candidates: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py::aten_pow, ::aten_sub, ::aten_mul, ::aten_maximum, and ::aten_minimum.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<int>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<int>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<int>([2, 2]));
        OnnxNode node = opType switch
        {
            "Pow" => new Onnxify.Pow("arithmetic", new Onnxify.PowInputOutputOptions { X = left, Y = right, Z = output }),
            "Sub" => new Onnxify.Sub("arithmetic", new Onnxify.SubInputOutputOptions { A = left, B = right, C = output }),
            "Mul" => new Onnxify.Mul("arithmetic", new Onnxify.MulInputOutputOptions { A = left, B = right, C = output }),
            "Max" => new Onnxify.Max("arithmetic", new Onnxify.MaxInputOutputOptions { Data0 = [left, right], OutputMax = output }),
            _ => new Onnxify.Min("arithmetic", new Onnxify.MinInputOutputOptions { Data0 = [left, right], OutputMin = output }),
        };
        model.Graph.AddNode(node);

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal(opType, Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var leftValues = new[] { 12, 20 };
        var rightValues = new[] { 3, 4 };
        using var eagerLeft = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var eagerRight = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var eagerOutput = opType switch
        {
            "Pow" => eagerLeft.pow(eagerRight),
            "Sub" => eagerLeft - eagerRight,
            "Mul" => eagerLeft * eagerRight,
            "Max" => eagerLeft.maximum(eagerRight),
            _ => eagerLeft.minimum(eagerRight),
        };
        var expected = eagerOutput.data<int>().ToArray();
        var explicitExpected = opType switch
        {
            "Pow" => new[] { 1728, 20736, 8000, 160000 },
            "Sub" => new[] { 9, 8, 17, 16 },
            "Mul" => new[] { 36, 48, 60, 80 },
            "Max" => new[] { 12, 12, 20, 20 },
            _ => new[] { 3, 4, 3, 4 },
        };
        Assert.Equal(explicitExpected, expected);
        Assert.Equal(expected, ExecuteGeneratedInt32BroadcastModule(generated, leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxInt32Broadcast(emitted.Value!, leftValues, rightValues));
    }

    [Theory]
    [InlineData("Add")]
    [InlineData("Mul")]
    [InlineData("Max")]
    [InlineData("Min")]
    public void UInt8PointwiseArithmeticImportAndExportPreserveDtypeAcrossBroadcasting(string opType)
    {
        // ONNX Runtime CPU opset-14 uint8_t kernels and ONNXScript tensor_typing.py TTensor intersection.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<byte>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<byte>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<byte>([2, 2]));
        OnnxNode node = opType switch
        {
            "Add" => new Onnxify.Add("arithmetic", new Onnxify.AddInputOutputOptions { A = left, B = right, C = output }),
            "Mul" => new Onnxify.Mul("arithmetic", new Onnxify.MulInputOutputOptions { A = left, B = right, C = output }),
            "Max" => new Onnxify.Max("arithmetic", new Onnxify.MaxInputOutputOptions { Data0 = [left, right], OutputMax = output }),
            _ => new Onnxify.Min("arithmetic", new Onnxify.MinInputOutputOptions { Data0 = [left, right], OutputMin = output }),
        };
        model.Graph.AddNode(node);

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var leftValues = new byte[] { 5, 10 };
        var rightValues = new byte[] { 3, 4 };
        using var eagerLeft = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Byte);
        using var eagerRight = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Byte);
        using var eagerOutput = opType == "Add"
            ? eagerLeft + eagerRight
            : opType switch
            {
                "Mul" => eagerLeft * eagerRight,
                "Max" => eagerLeft.maximum(eagerRight),
                _ => eagerLeft.minimum(eagerRight),
            };
        var expected = eagerOutput.data<byte>().ToArray();
        var explicitExpected = opType switch
        {
            "Add" => new byte[] { 8, 9, 13, 14 },
            "Mul" => new byte[] { 15, 20, 30, 40 },
            "Max" => new byte[] { 5, 5, 10, 10 },
            _ => new byte[] { 3, 4, 3, 4 },
        };
        Assert.Equal(explicitExpected, expected);
        Assert.Equal(expected, ExecuteGeneratedUInt8BroadcastModule(generated, leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxUInt8Broadcast(emitted.Value!, leftValues, rightValues));
    }

    [Fact]
    public void PowFloatBaseAndInt32ExponentPreserveTorchLibTypePromotion()
    {
        // TorchLib aten_pow has an explicit type-promotion TODO; ONNX Runtime case: Pow_float_int32.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<float>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<int>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2, 2]));
        model.Graph.AddNode(new Onnxify.Pow("pow", new Onnxify.PowInputOutputOptions
        {
            X = left, Y = right, Z = output,
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("Pow", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var leftValues = new[] { 2f, 3f };
        var rightValues = new[] { 2, 3 };
        using var eagerLeft = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var eagerRight = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var eagerOutput = eagerLeft.pow(eagerRight);
        var expected = eagerOutput.data<float>().ToArray();
        Assert.Equal([4f, 8f, 9f, 27f], expected);
        Assert.Equal(expected, ExecuteGeneratedFloatInt32BroadcastModule(generated, leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxFloatInt32Broadcast(emitted.Value!, leftValues, rightValues));
    }

    [Theory]
    [InlineData("Boolean")]
    [InlineData("UInt32")]
    [InlineData("Complex64")]
    public void PowRejectsTypesOutsideTorchLibOnnxRuntimeCpuIntersection(string unsupportedType)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var supportedInput = model.Graph.AddInput("supported", OnnxTensorType.Create<float>([2]));
        var unsupportedTensorType = unsupportedType switch
        {
            "Boolean" => OnnxTensorType.Create<bool>([2]),
            "UInt32" => OnnxTensorType.Create<uint>([2]),
            _ => OnnxTensorType.Create<Onnxify.Data.Numerics.Complex64>([2]),
        };
        var unsupportedInput = model.Graph.AddInput("unsupported", unsupportedTensorType);
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2]));
        model.Graph.AddNode(new Onnxify.Pow("pow", new Onnxify.PowInputOutputOptions
        {
            X = supportedInput, Y = unsupportedInput, Z = output,
        }));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics, static item => item.Severity == CompilerDiagnosticSeverity.Error);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Contains(unsupportedType, diagnostic.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PowRejectsMixedFloat16InputsOnCpu()
    {
        // ONNX Runtime CPU runs Float16×Float16 Pow, while Pow_float_float16 is gated to CUDA/CoreML in element_wise_ops_test.cc.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var halfBase = model.Graph.AddInput("half_base", OnnxTensorType.Create<Half>([2]));
        var floatExponent = model.Graph.AddInput("float_exponent", OnnxTensorType.Create<float>([2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>([2]));
        model.Graph.AddNode(new Onnxify.Pow("pow", new Onnxify.PowInputOutputOptions
        {
            X = halfBase, Y = floatExponent, Z = output,
        }));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics, static item => item.Severity == CompilerDiagnosticSeverity.Error);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Contains("both inputs are Float16", diagnostic.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("Div")]
    [InlineData("Pow")]
    [InlineData("Max")]
    [InlineData("Min")]
    public void BinaryFloat16ImportAndExportPreserveDtypeAcrossBroadcasting(string opType)
    {
        // ONNX Runtime CPU: Div TestBinaryFloat16, Pow Float16 TestBinaryFloat16, Max_12_MLFloat16, and Min_12_MLFloat16.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<Half>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<Half>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>([2, 2]));
        OnnxNode node = opType switch
        {
            "Div" => new Onnxify.Div("divide", new Onnxify.DivInputOutputOptions { A = left, B = right, C = output }),
            "Pow" => new Onnxify.Pow("power", new Onnxify.PowInputOutputOptions { X = left, Y = right, Z = output }),
            "Max" => new Onnxify.Max("maximum", new Onnxify.MaxInputOutputOptions { Data0 = [left, right], OutputMax = output }),
            _ => new Onnxify.Min("minimum", new Onnxify.MinInputOutputOptions { Data0 = [left, right], OutputMin = output }),
        };
        model.Graph.AddNode(node);

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var leftValues = new[] { 1f, 3f };
        var rightValues = opType == "Div" ? new[] { 2f, 1.5f } : new[] { 2f, 0f };
        using var eagerLeft = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerRight = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerOutput = opType switch
        {
            "Div" => eagerLeft.div(eagerRight),
            "Pow" => eagerLeft.pow(eagerRight),
            "Max" => eagerLeft.maximum(eagerRight),
            _ => eagerLeft.minimum(eagerRight),
        };
        var expected = eagerOutput.data<Half>().Select(static value => (float)value).ToArray();
        var explicitExpected = opType switch
        {
            "Div" => new[] { 0.5f, 0.66650390625f, 1.5f, 2f },
            "Pow" => new[] { 1f, 1f, 9f, 1f },
            "Max" => new[] { 2f, 1f, 3f, 3f },
            _ => new[] { 1f, 0f, 2f, 0f },
        };
        Assert.Equal(explicitExpected, expected);
        Assert.Equal(expected, ExecuteGeneratedHalfBroadcastModule(generated, leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxHalfBroadcast(emitted.Value!, leftValues, rightValues));
    }

    [Theory]
    [InlineData("Max", "Boolean")]
    [InlineData("Min", "Complex64")]
    [InlineData("Max", "Int16")]
    [InlineData("Min", "UInt32")]
    public void MaxMinRejectTypesOutsideTorchLibOnnxRuntimeCpuIntersection(string opType, string elementType)
    {
        // TorchLib candidates: aten_maximum/aten_minimum in core.py; CPU types are limited by element_wise_ops.cc's Max/Min type-control lists.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var type = elementType switch
        {
            "Boolean" => OnnxTensorType.Create<bool>([2]),
            "Complex64" => OnnxTensorType.Create<Onnxify.Data.Numerics.Complex64>([2]),
            "Int16" => OnnxTensorType.Create<short>([2]),
            _ => OnnxTensorType.Create<uint>([2]),
        };
        var left = model.Graph.AddInput("left", type);
        var right = model.Graph.AddInput("right", type);
        var output = model.Graph.AddOutput("output", type);
        OnnxNode node = opType == "Max"
            ? new Onnxify.Max("maximum", new Onnxify.MaxInputOutputOptions { Data0 = [left, right], OutputMax = output })
            : new Onnxify.Min("minimum", new Onnxify.MinInputOutputOptions { Data0 = [left, right], OutputMin = output });
        model.Graph.AddNode(node);

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics, static item => item.Severity == CompilerDiagnosticSeverity.Error);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Contains(elementType, diagnostic.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("Sub", "Complex64")]
    [InlineData("Sub", "UInt8")]
    [InlineData("Mul", "Boolean")]
    public void SubAndMulRejectUnsupportedTorchLibDtypes(string opType, string elementType)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var type = elementType switch
        {
            "Boolean" => OnnxTensorType.Create<bool>([2]),
            "UInt8" => OnnxTensorType.Create<byte>([2]),
            _ => OnnxTensorType.Create<Onnxify.Data.Numerics.Complex64>([2]),
        };
        var left = model.Graph.AddInput("left", type);
        var right = model.Graph.AddInput("right", type);
        var output = model.Graph.AddOutput("output", type);
        OnnxNode node = opType == "Sub"
            ? new Onnxify.Sub("arithmetic", new Onnxify.SubInputOutputOptions { A = left, B = right, C = output })
            : new Onnxify.Mul("arithmetic", new Onnxify.MulInputOutputOptions { A = left, B = right, C = output });
        model.Graph.AddNode(node);

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics, static item => item.Severity == CompilerDiagnosticSeverity.Error);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Contains(elementType, diagnostic.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AddRejectsMismatchedKnownInputElementTypes()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<int>([2]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<float>([2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2]));
        model.Graph.AddNode(new Onnxify.Add("add", new Onnxify.AddInputOutputOptions
        {
            A = left, B = right, C = output,
        }));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics, static item => item.Severity == CompilerDiagnosticSeverity.Error);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Contains("matching input element types", diagnostic.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DivRejectsIntegerTypesWhoseOnnxSemanticsDifferFromTorchSharp()
    {
        // TorchLib aten_div is restricted to TFloat and notes that PyTorch integer inputs are promoted to float.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<int>([2]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<int>([2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<int>([2]));
        model.Graph.AddNode(new Onnxify.Div("div", new Onnxify.DivInputOutputOptions
        {
            A = left, B = right, C = output,
        }));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics, static item => item.Severity == CompilerDiagnosticSeverity.Error);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Contains("promotion semantics", diagnostic.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PointwiseAdditionPreservesFloat64DtypeFromOnnxThroughGeneratedTorchSharp()
    {
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Add_double).
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (TorchLibOpInfo("add", core_ops.aten_add));
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<double>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<double>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<double>([2, 2]));
        model.Graph.AddNode(new Onnxify.Add(
            "add",
            new Onnxify.AddInputOutputOptions
            {
                A = left,
                B = right,
                C = output,
            }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));

        var leftValues = new[] { 1d, 2d };
        var rightValues = new[] { 10d, 20d };
        var expected = new[] { 11d, 21d, 12d, 22d };
        Assert.Equal(expected, ExecuteGeneratedDoubleBroadcastModule(generated, leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxDouble(model, leftValues, rightValues));

        var emitted = Compiler.GenerateOnnx(imported.Value!);

        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Add>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnxDouble(emitted.Value, leftValues, rightValues));
    }

    [Theory]
    [InlineData("Heaviside")]
    [InlineData("XLogY")]
    public void CompositeBinaryMappingsPreserveFloat64InitializersAndBroadcasting(string mappingName)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var leftInput = model.Graph.AddInput("left", OnnxTensorType.Create<double>([new OnnxDimension<long>(2), new OnnxDimension<long>(1)]));
        var rightInput = model.Graph.AddInput("right", OnnxTensorType.Create<double>([new OnnxDimension<long>(1), new OnnxDimension<long>(2)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<double>([new OnnxDimension<long>(2), new OnnxDimension<long>(2)]));
        model.Graph.AddNode(new OnnxNode(
            mappingName.ToLowerInvariant(),
            mappingName,
            "com.onnxify.compiler",
            string.Empty,
            [new OnnxEdge(leftInput.Name), new OnnxEdge(rightInput.Name)],
            [new OnnxEdge(output.Name)],
            []));

        var tree = Compiler.CreateTreeFromOnnx(model);
        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));

        var emitted = Compiler.GenerateOnnx(tree.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var leftValues = mappingName == "XLogY" ? new[] { 2d, 0d } : new[] { -1d, 0d };
        var rightValues = mappingName == "XLogY" ? new[] { 2d, double.NaN } : new[] { 0.25d, 0.75d };
        using var left = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var right = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var expectedTensor = mappingName == "Heaviside"
            ? global::TorchSharp.torch.heaviside(left, right)
            : global::TorchSharp.torch.xlogy(left, right);
        var actual = ExecuteOnnxDoubleBroadcast(emitted.Value!, leftValues, rightValues);
        var expected = expectedTensor.data<double>().ToArray();

        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
        {
            if (double.IsNaN(expected[index]))
            {
                Assert.True(double.IsNaN(actual[index]));
            }
            else
            {
                Assert.Equal(expected[index], actual[index], 12);
            }
        }

        var imported = Compiler.CreateTreeFromOnnx(emitted.Value!);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal(mappingName, Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var reemitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(reemitted.IsSuccess, FormatDiagnostics(reemitted.Diagnostics));
        Assert.Equal(actual, ExecuteOnnxDoubleBroadcast(reemitted.Value!, leftValues, rightValues));
    }

    [Fact]
    public void IsCloseFloat64MappingKeepsDoublePrecisionToleranceInitializers()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new OnnxDimension[] { new OnnxDimension<long>(5) };
        var leftInput = model.Graph.AddInput("left", OnnxTensorType.Create<double>(shape));
        var rightInput = model.Graph.AddInput("right", OnnxTensorType.Create<double>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<bool>(shape));
        model.Graph.AddNode(new OnnxNode(
            "isclose",
            "IsClose",
            "com.onnxify.compiler",
            string.Empty,
            [new OnnxEdge(leftInput.Name), new OnnxEdge(rightInput.Name)],
            [new OnnxEdge(output.Name)],
            []));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(1e-5d, Convert.ToDouble(Assert.IsAssignableFrom<OnnxTensor>(emitted.Value!.Graph.GetValue("isclose__rtol_source")).Values[0], System.Globalization.CultureInfo.InvariantCulture));
        Assert.Equal(1e-8d, Convert.ToDouble(Assert.IsAssignableFrom<OnnxTensor>(emitted.Value.Graph.GetValue("isclose__atol_source")).Values[0], System.Globalization.CultureInfo.InvariantCulture));

        var leftValues = new[] { 0.25d, 0.5d, 0.75d, double.PositiveInfinity, double.NaN };
        var rightValues = new[] { 0.250001d, 0.50001d, 0.75d, double.PositiveInfinity, double.NaN };
        using var leftTensor = global::TorchSharp.torch.tensor(leftValues, [5L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var rightTensor = global::TorchSharp.torch.tensor(rightValues, [5L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var expectedTensor = leftTensor.isclose(rightTensor, 1e-5d, 1e-8d, false);
        var expected = expectedTensor.data<bool>().ToArray();
        Assert.Equal(expected, ExecuteOnnxDoubleBoolean(emitted.Value, leftValues, rightValues));

        var roundTrip = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(roundTrip.IsSuccess, FormatDiagnostics(roundTrip.Diagnostics));
        Assert.Equal("IsClose", Assert.IsType<CompilerOnnxStep>(Assert.Single(roundTrip.Value!.Operations)).Descriptor.Name);
        var reemitted = Compiler.GenerateOnnx(roundTrip.Value);
        Assert.True(reemitted.IsSuccess, FormatDiagnostics(reemitted.Diagnostics));
        Assert.Equal(expected, ExecuteOnnxDoubleBoolean(reemitted.Value!, leftValues, rightValues));
    }

    [Fact]
    public void LogitFloat64EpsilonKeepsDoublePrecisionAndRoundTrips()
    {
        const double epsilon = 0.123456789012345d;
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<double>([new OnnxDimension<long>(5)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<double>([new OnnxDimension<long>(5)]));
        var epsilonInitializer = model.Graph.AddTensor("epsilon", [], [epsilon]);
        model.Graph.AddNode(new OnnxNode(
            "logit",
            "Logit",
            "com.onnxify.compiler",
            string.Empty,
            [new OnnxEdge(input.Name), new OnnxEdge(epsilonInitializer.Name)],
            [new OnnxEdge(output.Name)],
            []));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        var emittedUpperBound = Assert.IsAssignableFrom<OnnxTensor>(emitted.Value!.Graph.GetValue("logit__one_minus_eps_source"));
        Assert.Equal(1d - epsilon, Convert.ToDouble(emittedUpperBound.Values[0], System.Globalization.CultureInfo.InvariantCulture));

        var inputValues = new[] { 0d, epsilon / 2d, 0.5d, 1d - (epsilon / 2d), 1d };
        using var inputTensor = global::TorchSharp.torch.tensor(inputValues, [5L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var expectedTensor = global::TorchSharp.torch.logit(inputTensor, eps: epsilon);
        var expected = expectedTensor.data<double>().ToArray();
        var actual = ExecuteOnnxDoubleUnary(emitted.Value, inputValues);
        AssertCloseDouble(expected, actual);

        var roundTrip = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(roundTrip.IsSuccess, FormatDiagnostics(roundTrip.Diagnostics));
        Assert.Equal("Logit", Assert.IsType<CompilerOnnxStep>(Assert.Single(roundTrip.Value!.Operations)).Descriptor.Name);
        var reemitted = Compiler.GenerateOnnx(roundTrip.Value);
        Assert.True(reemitted.IsSuccess, FormatDiagnostics(reemitted.Diagnostics));
        AssertCloseDouble(expected, ExecuteOnnxDoubleUnary(reemitted.Value!, inputValues));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LogitFloat16CastsConstantsToInputDtypeAndRoundTrips(bool useEpsilon)
    {
        const double epsilon = 0.1d;
        var inputValues = useEpsilon
            ? new[] { 0f, 0.05f, 0.5f, 0.95f, 1f }
            : new[] { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f };
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<Half>([new OnnxDimension<long>(inputValues.Length)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>([new OnnxDimension<long>(inputValues.Length)]));
        var inputs = new List<OnnxEdge> { new(input.Name) };
        if (useEpsilon)
        {
            inputs.Add(new OnnxEdge(model.Graph.AddTensor("epsilon", [], [epsilon]).Name));
        }

        model.Graph.AddNode(new OnnxNode(
            "logit",
            "Logit",
            "com.onnxify.compiler",
            string.Empty,
            inputs,
            [new OnnxEdge(output.Name)],
            []));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(
            useEpsilon
                ? ["CastLike", "CastLike", "LessOrEqual", "Where", "Less", "Where", "CastLike", "Sub", "Div", "Log"]
                : ["CastLike", "Sub", "Div", "Log"],
            emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        using var eagerInput = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerOutput = useEpsilon
            ? global::TorchSharp.torch.logit(eagerInput, eps: epsilon)
            : global::TorchSharp.torch.logit(eagerInput);
        var expected = eagerOutput.data<Half>().Select(static value => (float)value).ToArray();
        AssertFloat16Close(expected, ExecuteOnnxHalfUnary(emitted.Value, inputValues));

        var roundTrip = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(roundTrip.IsSuccess, FormatDiagnostics(roundTrip.Diagnostics));
        Assert.Equal("Logit", Assert.IsType<CompilerOnnxStep>(Assert.Single(roundTrip.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(roundTrip.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertFloat16Close(expected, ExecuteGeneratedHalfUnaryModule(generated, inputValues, emitted.Value));
    }

    [Theory]
    [InlineData("Deg2Rad", Math.PI / 180d, new double[] { -180d, 45d, 123.456789012345d })]
    [InlineData("Rad2Deg", 180d / Math.PI, new double[] { -Math.PI, 0.123456789012345d, Math.PI / 4d })]
    public void DegreeRadianFloat64CompositeConstantsRoundTripWithoutNarrowing(
        string mappingName,
        double scale,
        double[] inputValues)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<double>([new OnnxDimension<long>(inputValues.Length)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<double>([new OnnxDimension<long>(inputValues.Length)]));
        var scaleInitializer = model.Graph.AddTensor("scale", [], [scale]);
        model.Graph.AddNode(new Onnxify.Mul(
            mappingName,
            new Onnxify.MulInputOutputOptions
            {
                A = new OnnxEdge(input.Name),
                B = scaleInitializer,
                C = new OnnxEdge(output.Name),
            }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal(mappingName, operation.Descriptor.Name);
        var reemitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(reemitted.IsSuccess, FormatDiagnostics(reemitted.Diagnostics));
        var emittedScale = Assert.IsAssignableFrom<OnnxTensor>(reemitted.Value!.Graph.GetValue($"{mappingName}__scale"));
        Assert.Equal(scale, Convert.ToDouble(emittedScale.Values[0], System.Globalization.CultureInfo.InvariantCulture));

        using var inputTensor = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var expectedTensor = mappingName switch
        {
            "Deg2Rad" => global::TorchSharp.torch.deg2rad(inputTensor),
            "Rad2Deg" => global::TorchSharp.torch.rad2deg(inputTensor),
            _ => throw new ArgumentOutOfRangeException(nameof(mappingName)),
        };
        var expected = expectedTensor.data<double>().ToArray();
        AssertCloseDouble(expected, ExecuteOnnxDoubleUnary(model, inputValues));
        AssertCloseDouble(expected, ExecuteOnnxDoubleUnary(reemitted.Value, inputValues));
    }

    [Theory]
    [InlineData("Erfc")]
    [InlineData("Expm1")]
    public void ErrorFunctionCompositesPreserveFloat64ThroughRuntimeAndRoundTrip(string mappingName)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(3) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<double>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<double>(shape));
        var intermediate = model.Graph.AddValue("intermediate", OnnxTensorType.Create<double>(shape));
        var one = model.Graph.AddTensor("one", [], [1d]);
        if (mappingName == "Erfc")
        {
            var erf = model.Graph.AddValue("erf", OnnxTensorType.Create<double>(shape));
            model.Graph.AddNode(new Onnxify.Erf("erf", new Onnxify.ErfInputOutputOptions
            {
                Input = new OnnxEdge(input.Name),
                Output = erf,
            }));
            model.Graph.AddNode(new Onnxify.Sub("erfc", new Onnxify.SubInputOutputOptions
            {
                A = new OnnxEdge(one.Name),
                B = new OnnxEdge(erf.Name),
                C = new OnnxEdge(output.Name),
            }));
        }
        else
        {
            model.Graph.AddNode(new Onnxify.Exp("exp", new Onnxify.ExpInputOutputOptions
            {
                Input = new OnnxEdge(input.Name),
                Output = intermediate,
            }));
            model.Graph.AddNode(new Onnxify.Sub("expm1", new Onnxify.SubInputOutputOptions
            {
                A = new OnnxEdge(intermediate.Name),
                B = new OnnxEdge(one.Name),
                C = new OnnxEdge(output.Name),
            }));
        }

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal(mappingName, operation.Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var inputValues = new[] { -1d, 0d, 1d };
        AssertCloseDouble(GetDoubleUnaryEager(mappingName, inputValues), ExecuteGeneratedDoubleUnaryModule(generated, inputValues, model));

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        if (mappingName == "Expm1")
        {
            AssertCloseDouble(GetDoubleUnaryEager(mappingName, inputValues), ExecuteOnnxDoubleUnary(emitted.Value!, inputValues));
        }
        else
        {
            Assert.Equal(["Erf", "Sub"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        }
        var roundTrip = Compiler.CreateTreeFromOnnx(emitted.Value!);
        Assert.True(roundTrip.IsSuccess, FormatDiagnostics(roundTrip.Diagnostics));
        Assert.Equal(mappingName, Assert.IsType<CompilerOnnxStep>(Assert.Single(roundTrip.Value!.Operations)).Descriptor.Name);
        var reemitted = Compiler.GenerateOnnx(roundTrip.Value);
        Assert.True(reemitted.IsSuccess, FormatDiagnostics(reemitted.Diagnostics));
        if (mappingName == "Expm1")
        {
            AssertCloseDouble(GetDoubleUnaryEager(mappingName, inputValues), ExecuteOnnxDoubleUnary(reemitted.Value!, inputValues));
        }
        else
        {
            Assert.Equal(["Erf", "Sub"], reemitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        }
    }

    [Theory]
    [InlineData(2, new float[] { -1.236f, -1.234f, 1.236f, 1.234f }, new float[] { -1.24f, -1.23f, 1.24f, 1.23f })]
    [InlineData(-1, new float[] { 123f, 125f, 135f, 145f }, new float[] { 120f, 120f, 140f, 140f })]
    public void RoundWithDecimalsLowersAndRoundTrips(long decimals, float[] inputValues, float[] expectedValues)
    {
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) {{ return torch.round(input, decimals: {decimals}); }}";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(tree.Value!.Operations));
        Assert.Equal("RoundDecimals", operation.Descriptor.Name);
        Assert.Equal(decimals, Assert.IsType<OnnxAttribute<long>>(Assert.Single(operation.Node.Attributes)).Value);

        var emitted = Compiler.GenerateOnnx(tree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["CastLike", "Mul", "Round", "Div"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        Assert.Equal(expectedValues, ExecuteOnnxUnary(emitted.Value, inputValues));

        using var eagerInput = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var eagerOutput = global::TorchSharp.torch.round(eagerInput, decimals: checked((int)decimals));
        Assert.Equal(expectedValues, eagerOutput.data<float>().ToArray());

        var imported = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("RoundDecimals", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal(expectedValues, ExecuteGeneratedUnaryModule(generated, inputValues, emitted.Value));
        var reemitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(reemitted.IsSuccess, FormatDiagnostics(reemitted.Diagnostics));
        Assert.Equal(expectedValues, ExecuteOnnxUnary(reemitted.Value!, inputValues));
    }

    [Fact]
    public void RoundWithZeroDecimalsUsesStandardOnnxRound()
    {
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(
            "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return torch.round(input, decimals: 0); }"));
        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        Assert.Equal("Round", Assert.IsType<CompilerOnnxStep>(Assert.Single(tree.Value!.Operations)).Descriptor.Name);
        var emitted = Compiler.GenerateOnnx(tree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["Round"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        Assert.Equal([-2f, 0f, 2f], ExecuteOnnxUnary(emitted.Value, [-1.5f, 0.5f, 2.1f]));
    }

    [Fact]
    public void RoundWithDecimalsFloat16UsesCastLikeScaleAndMatchesTorchSharp()
    {
        // ONNX Runtime: third_party/onnxruntime/onnxruntime/core/providers/cpu/math/round.cc registers Round for MLFloat16.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(4) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<Half>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>(shape));
        model.Graph.AddNode(new OnnxNode(
            "round_decimals",
            "RoundDecimals",
            "com.onnxify.compiler",
            string.Empty,
            [new OnnxEdge(input.Name)],
            [new OnnxEdge(output.Name)],
            [new OnnxAttribute<long>("decimals", 2L)]));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["CastLike", "Mul", "Round", "Div"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        var inputValues = new[] { -1.236f, -1.234f, 1.236f, 1.234f };
        using var eagerInput = global::TorchSharp.torch.tensor(inputValues, [4L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerOutput = global::TorchSharp.torch.round(eagerInput, decimals: 2);
        var expected = eagerOutput.data<Half>().Select(static value => (float)value).ToArray();
        AssertClose(expected, ExecuteGeneratedHalfUnaryModule(generated, inputValues, model));
        AssertClose(expected, ExecuteOnnxHalfUnary(emitted.Value, inputValues));
    }

    [Fact]
    public void RoundWithDecimalsFloat64PreservesPrecisionThroughCastLikeScale()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(4) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<double>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<double>(shape));
        model.Graph.AddNode(new OnnxNode(
            "round_decimals",
            "RoundDecimals",
            "com.onnxify.compiler",
            string.Empty,
            [new OnnxEdge(input.Name)],
            [new OnnxEdge(output.Name)],
            [new OnnxAttribute<long>("decimals", 2L)]));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["CastLike", "Mul", "Round", "Div"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        var inputValues = new[] { -1.236d, -1.234d, 1.236d, 1.234d };
        using var eagerInput = global::TorchSharp.torch.tensor(inputValues, [4L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var eagerOutput = global::TorchSharp.torch.round(eagerInput, decimals: 2);
        var expected = eagerOutput.data<double>().ToArray();
        AssertCloseDouble(expected, ExecuteGeneratedDoubleUnaryModule(generated, inputValues, model));
        AssertCloseDouble(expected, ExecuteOnnxDoubleUnary(emitted.Value, inputValues));
    }

    [Theory]
    [InlineData(5L)]
    [InlineData(-8L)]
    public void ImportedRoundDecimalsRejectsScaleOutsideFloat16Range(long decimals)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(1) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<Half>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>(shape));
        model.Graph.AddNode(new OnnxNode(
            "round_decimals",
            "RoundDecimals",
            "com.onnxify.compiler",
            string.Empty,
            [new OnnxEdge(input.Name)],
            [new OnnxEdge(output.Name)],
            [new OnnxAttribute<long>("decimals", decimals)]));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.False(emitted.IsSuccess);
        Assert.Contains("outside the supported range for the input tensor dtype", FormatDiagnostics(emitted.Diagnostics), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(2L, new float[] { -1.236f, -1.234f, 1.236f, 1.234f }, new float[] { -1.24f, -1.23f, 1.24f, 1.23f })]
    [InlineData(-1L, new float[] { 123f, 125f, 135f, 145f }, new float[] { 120f, 120f, 140f, 140f })]
    public void TorchLibRoundDecimalsCastLikePowPatternImportsAndRoundTrips(long decimals, float[] inputValues, float[] expectedValues)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(inputValues.Length) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>(shape));
        var ten = model.Graph.AddTensor("ten", [], [10f]);
        var decimalsValue = model.Graph.AddTensor("decimals", [], [decimals]);
        var tenLike = model.Graph.AddValue("ten_like", OnnxTensorType.Create<float>([]));
        var decimalsLike = model.Graph.AddValue("decimals_like", OnnxTensorType.Create<float>([]));
        var scale = model.Graph.AddValue("scale", OnnxTensorType.Create<float>([]));
        var scaled = model.Graph.AddValue("scaled", OnnxTensorType.Create<float>(shape));
        var rounded = model.Graph.AddValue("rounded", OnnxTensorType.Create<float>(shape));
        model.Graph.AddNode(new OnnxNode("cast_ten", "CastLike", string.Empty, string.Empty,
            [new OnnxEdge(ten.Name), new OnnxEdge(input.Name)], [new OnnxEdge(tenLike.Name)], []));
        model.Graph.AddNode(new OnnxNode("cast_decimals", "CastLike", string.Empty, string.Empty,
            [new OnnxEdge(decimalsValue.Name), new OnnxEdge(input.Name)], [new OnnxEdge(decimalsLike.Name)], []));
        model.Graph.AddNode(new Onnxify.Pow("pow", new Onnxify.PowInputOutputOptions
        {
            X = new OnnxEdge(tenLike.Name),
            Y = new OnnxEdge(decimalsLike.Name),
            Z = new OnnxEdge(scale.Name),
        }));
        model.Graph.AddNode(new Onnxify.Mul("mul", new Onnxify.MulInputOutputOptions
        {
            A = new OnnxEdge(input.Name),
            B = new OnnxEdge(scale.Name),
            C = new OnnxEdge(scaled.Name),
        }));
        model.Graph.AddNode(new Onnxify.Round("round", new Onnxify.RoundInputOutputOptions
        {
            X = new OnnxEdge(scaled.Name),
            Y = new OnnxEdge(rounded.Name),
        }));
        model.Graph.AddNode(new Onnxify.Div("div", new Onnxify.DivInputOutputOptions
        {
            A = new OnnxEdge(rounded.Name),
            B = new OnnxEdge(scale.Name),
            C = new OnnxEdge(output.Name),
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("RoundDecimals", operation.Descriptor.Name);
        Assert.Equal(decimals, Assert.IsType<OnnxAttribute<long>>(Assert.Single(operation.Node.Attributes)).Value);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal(expectedValues, ExecuteGeneratedUnaryModule(generated, inputValues, model));

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["CastLike", "Mul", "Round", "Div"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        Assert.Equal(expectedValues, ExecuteOnnxUnary(emitted.Value, inputValues));
        var roundTrip = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(roundTrip.IsSuccess, FormatDiagnostics(roundTrip.Diagnostics));
        Assert.Equal("RoundDecimals", Assert.IsType<CompilerOnnxStep>(Assert.Single(roundTrip.Value!.Operations)).Descriptor.Name);
    }

    [Fact]
    public void TorchLibSincCastLikePatternImportsAndRoundTrips()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(4) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>(shape));
        var pi = model.Graph.AddTensor("pi", [], [Math.PI]);
        var zero = model.Graph.AddTensor("zero", [], [0L]);
        var one = model.Graph.AddTensor("one", [], [1L]);
        var piLike = model.Graph.AddValue("pi_like", OnnxTensorType.Create<float>([]));
        var piInput = model.Graph.AddValue("pi_input", OnnxTensorType.Create<float>(shape));
        var zeroLike = model.Graph.AddValue("zero_like", OnnxTensorType.Create<float>([]));
        var condition = model.Graph.AddValue("condition", OnnxTensorType.Create<bool>(shape));
        var oneLike = model.Graph.AddValue("one_like", OnnxTensorType.Create<float>([]));
        var sine = model.Graph.AddValue("sine", OnnxTensorType.Create<float>(shape));
        var ratio = model.Graph.AddValue("ratio", OnnxTensorType.Create<float>(shape));
        model.Graph.AddNode(new OnnxNode("cast_pi", "CastLike", string.Empty, string.Empty,
            [new OnnxEdge(pi.Name), new OnnxEdge(input.Name)], [new OnnxEdge(piLike.Name)], []));
        model.Graph.AddNode(new Onnxify.Mul("multiply", new Onnxify.MulInputOutputOptions
        {
            A = new OnnxEdge(input.Name), B = new OnnxEdge(piLike.Name), C = new OnnxEdge(piInput.Name),
        }));
        model.Graph.AddNode(new OnnxNode("cast_zero", "CastLike", string.Empty, string.Empty,
            [new OnnxEdge(zero.Name), new OnnxEdge(input.Name)], [new OnnxEdge(zeroLike.Name)], []));
        model.Graph.AddNode(new Onnxify.Equal("equal", new Onnxify.EqualInputOutputOptions
        {
            A = new OnnxEdge(input.Name), B = new OnnxEdge(zeroLike.Name), C = new OnnxEdge(condition.Name),
        }));
        model.Graph.AddNode(new OnnxNode("cast_one", "CastLike", string.Empty, string.Empty,
            [new OnnxEdge(one.Name), new OnnxEdge(input.Name)], [new OnnxEdge(oneLike.Name)], []));
        model.Graph.AddNode(new Onnxify.Sin("sin", new Onnxify.SinInputOutputOptions
        {
            Input = new OnnxEdge(piInput.Name), Output = new OnnxEdge(sine.Name),
        }));
        model.Graph.AddNode(new Onnxify.Div("divide", new Onnxify.DivInputOutputOptions
        {
            A = new OnnxEdge(sine.Name), B = new OnnxEdge(piInput.Name), C = new OnnxEdge(ratio.Name),
        }));
        model.Graph.AddNode(new Onnxify.Where("where", new Onnxify.WhereInputOutputOptions
        {
            Condition = new OnnxEdge(condition.Name), X = new OnnxEdge(oneLike.Name),
            Y = new OnnxEdge(ratio.Name), Output = new OnnxEdge(output.Name),
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("Sinc", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var values = new[] { -1f, 0f, 0.5f, 1f };
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertClose([0f, 1f, 0.6366198f, 0f], ExecuteGeneratedUnaryModule(generated, values, model));

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["Mul", "Sin", "Div", "Equal", "Where"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        AssertClose([0f, 1f, 0.6366198f, 0f], ExecuteOnnxUnary(emitted.Value, values));
        var roundTrip = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(roundTrip.IsSuccess, FormatDiagnostics(roundTrip.Diagnostics));
        Assert.Equal("Sinc", Assert.IsType<CompilerOnnxStep>(Assert.Single(roundTrip.Value!.Operations)).Descriptor.Name);
    }

    [Theory]
    [InlineData("torch.round(input, decimals: decimals)")]
    [InlineData("input.round(decimals: decimals)")]
    [InlineData("torch.round(input, decimals: 100)")]
    public void RoundWithUnsupportedDecimalsAreRejected(string call)
    {
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, int decimals) {{ return {call}; }}";
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.NotNull(diagnostic.Span);

        var yValues = new[] { float.NaN, float.PositiveInfinity, float.PositiveInfinity };
        var xValues = new[] { 1f, float.PositiveInfinity, float.NegativeInfinity };
        using var y = global::TorchSharp.torch.tensor(yValues);
        using var x = global::TorchSharp.torch.tensor(xValues);
        using var eager = global::TorchSharp.torch.atan2(y, x);
        var eagerValues = eager.data<float>().ToArray();
        Assert.True(float.IsNaN(eagerValues[0]));
        Assert.Equal(MathF.PI / 4f, eagerValues[1]);
        Assert.Equal(3f * MathF.PI / 4f, eagerValues[2]);
    }

    private static double[] GetDoubleUnaryEager(string mappingName, double[] inputValues)
    {
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var output = mappingName switch
        {
            "Erfc" => global::TorchSharp.torch.erfc(input),
            "Expm1" => global::TorchSharp.torch.expm1(input),
            _ => throw new ArgumentOutOfRangeException(nameof(mappingName)),
        };
        return output.data<double>().ToArray();
    }

    [Fact]
    public void TruncMappingLowersToRuntimeSupportedOnnxAndDeclaresItsDirection()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (trunc), exercised by ops_test.py::test_output_match_opinfo_.
        // ONNX Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (ModOpTest.Fmod_float_mixed_sign).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.trunc(); }";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(tree.Value!.Operations));
        Assert.Equal("Trunc", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(tree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["Mod", "Sub"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        var inputValues = new[] { -1.9f, 0.5f, 2.1f };
        var expectedValues = ExecuteOnnxUnary(emitted.Value, inputValues);
        AssertClose([-1f, 0f, 2f], expectedValues);
        AssertClose(expectedValues, AssertCompositeRoundTrip(emitted.Value, "Trunc", inputValues));
    }

    [Fact]
    public void Exp2PatternRejectsChangedConstantsAndEscapedIntermediateValues()
    {
        var changedConstant = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var changedInput = changedConstant.Graph.AddInput("input", OnnxTensorType.Create<float>([3]));
        var changedOutput = changedConstant.Graph.AddOutput("output", OnnxTensorType.Create<float>([3]));
        var three = changedConstant.Graph.AddTensor("three", [], [3f]);
        changedConstant.Graph.AddNode(new Onnxify.Pow(
            "power",
            new Onnxify.PowInputOutputOptions
            {
                X = three,
                Y = changedInput,
                Z = changedOutput,
            }));

        var changedTree = Compiler.CreateTreeFromOnnx(changedConstant);
        Assert.True(changedTree.IsSuccess, FormatDiagnostics(changedTree.Diagnostics));
        Assert.DoesNotContain(changedTree.Value!.Operations.OfType<CompilerOnnxStep>(),
            static operation => operation.Descriptor.Name == "Exp2");
        Assert.Equal("Pow", Assert.IsType<CompilerOnnxStep>(Assert.Single(changedTree.Value.Operations)).Descriptor.Name);

        var escapedIntermediate = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = escapedIntermediate.Graph.AddInput("input", OnnxTensorType.Create<float>([3]));
        var output = escapedIntermediate.Graph.AddOutput("output", OnnxTensorType.Create<float>([3]));
        var extraOutput = escapedIntermediate.Graph.AddOutput("extra_output", OnnxTensorType.Create<float>([3]));
        var one = escapedIntermediate.Graph.AddTensor("one", [], [1f]);
        var sum = new OnnxEdge("sum");
        escapedIntermediate.Graph.AddNode(new Onnxify.Add(
            "add",
            new Onnxify.AddInputOutputOptions
            {
                A = input,
                B = one,
                C = sum,
            }));
        escapedIntermediate.Graph.AddNode(new Onnxify.Log(
            "log",
            new Onnxify.LogInputOutputOptions
            {
                Input = sum,
                Output = output,
            }));
        escapedIntermediate.Graph.AddNode(new Onnxify.Neg(
            "negate",
            new Onnxify.NegInputOutputOptions
            {
                X = sum,
                Y = extraOutput,
            }));

        var escapedTree = Compiler.CreateTreeFromOnnx(escapedIntermediate);
        Assert.True(escapedTree.IsSuccess, FormatDiagnostics(escapedTree.Diagnostics));
        Assert.DoesNotContain(escapedTree.Value!.Operations.OfType<CompilerOnnxStep>(),
            static operation => operation.Descriptor.Name == "Log1p");
        Assert.Equal(["Add", "Log", "Neg"], escapedTree.Value.Operations
            .OfType<CompilerOnnxStep>()
            .Select(static operation => operation.Descriptor.Name));
    }

    [Fact]
    public void Exp2CompositeRejectsIntegerInputWithTorchLibTypeDiagnostic()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<int>([3]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([3]));
        var two = model.Graph.AddTensor("two", [], [2f]);
        model.Graph.AddNode(new Onnxify.Pow(
            "power",
            new Onnxify.PowInputOutputOptions
            {
                X = two,
                Y = input,
                Z = output,
            }));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("supports Float16/Float32/Float64 tensor inputs", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("Log1p")]
    [InlineData("Log10")]
    [InlineData("Log2")]
    [InlineData("Rsqrt")]
    public void CompositeUnaryMappingsRejectFloat16WhenCpuComponentsDoNotSupportIt(string mappingName)
    {
        // ONNX Runtime CPU has no Float16 registration for Log, Sqrt, or Reciprocal in element_wise_ops.cc.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<Half>([3]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>([3]));
        if (mappingName == "Rsqrt")
        {
            var squareRoot = model.Graph.AddValue("square_root", OnnxTensorType.Create<Half>([3]));
            model.Graph.AddNode(new Onnxify.Sqrt("sqrt", new Onnxify.SqrtInputOutputOptions
            {
                X = input,
                Y = squareRoot,
            }));
            model.Graph.AddNode(new Onnxify.Reciprocal("reciprocal", new Onnxify.ReciprocalInputOutputOptions
            {
                X = new OnnxEdge(squareRoot.Name),
                Y = new OnnxEdge(output.Name),
            }));
        }
        else if (mappingName == "Log1p")
        {
            var one = model.Graph.AddTensor("one", [], [(Half)1f]);
            var sum = model.Graph.AddValue("sum", OnnxTensorType.Create<Half>([3]));
            model.Graph.AddNode(new Onnxify.Add("add_one", new Onnxify.AddInputOutputOptions
            {
                A = input,
                B = one,
                C = sum,
            }));
            model.Graph.AddNode(new Onnxify.Log("log", new Onnxify.LogInputOutputOptions
            {
                Input = new OnnxEdge(sum.Name),
                Output = new OnnxEdge(output.Name),
            }));
        }
        else
        {
            var logBase = model.Graph.AddTensor("log_base", [], [(Half)(mappingName == "Log10" ? 10f : 2f)]);
            var baseLog = model.Graph.AddValue("base_log", OnnxTensorType.Create<Half>([]));
            var inputLog = model.Graph.AddValue("input_log", OnnxTensorType.Create<Half>([3]));
            model.Graph.AddNode(new Onnxify.Log("log_base_node", new Onnxify.LogInputOutputOptions
            {
                Input = logBase,
                Output = baseLog,
            }));
            model.Graph.AddNode(new Onnxify.Log("log_input_node", new Onnxify.LogInputOutputOptions
            {
                Input = input,
                Output = inputLog,
            }));
            model.Graph.AddNode(new Onnxify.Div("divide", new Onnxify.DivInputOutputOptions
            {
                A = new OnnxEdge(inputLog.Name),
                B = new OnnxEdge(baseLog.Name),
                C = new OnnxEdge(output.Name),
            }));
        }

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("Float16", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("Lerp", 3)]
    [InlineData("LogAddExp", 2)]
    [InlineData("LogAddExp2", 2)]
    public void CompositeTensorMappingsRejectFloat16WithoutSupportedCpuDecomposition(string mappingName, int inputCount)
    {
        // Lerp also uses Less and Where, whose CPU/TorchLib intersection excludes Float16; LogAddExp variants use Float16-unsupported math components.
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var type = OnnxTensorType.Create<Half>([2]);
        var inputs = Enumerable.Range(0, inputCount)
            .Select(index => model.Graph.AddInput($"input{index}", type))
            .ToArray();
        var output = model.Graph.AddOutput("output", type);
        model.Graph.AddNode(new OnnxNode(
            $"unsupported_{mappingName.ToLowerInvariant()}",
            mappingName,
            "com.onnxify.compiler",
            string.Empty,
            inputs.Select(static input => (IOnnxGraphEdge)new OnnxEdge(input.Name)),
            [new OnnxEdge(output.Name)],
            []));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("Float16", StringComparison.Ordinal));
    }

    [Fact]
    public void CompositePatternsRejectChangedConnectionsAndAttributes()
    {
        const string unarySource = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return torch.log1p(input); }";
        var unaryTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(unarySource));
        Assert.True(unaryTree.IsSuccess, FormatDiagnostics(unaryTree.Diagnostics));
        var unaryModel = Compiler.GenerateOnnx(unaryTree.Value!);
        Assert.True(unaryModel.IsSuccess, FormatDiagnostics(unaryModel.Diagnostics));
        var log = Assert.Single(unaryModel.Value!.Graph.Nodes.OfType<Onnxify.Log>());
        log.Input = new OnnxEdge("input");

        var connectionNearMiss = Compiler.CreateTreeFromOnnx(unaryModel.Value);
        Assert.True(connectionNearMiss.IsSuccess, FormatDiagnostics(connectionNearMiss.Diagnostics));
        Assert.DoesNotContain(connectionNearMiss.Value!.Operations.OfType<CompilerOnnxStep>(),
            static operation => operation.Descriptor.Name == "Log1p");

        const string binarySource = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) { return torch.logaddexp(left, right); }";
        var binaryTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(binarySource));
        Assert.True(binaryTree.IsSuccess, FormatDiagnostics(binaryTree.Diagnostics));
        var binaryModel = Compiler.GenerateOnnx(binaryTree.Value!);
        Assert.True(binaryModel.IsSuccess, FormatDiagnostics(binaryModel.Diagnostics));
        var isInf = binaryModel.Value!.Graph.Nodes.OfType<Onnxify.IsInf>().First();
        isInf.DetectNegative = 0;

        var attributeNearMiss = Compiler.CreateTreeFromOnnx(binaryModel.Value);
        Assert.True(attributeNearMiss.IsSuccess, FormatDiagnostics(attributeNearMiss.Diagnostics));
        Assert.DoesNotContain(attributeNearMiss.Value!.Operations.OfType<CompilerOnnxStep>(),
            static operation => operation.Descriptor.Name == "LogAddExp");
    }

    [Fact]
    public void RemainderMappingLowersToRuntimeSupportedOnnxAndMatchesTorchSharp()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (TorchLibOpInfo("remainder", core_ops.aten_remainder)).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (ModOpTest.Int32_mixed_sign and ModOpTest.Int32_mod_bcast).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) { return left.remainder(right); }";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("Mod", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["Div", "Floor", "Mul", "Sub"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        var leftValues = new[] { -5.5f, 5.5f, -5.5f, 5.5f, -2.5f, 2.5f };
        var rightValues = new[] { 2f, -2f, 3f };
        var expected = new[] { 0.5f, -0.5f, 0.5f, 1.5f, -0.5f, 2.5f };
        AssertClose(expected, ExecuteOnnx(emitted.Value, leftValues, [2, 3], rightValues, [3]));

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertClose(expected, ExecuteGeneratedModule(generated, leftValues, rightValues));
    }

    [Fact]
    public void RemainderAndFmodImportPreserveTheirDistinctTorchSemantics()
    {
        var model = CreateModModel(fmod: 0);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("Mod", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("left.remainder(right)", generated.SourceCode());

        var fmodModel = CreateModModel(fmod: 1);
        var fmodImported = Compiler.CreateTreeFromOnnx(fmodModel);
        Assert.True(fmodImported.IsSuccess, FormatDiagnostics(fmodImported.Diagnostics));
        var fmodOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(fmodImported.Value!.Operations));
        Assert.Equal("Fmod", fmodOperation.Descriptor.Name);
        var fmodSource = Compiler.GenerateSourceCode(fmodImported.Value);
        Assert.True(fmodSource.IsSuccess, FormatDiagnostics(fmodSource.Diagnostics));
        Assert.Contains("torch.fmod(left, right)", fmodSource.SourceCode());
        var fmodReemitted = Compiler.GenerateOnnx(fmodImported.Value);
        Assert.True(fmodReemitted.IsSuccess, FormatDiagnostics(fmodReemitted.Diagnostics));
        Assert.Equal(1L, Assert.IsType<Onnxify.Mod>(Assert.Single(fmodReemitted.Value!.Graph.Nodes)).Fmod);
    }

    [Fact]
    public void FmodMappingUsesOnnxFmodSemanticsAndMatchesTorchSharp()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (TorchLibOpInfo("fmod", core_ops.aten_fmod)).
        // ONNX Runtime kernel: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (ModOpTest).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) { return torch.fmod(left, right); }";
        var imported = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("Fmod", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        var mod = Assert.IsType<Onnxify.Mod>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(1L, mod.Fmod);

        var leftValues = new[] { -5.5f, 5.5f, -5.5f, 5.5f, -2.5f, 2.5f };
        var rightValues = new[] { 2f, -2f, 3f };
        var expected = new[] { -1.5f, 1.5f, -2.5f, 1.5f, -0.5f, 2.5f };
        AssertClose(expected, ExecuteOnnx(emitted.Value, leftValues, [2, 3], rightValues, [3]));

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertClose(expected, ExecuteGeneratedModule(generated, leftValues, rightValues));
    }

    [Fact]
    public void FmodFloat64ImportPreservesDtypeThroughOnnxAndGeneratedTorchSharp()
    {
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (ModOpTest.Fmod_double_mixed_sign).
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (TorchLibOpInfo("fmod", core_ops.aten_fmod)).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<double>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<double>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<double>([2, 2]));
        model.Graph.AddNode(new Onnxify.Mod("fmod", new Onnxify.ModInputOutputOptions
        {
            A = left, B = right, Fmod = 1, C = output,
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("Fmod", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));

        var leftValues = new[] { -5.5d, 5.5d };
        var rightValues = new[] { 2d, -2d };
        using var leftTensor = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var rightTensor = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var expectedTensor = global::TorchSharp.torch.fmod(leftTensor, rightTensor);
        var expected = expectedTensor.data<double>().ToArray();
        AssertCloseDouble(expected, ExecuteGeneratedDoubleBroadcastModule(generated, leftValues, rightValues));
        AssertCloseDouble(expected, ExecuteOnnxDoubleBroadcast(model, leftValues, rightValues));

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(1L, Assert.IsType<Onnxify.Mod>(Assert.Single(emitted.Value!.Graph.Nodes)).Fmod);
        AssertCloseDouble(expected, ExecuteOnnxDoubleBroadcast(emitted.Value, leftValues, rightValues));
    }

    [Fact]
    public void FmodFloat16ImportPreservesDtypeAndMatchesOnnxRuntimeAndTorchSharp()
    {
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (ModOpTest.Fmod_float16_mixed_sign).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<Half>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<Half>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>([2, 2]));
        model.Graph.AddNode(new Onnxify.Mod("fmod", new Onnxify.ModInputOutputOptions
        {
            A = left, B = right, Fmod = 1, C = output,
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("Fmod", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var leftValues = new[] { -5.5f, 5.5f };
        var rightValues = new[] { 2f, -2f };
        using var leftTensor = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var rightTensor = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var expectedTensor = global::TorchSharp.torch.fmod(leftTensor, rightTensor);
        var expected = expectedTensor.data<Half>().Select(static value => (float)value).ToArray();
        var actual = ExecuteOnnxHalfBroadcast(emitted.Value!, leftValues, rightValues);
        AssertClose(expected, actual);
    }

    [Fact]
    public void FmodInt32ImportPreservesDtypeThroughOnnxAndGeneratedTorchSharp()
    {
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (ModOpTest.Int32_mixed_sign_fmod).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<int>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<int>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<int>([2, 2]));
        model.Graph.AddNode(new Onnxify.Mod("fmod", new Onnxify.ModInputOutputOptions
        {
            A = left, B = right, Fmod = 1, C = output,
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("Fmod", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var leftValues = new[] { -5, 5 };
        var rightValues = new[] { 2, -2 };
        using var leftTensor = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var rightTensor = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var expectedTensor = global::TorchSharp.torch.fmod(leftTensor, rightTensor);
        var expected = expectedTensor.data<int>().ToArray();
        Assert.Equal(expected, ExecuteGeneratedInt32BroadcastModule(generated, leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxInt32Broadcast(emitted.Value!, leftValues, rightValues));
    }

    [Fact]
    public void FmodInt64ImportPreservesDtypeThroughOnnxAndGeneratedTorchSharp()
    {
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (ModOpTest.Int64_mixed_sign_fmod).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<long>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<long>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<long>([2, 2]));
        model.Graph.AddNode(new Onnxify.Mod("fmod", new Onnxify.ModInputOutputOptions
        {
            A = left, B = right, Fmod = 1, C = output,
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("Fmod", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var leftValues = new[] { -5L, 5L };
        var rightValues = new[] { 2L, -2L };
        using var leftTensor = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var rightTensor = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var expectedTensor = global::TorchSharp.torch.fmod(leftTensor, rightTensor);
        var expected = expectedTensor.data<long>().ToArray();
        Assert.Equal(expected, ExecuteGeneratedInt64BroadcastModule(generated, leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxInt64Broadcast(emitted.Value!, leftValues, rightValues));
    }

    [Fact]
    public void FmodRemainingTorchLibIntegerTypesMatchOnnxRuntimeAndTorchSharp()
    {
        AssertFmodIntegerType(
            new sbyte[] { -5, 5 }, new sbyte[] { 2, -2 },
            static (values, shape) => global::TorchSharp.torch.tensor(values, shape, dtype: global::TorchSharp.torch.ScalarType.Int8),
            static tensor => tensor.data<sbyte>().ToArray());
        AssertFmodIntegerType(
            new short[] { -5, 5 }, new short[] { 2, -2 },
            static (values, shape) => global::TorchSharp.torch.tensor(values, shape, dtype: global::TorchSharp.torch.ScalarType.Int16),
            static tensor => tensor.data<short>().ToArray());
        AssertFmodIntegerType(
            new byte[] { 5, 8 }, new byte[] { 2, 3 },
            static (values, shape) => global::TorchSharp.torch.tensor(values, shape, dtype: global::TorchSharp.torch.ScalarType.Byte),
            static tensor => tensor.data<byte>().ToArray());
    }

    [Fact]
    public void IntegerRemainderKeepsNativeOnnxModAndBroadcasts()
    {
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (ModOpTest.Int32_mixed_sign, ModOpTest.Int32_mod_bcast).
        var model = CreateInt32ModModel();
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal("Mod", Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        Assert.Equal([1, -1, 1, 1, 0, 2], ExecuteOnnxInt32(emitted.Value!));
    }

    [Fact]
    public void PointwiseWhereMappingBroadcastsConditionAndValuesInBothDirections()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_where).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/where_op_test.cc (WhereOpTest.Broadcast).
        var model = CreateWhereModel();
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("Where", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);
        var source = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(source.IsSuccess, FormatDiagnostics(source.Diagnostics));
        Assert.Contains("torch.where(condition, left, right)", source.SourceCode());

        var conditions = new[] { true, false, true, false, true, false };
        var leftValues = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var rightValues = new[] { 10f, 20f, 30f };
        var expectedValues = new[] { 1f, 20f, 3f, 10f, 5f, 30f };
        AssertClose(expectedValues, ExecuteWhereModule(source, conditions, leftValues, rightValues));

        const string torchSource = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor condition, global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) { return torch.where(condition, left, right); }";
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(torchSource));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var torchOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations));
        Assert.Equal(operation.Descriptor, torchOperation.Descriptor);
        var emitted = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(expectedValues, ExecuteWhereOnnx(emitted.Value!, conditions, leftValues, rightValues));
    }

    [Fact]
    public void PointwiseCastMappingPreservesTargetDtypeInBothDirections()
    {
        // No Cast converter entry exists in third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py; the TorchSharp source uses Tensor.to_type with a static ScalarType.
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/cast_op_test.cc (CastOpTest.NonStringTypes).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([new OnnxDimension<long>(3)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<long>([new OnnxDimension<long>(3)]));
        model.Graph.AddNode("cast", "Cast", string.Empty, string.Empty, [input], [output],
            [new OnnxAttribute<long>("to", 7L)]);

        var onnxTree = Compiler.CreateTreeFromOnnx(model);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));
        var onnxOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(onnxTree.Value!.Operations));
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxOperation.Descriptor.Capability);
        var generated = Compiler.GenerateSourceCode(onnxTree.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("to_type(torch.ScalarType.Int64)", generated.SourceCode());
        Assert.Equal([1L, -2L, 3L], ExecuteGeneratedCastModule(generated, [1.9f, -2.2f, 3.8f]));

        const string torchSource = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.to_type(torch.ScalarType.Int64); }";
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(torchSource));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var torchOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations));
        Assert.Equal(onnxOperation.Descriptor, torchOperation.Descriptor);
        var emitted = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(7L, Assert.IsType<Onnxify.Cast>(torchOperation.Node).To);
        Assert.Equal([1L, -2L, 3L], ExecuteOnnxCast(emitted.Value!, [1.9f, -2.2f, 3.8f]));
    }

    [Theory]
    [InlineData("input.to_type(dtype)")]
    [InlineData("input.to_type(torch.ScalarType.ComplexFloat32)")]
    public void CastRejectsDynamicOrUnsupportedTargetTypes(string castExpression)
    {
        // No Cast converter entry exists in third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py; the compiler supports only statically named TorchSharp ScalarType targets.
        // ONNX Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/cast_op_test.cc (CastOpTest.NonStringTypes).
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.ScalarType dtype) {{ return {castExpression}; }}";
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.NotNull(diagnostic.Span);
    }

    [Theory]
    [InlineData("torch.exp2(input)", "Exp2", new string[] { "Pow" }, new float[] { -2f, 0f, 3f }, new float[] { 0.25f, 1f, 8f })]
    [InlineData("torch.log1p(input)", "Log1p", new string[] { "Add", "Log" }, new float[] { 0f, 1f, 3f }, new float[] { 0f, 0.6931472f, 1.3862944f })]
    [InlineData("torch.log10(input)", "Log10", new string[] { "Log", "Log", "Div" }, new float[] { 0.1f, 1f, 10f }, new float[] { -1f, 0f, 1f })]
    [InlineData("torch.log2(input)", "Log2", new string[] { "Log", "Log", "Div" }, new float[] { 0.25f, 1f, 8f }, new float[] { -2f, 0f, 3f })]
    [InlineData("torch.erfc(input)", "Erfc", new string[] { "Erf", "Sub" }, new float[] { -1f, 0f, 1f }, new float[] { 1.8427008f, 1f, 0.1572992f })]
    [InlineData("torch.expm1(input)", "Expm1", new string[] { "Exp", "Sub" }, new float[] { -1f, 0f, 1f }, new float[] { -0.63212055f, 0f, 1.7182817f })]
    [InlineData("torch.sinc(input)", "Sinc", new string[] { "Mul", "Sin", "Div", "Equal", "Where" }, new float[] { -1f, 0f, 0.5f, 1f }, new float[] { 0f, 1f, 0.6366198f, 0f })]
    [InlineData("torch.deg2rad(input)", "Deg2Rad", new string[] { "Mul" }, new float[] { -180f, 0f, 90f }, new float[] { -MathF.PI, 0f, MathF.PI / 2f })]
    [InlineData("torch.rad2deg(input)", "Rad2Deg", new string[] { "Mul" }, new float[] { -MathF.PI, 0f, MathF.PI / 2f }, new float[] { -180f, 0f, 90f })]
    [InlineData("torch.rsqrt(input)", "Rsqrt", new string[] { "Sqrt", "Reciprocal" }, new float[] { 0.25f, 1f, 4f }, new float[] { 2f, 1f, 0.5f })]
    [InlineData("torch.frac(input)", "Frac", new string[] { "Abs", "Floor", "Sign", "Mul", "Sub" }, new float[] { -1.5f, -0.25f, 0f, 1.75f }, new float[] { -0.5f, -0.25f, 0f, 0.75f })]
    [InlineData("torch.logit(input)", "Logit", new string[] { "CastLike", "Sub", "Div", "Log" }, new float[] { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f }, new float[] { -2.1972246f, -1.0986123f, 0f, 1.0986123f, 2.1972246f })]
    [InlineData("torch.logit(input, eps: 0.1)", "Logit", new string[] { "CastLike", "CastLike", "LessOrEqual", "Where", "Less", "Where", "CastLike", "Sub", "Div", "Log" }, new float[] { 0f, 0.05f, 0.5f, 0.95f, 1f }, new float[] { -2.1972246f, -2.1972246f, 0f, 2.1972246f, 2.1972246f })]
    public void CompositeUnaryPointwiseMappingsLowerToOnnxAndMatchRuntime(
        string torchExpression,
        string mappingName,
        string[] expectedOperators,
        float[] inputValues,
        float[] expectedValues
    )
    {
        // TorchLib converter cases: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py and special.py (aten_exp2, aten_log1p, aten_log10, aten_log2, aten_deg2rad, aten_rad2deg, aten_rsqrt, aten_frac, aten_special_erfc, aten_special_expm1, aten_special_sinc).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Pow_Float, MathOpTest.Log, MathOpTest.Exp_float, MathOpTest.Abs, MathOpTest.Floor, MathOpTest.Sign, MathOpTest.Mul, MathOpTest.Sub).
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) {{ return {torchExpression}; }}";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(tree.Value!.Operations));
        Assert.Equal(mappingName, operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(tree.Value);

        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(expectedOperators, emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        var actualValues = ExecuteOnnxUnary(emitted.Value, inputValues);
        AssertClose(expectedValues, actualValues);

        using var torchInput = global::TorchSharp.torch.tensor(
            inputValues,
            [inputValues.Length],
            dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var torchOutput = mappingName switch
        {
            "Exp2" => global::TorchSharp.torch.exp2(torchInput),
            "Log1p" => global::TorchSharp.torch.log1p(torchInput),
            "Log10" => global::TorchSharp.torch.log10(torchInput),
            "Log2" => global::TorchSharp.torch.log2(torchInput),
            "Erfc" => global::TorchSharp.torch.erfc(torchInput),
            "Expm1" => global::TorchSharp.torch.expm1(torchInput),
            "Sinc" => global::TorchSharp.torch.sinc(torchInput),
            "Deg2Rad" => global::TorchSharp.torch.deg2rad(torchInput),
            "Rad2Deg" => global::TorchSharp.torch.rad2deg(torchInput),
            "Rsqrt" => global::TorchSharp.torch.rsqrt(torchInput),
            "Frac" => global::TorchSharp.torch.frac(torchInput),
            "Logit" when expectedOperators.Length == 3 => global::TorchSharp.torch.logit(torchInput),
            "Logit" => global::TorchSharp.torch.logit(torchInput, eps: 0.1),
            _ => throw new ArgumentOutOfRangeException(nameof(mappingName)),
        };
        AssertClose(torchOutput.data<float>().ToArray(), actualValues);

        AssertClose(actualValues, AssertCompositeRoundTrip(emitted.Value, mappingName, inputValues));
    }

    [Theory]
    [InlineData("torch.exp2(input)", "Exp2", new float[] { -2f, 0f, 3f })]
    [InlineData("torch.deg2rad(input)", "Deg2Rad", new float[] { -180f, 0f, 90f })]
    [InlineData("torch.rad2deg(input)", "Rad2Deg", new float[] { -3.1415927f, 0f, 1.5707964f })]
    public void Float16FloatingCompositeMappingsRoundTrip(string torchExpression, string mappingName, float[] inputValues)
    {
        // TorchLib: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py (aten_exp2, aten_deg2rad, aten_rad2deg); all use TFloat, including Float16.
        // Runtime: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Pow_Float and Mul registrations).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(inputValues.Length) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<Half>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>(shape));
        var constant = model.Graph.AddTensor(
            "constant",
            [],
            [mappingName switch
            {
                "Exp2" => 2f,
                "Deg2Rad" => (float)(Math.PI / 180d),
                "Rad2Deg" => (float)(180d / Math.PI),
                _ => throw new ArgumentOutOfRangeException(nameof(mappingName)),
            }]);
        var typedConstant = new OnnxEdge("typed_constant");
        model.Graph.AddNode(new OnnxNode(
            "constant_cast",
            "CastLike",
            string.Empty,
            string.Empty,
            [constant, new OnnxEdge(input.Name)],
            [typedConstant],
            []));
        if (mappingName == "Exp2")
        {
            model.Graph.AddNode(new Onnxify.Pow("exp2", new Onnxify.PowInputOutputOptions
            {
                X = typedConstant,
                Y = new OnnxEdge(input.Name),
                Z = new OnnxEdge(output.Name),
            }));
        }
        else
        {
            model.Graph.AddNode(new Onnxify.Mul($"{mappingName}_mul", new Onnxify.MulInputOutputOptions
            {
                A = new OnnxEdge(input.Name),
                B = typedConstant,
                C = new OnnxEdge(output.Name),
            }));
        }

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal(mappingName, Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);

        var reemitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(reemitted.IsSuccess, FormatDiagnostics(reemitted.Diagnostics));
        var expectedOperators = mappingName == "Exp2" ? new[] { "CastLike", "Pow" } : ["CastLike", "Mul"];
        Assert.Equal(expectedOperators, reemitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        using var torchInput = global::TorchSharp.torch.tensor(
            inputValues,
            [inputValues.Length],
            dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var torchOutput = mappingName switch
        {
            "Exp2" => global::TorchSharp.torch.exp2(torchInput),
            "Deg2Rad" => global::TorchSharp.torch.deg2rad(torchInput),
            "Rad2Deg" => global::TorchSharp.torch.rad2deg(torchInput),
            _ => throw new ArgumentOutOfRangeException(nameof(mappingName)),
        };
        var expectedValues = torchOutput.data<Half>().Select(static value => (float)value).ToArray();
        AssertFloat16Close(expectedValues, ExecuteOnnxHalfUnary(model, inputValues));
        AssertFloat16Close(expectedValues, ExecuteOnnxHalfUnary(reemitted.Value, inputValues));
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertFloat16Close(expectedValues, ExecuteGeneratedHalfUnaryModule(generated, inputValues, model));

        var torchSource = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) {{ return {torchExpression}; }}";
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(torchSource));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        Assert.Equal(mappingName, Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations)).Descriptor.Name);
    }

    [Fact]
    public void ErfcFloat16ImportsEmitsAndRoundTripsCastLikePattern()
    {
        // TorchLib: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/special.py::aten_special_erfc (TFloat includes Float16).
        // Runtime: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Erf Float16 registration; Sub Float16 registration).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(3) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<Half>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>(shape));
        var one = model.Graph.AddTensor("one", [], [1f]);
        var typedOne = new OnnxEdge("typed_one");
        var erf = model.Graph.AddValue("erf", OnnxTensorType.Create<Half>(shape));
        model.Graph.AddNode(new OnnxNode(
            "one_cast",
            "CastLike",
            string.Empty,
            string.Empty,
            [one, new OnnxEdge(input.Name)],
            [typedOne],
            []));
        model.Graph.AddNode(new Onnxify.Erf("erf", new Onnxify.ErfInputOutputOptions
        {
            Input = new OnnxEdge(input.Name),
            Output = erf,
        }));
        model.Graph.AddNode(new Onnxify.Sub("erfc", new Onnxify.SubInputOutputOptions
        {
            A = typedOne,
            B = new OnnxEdge(erf.Name),
            C = new OnnxEdge(output.Name),
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("Erfc", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["Erf", "CastLike", "Sub"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        var inputValues = new[] { -1f, 0f, 1f };
        using var torchInput = global::TorchSharp.torch.tensor(inputValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var torchOutput = global::TorchSharp.torch.erfc(torchInput);
        var expected = torchOutput.data<Half>().Select(static value => (float)value).ToArray();
        AssertFloat16Close(expected, ExecuteOnnxHalfUnary(model, inputValues));
        AssertFloat16Close(expected, ExecuteOnnxHalfUnary(emitted.Value, inputValues));

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertFloat16Close(expected, ExecuteGeneratedHalfUnaryModule(generated, inputValues, model));
        var reimported = Compiler.CreateTreeFromOnnx(emitted.Value!);
        Assert.True(reimported.IsSuccess, FormatDiagnostics(reimported.Diagnostics));
        Assert.Equal("Erfc", Assert.IsType<CompilerOnnxStep>(Assert.Single(reimported.Value!.Operations)).Descriptor.Name);
    }

    [Fact]
    public void LogitRejectsDynamicEpsilon()
    {
        var source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, double eps) { return torch.logit(input, eps: eps); }";
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.NotNull(diagnostic.Span);
    }

    [Theory]
    [InlineData("torch.div(left, right, roundingMode: \"trunc\")")]
    [InlineData("torch.div(left, right, roundingMode: \"floor\")")]
    public void DivModeWithRuntimeDtypeDependentRoundingIsRejected(string call)
    {
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) {{ return {call}; }}";
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.NotNull(diagnostic.Span);
    }

    [Fact]
    public void FloorDivideImportsFloatDivisionAndFloorAsExplicitOneWayMapping()
    {
        // TorchLib: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py::aten_floor_divide (floating branch).
        // ONNX Runtime: math/element_wise_ops_test.cc (Div and Floor).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new OnnxDimension[] { new OnnxDimension<long>(4) };
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<float>(shape));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<float>(shape));
        var quotient = model.Graph.AddValue("quotient", OnnxTensorType.Create<float>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>(shape));
        model.Graph.AddNode(new Onnxify.Div("floor_divide_div", new Onnxify.DivInputOutputOptions
        {
            A = new OnnxEdge(left.Name), B = new OnnxEdge(right.Name), C = quotient,
        }));
        model.Graph.AddNode(new Onnxify.Floor("floor_divide_floor", new Onnxify.FloorInputOutputOptions
        {
            X = new OnnxEdge(quotient.Name), Y = new OnnxEdge(output.Name),
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("FloorDivide", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.ImportOnly, operation.Descriptor.Capability);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("torch.floor_divide", generated.SourceCode(), StringComparison.Ordinal);

        var leftValues = new[] { -7f, 7f, -7.5f, 7.5f };
        var rightValues = new[] { 3f, -3f, 3f, -3f };
        var expected = new[] { -3f, -3f, -3f, -3f };
        AssertClose(expected, ExecuteOnnx(model, leftValues, [4], rightValues, [4]));
        AssertClose(expected, ExecuteGeneratedComposite(generated, [leftValues, rightValues], model));
        Assert.False(Compiler.GenerateOnnx(imported.Value).IsSuccess);
    }

    [Fact]
    public void FloorDivideImportsSignedIntegerCorrectionPattern()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new OnnxDimension[] { new OnnxDimension<long>(4) };
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<int>(shape));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<int>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<int>(shape));
        OnnxEdge Value(string name)
        {
            model.Graph.AddValue(name, OnnxTensorType.Create<int>(shape));
            return new OnnxEdge(name);
        }

        OnnxEdge BoolValue(string name)
        {
            model.Graph.AddValue(name, OnnxTensorType.Create<bool>(shape));
            return new OnnxEdge(name);
        }
        var signLeft = Value("sign_left");
        var signRight = Value("sign_right");
        var equalSigns = BoolValue("equal_signs");
        var differentSigns = BoolValue("different_signs");
        var remainder = Value("remainder");
        var remainderNonZero = BoolValue("remainder_nonzero");
        var correctionCondition = BoolValue("correction_condition");
        var correction = Value("correction");
        var quotient = Value("quotient");

        model.Graph.AddNode(new Onnxify.Sign("sign_left_node", new Onnxify.SignInputOutputOptions { Input = new OnnxEdge(left.Name), Output = signLeft }));
        model.Graph.AddNode(new Onnxify.Sign("sign_right_node", new Onnxify.SignInputOutputOptions { Input = new OnnxEdge(right.Name), Output = signRight }));
        model.Graph.AddNode(new Onnxify.Equal("equal_signs_node", new Onnxify.EqualInputOutputOptions { A = signLeft, B = signRight, C = equalSigns }));
        model.Graph.AddNode(new Onnxify.Not("different_signs_node", new Onnxify.NotInputOutputOptions { X = equalSigns, Y = differentSigns }));
        model.Graph.AddNode(new Onnxify.Mod("remainder_node", new Onnxify.ModInputOutputOptions { A = new OnnxEdge(left.Name), B = new OnnxEdge(right.Name), C = remainder }));
        model.Graph.AddNode(new Onnxify.Cast("remainder_nonzero_node", new Onnxify.CastInputOutputOptions { Input = remainder, To = 9, Output = remainderNonZero }));
        model.Graph.AddNode(new Onnxify.And("correction_condition_node", new Onnxify.AndInputOutputOptions { A = differentSigns, B = remainderNonZero, C = correctionCondition }));
        model.Graph.AddNode(new Onnxify.Cast("correction_node", new Onnxify.CastInputOutputOptions { Input = correctionCondition, To = 6, Output = correction }));
        model.Graph.AddNode(new Onnxify.Div("quotient_node", new Onnxify.DivInputOutputOptions { A = new OnnxEdge(left.Name), B = new OnnxEdge(right.Name), C = quotient }));
        model.Graph.AddNode(new Onnxify.Sub("floor_divide_node", new Onnxify.SubInputOutputOptions { A = quotient, B = correction, C = new OnnxEdge(output.Name) }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("FloorDivide", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.ImportOnly, operation.Descriptor.Capability);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("torch.floor_divide", generated.SourceCode(), StringComparison.Ordinal);

        var leftValues = new[] { -7, 7, -6, 6 };
        var rightValues = new[] { 3, -3, 3, -3 };
        var expected = new[] { -3, -3, -2, -2 };
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-floor-divide-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var outputs = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<int>(leftValues, [4])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<int>(rightValues, [4])),
            ]);
            Assert.Equal(expected, outputs.Single().AsTensor<int>().ToArray());
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }

        using var torchLeft = global::TorchSharp.torch.tensor(leftValues, [4L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var torchRight = global::TorchSharp.torch.tensor(rightValues, [4L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var torchExpected = global::TorchSharp.torch.floor_divide(torchLeft, torchRight);
        Assert.Equal(torchExpected.data<int>().ToArray(), expected);
    }

    [Fact]
    public void FloorDivideImportsUnsignedDivisionAsFloorDivision()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new OnnxDimension[] { new OnnxDimension<long>(3) };
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<uint>(shape));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<uint>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<uint>(shape));
        model.Graph.AddNode(new Onnxify.Div("unsigned_floor_divide", new Onnxify.DivInputOutputOptions
        {
            A = new OnnxEdge(left.Name), B = new OnnxEdge(right.Name), C = new OnnxEdge(output.Name),
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("FloorDivide", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.ImportOnly, operation.Descriptor.Capability);
    }

    [Theory]
    [InlineData("torch.logaddexp(left, right)", "LogAddExp", new string[] { "Sub", "Abs", "Neg", "Exp", "Add", "Log", "Max", "Add", "IsInf", "IsInf", "Equal", "And", "And", "Where" })]
    [InlineData("torch.logaddexp2(left, right)", "LogAddExp2", new string[] { "Sub", "Abs", "Neg", "Pow", "Add", "Log", "Log", "Div", "Max", "Add", "IsInf", "IsInf", "Equal", "And", "And", "Where" })]
    [InlineData("torch.heaviside(left, right)", "Heaviside", new string[] { "CastLike", "CastLike", "Less", "Where", "Equal", "Where" })]
    [InlineData("torch.xlogy(left, right)", "XLogY", new string[] { "CastLike", "IsNaN", "Equal", "Log", "Mul", "Where", "Where" })]
    public void CompositeBinaryPointwiseMappingsLowerToOnnxAndMatchRuntime(
        string torchExpression,
        string mappingName,
        string[] expectedOperators
    )
    {
        // TorchLib converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops/core.py (aten_logaddexp, aten_logaddexp2) and ops/special.py (aten_special_xlogy).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Exp_float, MathOpTest.Log, MathOpTest.Pow_Float, MathOpTest.Add_float).
        var leftValues = mappingName switch
        {
            "XLogY" => new[] { -2f, 0f, 100f, 1f, -1f },
            _ => new[] { -2f, 0f, 100f, float.PositiveInfinity, float.NegativeInfinity },
        };
        var rightValues = mappingName switch
        {
            "Heaviside" => new[] { 2f, 3f, 100f, float.PositiveInfinity, float.NegativeInfinity },
            "XLogY" => new[] { 2f, float.NaN, 100f, 3f, 4f },
            _ => new[] { 2f, 1f, 100f, float.PositiveInfinity, float.NegativeInfinity },
        };
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) {{ return {torchExpression}; }}";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(tree.Value!.Operations));
        Assert.Equal(mappingName, operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(tree.Value);

        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(expectedOperators, emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        using var left = global::TorchSharp.torch.tensor(leftValues, [leftValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var right = global::TorchSharp.torch.tensor(rightValues, [rightValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var torchOutput = mappingName switch
        {
            "LogAddExp" => global::TorchSharp.torch.logaddexp(left, right),
            "LogAddExp2" => global::TorchSharp.torch.logaddexp2(left, right),
            "Heaviside" => global::TorchSharp.torch.heaviside(left, right),
            "XLogY" => global::TorchSharp.torch.xlogy(left, right),
            _ => throw new ArgumentOutOfRangeException(nameof(mappingName)),
        };
        AssertClose(torchOutput.data<float>().ToArray(), ExecuteOnnx(emitted.Value, leftValues, [5], rightValues, [5]));

        var onnxValues = ExecuteOnnx(emitted.Value, leftValues, [5], rightValues, [5]);
        AssertClose(onnxValues, AssertCompositeRoundTrip(emitted.Value, mappingName, leftValues, rightValues));
    }

    [Fact]
    public void HeavisideFloat16UsesInputTypedConstantsAndMatchesTorchSharpAndOnnxRuntime()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<Half>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<Half>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>([2, 2]));
        model.Graph.AddNode(new OnnxNode(
            "heaviside",
            "Heaviside",
            "com.onnxify.compiler",
            string.Empty,
            [new OnnxEdge(left.Name), new OnnxEdge(right.Name)],
            [new OnnxEdge(output.Name)],
            []));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["CastLike", "CastLike", "Less", "Where", "Equal", "Where"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        var leftValues = new[] { (Half)(-1f), (Half)0f };
        var rightValues = new[] { (Half)0f, (Half)1f };
        using var eagerLeft = global::TorchSharp.torch.tensor(leftValues.Select(static value => (float)value).ToArray(), [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerRight = global::TorchSharp.torch.tensor(rightValues.Select(static value => (float)value).ToArray(), [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerOutput = global::TorchSharp.torch.heaviside(eagerLeft, eagerRight);
        var expected = eagerOutput.data<Half>().ToArray();
        var actual = ExecuteOnnxHalfBroadcast(
            emitted.Value,
            leftValues.Select(static value => (float)value).ToArray(),
            rightValues.Select(static value => (float)value).ToArray());
        AssertClose(expected.Select(static value => (float)value).ToArray(), actual);

        var roundTrip = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(roundTrip.IsSuccess, FormatDiagnostics(roundTrip.Diagnostics));
        Assert.Equal("Heaviside", Assert.IsType<CompilerOnnxStep>(Assert.Single(roundTrip.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(roundTrip.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var generatedOutput = ExecuteGeneratedGenericBinaryModule(
            generated,
            leftValues,
            rightValues,
            static (values, shape) => global::TorchSharp.torch.tensor(values.Select(static value => (float)value).ToArray(), shape, dtype: global::TorchSharp.torch.ScalarType.Float16),
            static tensor => tensor.data<Half>().ToArray(),
            emitted.Value);
        Assert.Equal(expected, generatedOutput);
    }

    [Fact]
    public void XLogYFloat16UsesInputTypedZeroAndMatchesTorchSharpAndOnnxRuntime()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<Half>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<Half>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>([2, 2]));
        model.Graph.AddNode(new OnnxNode(
            "xlogy",
            "XLogY",
            "com.onnxify.compiler",
            string.Empty,
            [new OnnxEdge(left.Name), new OnnxEdge(right.Name)],
            [new OnnxEdge(output.Name)],
            []));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["CastLike", "IsNaN", "Equal", "Log", "Mul", "Where", "Where"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        var leftValues = new[] { (Half)(-2f), (Half)0f };
        var rightValues = new[] { (Half)2f, Half.NaN };
        using var eagerLeft = global::TorchSharp.torch.tensor(leftValues.Select(static value => (float)value).ToArray(), [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerRight = global::TorchSharp.torch.tensor(rightValues.Select(static value => (float)value).ToArray(), [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerOutput = global::TorchSharp.torch.xlogy(eagerLeft, eagerRight);
        var expected = eagerOutput.data<Half>().Select(static value => (float)value).ToArray();
        var actual = ExecuteOnnxHalfBroadcast(
            emitted.Value,
            leftValues.Select(static value => (float)value).ToArray(),
            rightValues.Select(static value => (float)value).ToArray());
        AssertClose(expected, actual);
        Assert.True(float.IsNaN(expected[^1]));

        var roundTrip = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(roundTrip.IsSuccess, FormatDiagnostics(roundTrip.Diagnostics));
        Assert.Equal("XLogY", Assert.IsType<CompilerOnnxStep>(Assert.Single(roundTrip.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(roundTrip.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var generatedOutput = ExecuteGeneratedGenericBinaryModule(
            generated,
            leftValues,
            rightValues,
            static (values, shape) => global::TorchSharp.torch.tensor(values.Select(static value => (float)value).ToArray(), shape, dtype: global::TorchSharp.torch.ScalarType.Float16),
            static tensor => tensor.data<Half>().ToArray(),
            emitted.Value);
        AssertClose(expected, generatedOutput.Select(static value => (float)value).ToArray());
    }

    [Theory]
    [InlineData("torch.addcdiv(input, first, second, value: 1f)", "AddCdiv", new string[] { "Div", "Mul", "Add" })]
    [InlineData("torch.addcmul(input, first, second, value: 1f)", "AddCmul", new string[] { "Mul", "Mul", "Add" })]
    [InlineData("torch.addcdiv(input, first, second, value: 0.5f)", "AddCdiv", new string[] { "Div", "Mul", "Add" })]
    [InlineData("torch.addcmul(input, first, second, value: 0.5f)", "AddCmul", new string[] { "Mul", "Mul", "Add" })]
    public void CompositeTernaryPointwiseMappingsSupportStaticScale(
        string torchExpression,
        string mappingName,
        string[] expectedOperators
    )
    {
        // TorchLib converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops/core.py (aten_addcdiv, aten_addcmul).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Div_float, MathOpTest.Mul_float, MathOpTest.Add_float).
        var inputValues = new[] { 1f, 2f, 3f };
        var firstValues = new[] { 2f, 4f, 6f };
        var secondValues = new[] { 1f, 2f, 3f };
        var source = $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor first, global::TorchSharp.torch.Tensor second) {{ return {torchExpression}; }}";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(tree.Value!.Operations));
        Assert.Equal(mappingName, operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(tree.Value);

        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(expectedOperators, emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        using var input = global::TorchSharp.torch.tensor(inputValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var first = global::TorchSharp.torch.tensor(firstValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var second = global::TorchSharp.torch.tensor(secondValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var eagerOutput = mappingName == "AddCdiv"
            ? torchExpression.Contains("0.5f", StringComparison.Ordinal)
                ? global::TorchSharp.torch.addcdiv(input, first, second, value: 0.5)
                : global::TorchSharp.torch.addcdiv(input, first, second, value: 1.0)
            : torchExpression.Contains("0.5f", StringComparison.Ordinal)
                ? global::TorchSharp.torch.addcmul(input, first, second, value: 0.5)
                : global::TorchSharp.torch.addcmul(input, first, second, value: 1.0);
        AssertClose(eagerOutput.data<float>().ToArray(), ExecuteOnnxThreeInputs(
            emitted.Value,
            ["input", "first", "second"],
            [inputValues, firstValues, secondValues]));
        AssertClose(eagerOutput.data<float>().ToArray(), AssertCompositeRoundTrip(
            emitted.Value,
            mappingName,
            inputValues,
            firstValues,
            secondValues));
    }

    [Theory]
    [InlineData("AddCdiv")]
    [InlineData("AddCmul")]
    public void TernaryArithmeticCompositesSupportFloat16RoundTrip(string mappingName)
    {
        // TorchLib: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_addcdiv, aten_addcmul; Float16 cases).
        // Runtime: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Float16 Div, Mul, and Add kernels).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(2) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<Half>(shape));
        var first = model.Graph.AddInput("first", OnnxTensorType.Create<Half>(shape));
        var second = model.Graph.AddInput("second", OnnxTensorType.Create<Half>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<Half>(shape));
        var value = model.Graph.AddTensor("value", [], [(Half)0.5f]);

        if (mappingName == "AddCdiv")
        {
            var quotient = model.Graph.AddValue("quotient", OnnxTensorType.Create<Half>(shape));
            var scaled = model.Graph.AddValue("scaled", OnnxTensorType.Create<Half>(shape));
            model.Graph.AddNode(new Onnxify.Div("divide", new Onnxify.DivInputOutputOptions
            {
                A = new OnnxEdge(first.Name), B = new OnnxEdge(second.Name), C = quotient,
            }));
            model.Graph.AddNode(new Onnxify.Mul("scale", new Onnxify.MulInputOutputOptions
            {
                A = new OnnxEdge(quotient.Name), B = value, C = scaled,
            }));
            model.Graph.AddNode(new Onnxify.Add("addcdiv", new Onnxify.AddInputOutputOptions
            {
                A = new OnnxEdge(input.Name), B = new OnnxEdge(scaled.Name), C = new OnnxEdge(output.Name),
            }));
        }
        else
        {
            var product = model.Graph.AddValue("product", OnnxTensorType.Create<Half>(shape));
            var scaled = model.Graph.AddValue("scaled", OnnxTensorType.Create<Half>(shape));
            model.Graph.AddNode(new Onnxify.Mul("scale", new Onnxify.MulInputOutputOptions
            {
                A = value, B = new OnnxEdge(first.Name), C = product,
            }));
            model.Graph.AddNode(new Onnxify.Mul("multiply", new Onnxify.MulInputOutputOptions
            {
                A = new OnnxEdge(product.Name), B = new OnnxEdge(second.Name), C = scaled,
            }));
            model.Graph.AddNode(new Onnxify.Add("addcmul", new Onnxify.AddInputOutputOptions
            {
                A = new OnnxEdge(input.Name), B = new OnnxEdge(scaled.Name), C = new OnnxEdge(output.Name),
            }));
        }

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal(mappingName, Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var inputValues = new[] { 1f, 2f };
        var firstValues = new[] { 2f, 4f };
        var secondValues = new[] { 2f, 2f };
        using var eagerInput = global::TorchSharp.torch.tensor(inputValues, [2L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerFirst = global::TorchSharp.torch.tensor(firstValues, [2L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerSecond = global::TorchSharp.torch.tensor(secondValues, [2L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerOutput = mappingName == "AddCdiv"
            ? global::TorchSharp.torch.addcdiv(eagerInput, eagerFirst, eagerSecond, value: 0.5)
            : global::TorchSharp.torch.addcmul(eagerInput, eagerFirst, eagerSecond, value: 0.5);
        var expected = eagerOutput.data<Half>().Select(static item => (float)item).ToArray();
        AssertFloat16Close(expected, ExecuteOnnxHalfThreeInputs(model, inputValues, firstValues, secondValues));
        AssertFloat16Close(expected, ExecuteOnnxHalfThreeInputs(emitted.Value!, inputValues, firstValues, secondValues));

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertFloat16Close(expected, ExecuteGeneratedHalfThreeInputs(generated, inputValues, firstValues, secondValues, model));

        var reimported = Compiler.CreateTreeFromOnnx(emitted.Value!);
        Assert.True(reimported.IsSuccess, FormatDiagnostics(reimported.Diagnostics));
        Assert.Equal(mappingName, Assert.IsType<CompilerOnnxStep>(Assert.Single(reimported.Value!.Operations)).Descriptor.Name);
    }

    [Fact]
    public void AddCmulSupportsInt32Tensors()
    {
        // TorchLib: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_addcmul uses TReal and tests Int32).
        // Runtime: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Mul_int32 and Add_int32).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(2) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<int>(shape));
        var first = model.Graph.AddInput("first", OnnxTensorType.Create<int>(shape));
        var second = model.Graph.AddInput("second", OnnxTensorType.Create<int>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<int>(shape));
        var value = model.Graph.AddTensor("value", [], [1]);
        var product = model.Graph.AddValue("product", OnnxTensorType.Create<int>(shape));
        var scaled = model.Graph.AddValue("scaled", OnnxTensorType.Create<int>(shape));
        model.Graph.AddNode(new Onnxify.Mul("scale", new Onnxify.MulInputOutputOptions
        {
            A = value, B = new OnnxEdge(first.Name), C = product,
        }));
        model.Graph.AddNode(new Onnxify.Mul("multiply", new Onnxify.MulInputOutputOptions
        {
            A = new OnnxEdge(product.Name), B = new OnnxEdge(second.Name), C = scaled,
        }));
        model.Graph.AddNode(new Onnxify.Add("addcmul", new Onnxify.AddInputOutputOptions
        {
            A = new OnnxEdge(input.Name), B = new OnnxEdge(scaled.Name), C = new OnnxEdge(output.Name),
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("AddCmul", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var inputValues = new[] { 1, 2 };
        var firstValues = new[] { 2, 4 };
        var secondValues = new[] { 3, 2 };
        using var eagerInput = global::TorchSharp.torch.tensor(inputValues, [2L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var eagerFirst = global::TorchSharp.torch.tensor(firstValues, [2L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var eagerSecond = global::TorchSharp.torch.tensor(secondValues, [2L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var eagerOutput = global::TorchSharp.torch.addcmul(eagerInput, eagerFirst, eagerSecond, value: 1.0);
        var expected = eagerOutput.data<int>().ToArray();
        Assert.Equal(expected, ExecuteOnnxInt32ThreeInputs(model, inputValues, firstValues, secondValues));
        Assert.Equal(expected, ExecuteOnnxInt32ThreeInputs(emitted.Value!, inputValues, firstValues, secondValues));

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedInt32ThreeInputs(generated, inputValues, firstValues, secondValues, model));

        var reimported = Compiler.CreateTreeFromOnnx(emitted.Value!);
        Assert.True(reimported.IsSuccess, FormatDiagnostics(reimported.Diagnostics));
        Assert.Equal("AddCmul", Assert.IsType<CompilerOnnxStep>(Assert.Single(reimported.Value!.Operations)).Descriptor.Name);
    }

    [Fact]
    public void AddCmulSupportsInt64Tensors()
    {
        // TorchLib: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_addcmul uses TReal; ops_test.py includes Int64).
        // Runtime: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Mul_int64 and Add_int64).
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new[] { new OnnxDimension<long>(2) };
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<long>(shape));
        var first = model.Graph.AddInput("first", OnnxTensorType.Create<long>(shape));
        var second = model.Graph.AddInput("second", OnnxTensorType.Create<long>(shape));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<long>(shape));
        var value = model.Graph.AddTensor("value", [], [1L]);
        var product = model.Graph.AddValue("product", OnnxTensorType.Create<long>(shape));
        var scaled = model.Graph.AddValue("scaled", OnnxTensorType.Create<long>(shape));
        model.Graph.AddNode(new Onnxify.Mul("scale", new Onnxify.MulInputOutputOptions
        {
            A = value, B = new OnnxEdge(first.Name), C = product,
        }));
        model.Graph.AddNode(new Onnxify.Mul("multiply", new Onnxify.MulInputOutputOptions
        {
            A = new OnnxEdge(product.Name), B = new OnnxEdge(second.Name), C = scaled,
        }));
        model.Graph.AddNode(new Onnxify.Add("addcmul", new Onnxify.AddInputOutputOptions
        {
            A = new OnnxEdge(input.Name), B = new OnnxEdge(scaled.Name), C = new OnnxEdge(output.Name),
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("AddCmul", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var inputValues = new[] { 1L, 2L };
        var firstValues = new[] { 2L, 4L };
        var secondValues = new[] { 3L, 2L };
        using var eagerInput = global::TorchSharp.torch.tensor(inputValues, [2L], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var eagerFirst = global::TorchSharp.torch.tensor(firstValues, [2L], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var eagerSecond = global::TorchSharp.torch.tensor(secondValues, [2L], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var eagerOutput = global::TorchSharp.torch.addcmul(eagerInput, eagerFirst, eagerSecond, value: 1.0);
        var expected = eagerOutput.data<long>().ToArray();
        Assert.Equal(expected, ExecuteOnnxInt64ThreeInputs(model, inputValues, firstValues, secondValues));
        Assert.Equal(expected, ExecuteOnnxInt64ThreeInputs(emitted.Value!, inputValues, firstValues, secondValues));

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedInt64ThreeInputs(generated, inputValues, firstValues, secondValues, model));

        var reimported = Compiler.CreateTreeFromOnnx(emitted.Value!);
        Assert.True(reimported.IsSuccess, FormatDiagnostics(reimported.Diagnostics));
        Assert.Equal("AddCmul", Assert.IsType<CompilerOnnxStep>(Assert.Single(reimported.Value!.Operations)).Descriptor.Name);
    }

    [Fact]
    public void LerpCompositeSupportsTensorWeightAndMatchesRuntime()
    {
        // TorchLib converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops/core.py (aten_lerp, Tensor weight form).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Less, MathOpTest.Where, MathOpTest.Add_float, MathOpTest.Sub_float, MathOpTest.Mul_float).
        var startValues = new[] { -2f, 1f, 4f };
        var endValues = new[] { 2f, 5f, -2f };
        var weightValues = new[] { 0.25f, 0.5f, 0.75f };
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor start, global::TorchSharp.torch.Tensor end, global::TorchSharp.torch.Tensor weight) { return torch.lerp(start, end, weight); }";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(tree.Value!.Operations));
        Assert.Equal("Lerp", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var emitted = Compiler.GenerateOnnx(tree.Value);

        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["Sub", "Mul", "Add", "Sub", "Mul", "Sub", "Less", "Where"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        using var start = global::TorchSharp.torch.tensor(startValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var end = global::TorchSharp.torch.tensor(endValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var weight = global::TorchSharp.torch.tensor(weightValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var eagerOutput = global::TorchSharp.torch.lerp(start, end, weight);
        AssertClose(eagerOutput.data<float>().ToArray(), ExecuteOnnxThreeInputs(
            emitted.Value,
            ["start", "end", "weight"],
            [startValues, endValues, weightValues]));
        AssertClose(eagerOutput.data<float>().ToArray(), AssertCompositeRoundTrip(
            emitted.Value,
            "Lerp",
            startValues,
            endValues,
            weightValues));
    }

    [Fact]
    public void LerpCompositeSupportsScalarWeightAndMatchesRuntime()
    {
        // TorchLib converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops/core.py (aten_lerp, Scalar weight form).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Less, MathOpTest.Where, MathOpTest.Add_float, MathOpTest.Sub_float, MathOpTest.Mul_float).
        var startValues = new[] { -2f, 1f, 4f };
        var endValues = new[] { 2f, 5f, -2f };
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor start, global::TorchSharp.torch.Tensor end) { return torch.lerp(start, end, 0.25f); }";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        Assert.Single(tree.Value!.Initializers);

        var emitted = Compiler.GenerateOnnx(tree.Value);

        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        using var start = global::TorchSharp.torch.tensor(startValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var end = global::TorchSharp.torch.tensor(endValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var eagerOutput = global::TorchSharp.torch.lerp(start, end, 0.25);
        AssertClose(eagerOutput.data<float>().ToArray(), ExecuteOnnxThreeInputs(
            emitted.Value!,
            ["start", "end"],
            [startValues, endValues]));
        AssertClose(eagerOutput.data<float>().ToArray(), AssertCompositeRoundTrip(
            emitted.Value!,
            "Lerp",
            startValues,
            endValues));
    }

    [Theory]
    [InlineData("public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor start, global::TorchSharp.torch.Tensor end, float weight) { return torch.lerp(start, end, weight); }")]
    [InlineData("public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor first, global::TorchSharp.torch.Tensor second, float value) { return torch.addcdiv(input, first, second, value: value); }")]
    public void CompositePointwiseMappingsRejectDynamicScalarArguments(string source)
    {
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.NotNull(diagnostic.Span);
    }

    [Fact]
    public void TorchAtan2ReportsUnsupportedInsteadOfUsingInexactTorchLibDecomposition()
    {
        // TorchLib aten_atan2 in third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py maps every NaN result to zero, including NaN inputs and inf/inf.
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor y, global::TorchSharp.torch.Tensor x) { return global::TorchSharp.torch.atan2(y, x); }";

        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.NotNull(diagnostic.Span);
    }

    [Theory]
    [MemberData(nameof(UnaryPointwiseCases))]
    public void UnaryPointwiseMappingsAreBidirectionalAndMatchOnnxRuntime(
        string onnxName,
        string torchMethod,
        Func<float, float> expected
    )
    {
        // Torch converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (the matching aten_{operator} entry).
        // ONNX Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (the corresponding MathOpTest case; Erf is covered by MathOpTest.Erf).
        var inputValues = onnxName switch
        {
            "Abs" or "Neg" => new[] { -2f, -0.5f, 3f },
            "Exp" => new[] { -2f, 0f, 1f },
            "Log" or "Sqrt" => new[] { 0.25f, 1f, 4f },
            "Floor" or "Ceil" => new[] { -1.5f, 0.5f, 2.1f },
            "Tan" or "Asin" or "Acos" => new[] { -0.5f, 0f, 0.5f },
            "Atan" or "Sinh" or "Cosh" or "Asinh" => new[] { -1f, 0f, 1f },
            "Acosh" => new[] { 1f, 1.5f, 3f },
            "Atanh" => new[] { -0.5f, 0f, 0.5f },
            "Reciprocal" => new[] { -2f, 0.5f, 2f },
            "Round" => new[] { -1.5f, 0.5f, 2.1f },
            _ => new[] { -1f, 0f, 1f },
        };
        var expectedValues = inputValues.Select(expected).ToArray();
        var model = CreateUnaryModel(onnxName);
        var onnxTree = Compiler.CreateTreeFromOnnx(model);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));
        var onnxOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(onnxTree.Value!.Operations));
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxOperation.Descriptor.Capability);
        Assert.Equal(onnxName, onnxOperation.Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(onnxTree.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertClose(expectedValues, ExecuteGeneratedUnaryModule(generated, inputValues));

        var expression = onnxName == "Neg" ? "-input" : $"input.{torchMethod}()";
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(
            $"public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) {{ return {expression}; }}"));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var torchOperation = Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations));
        Assert.Equal(onnxOperation.Descriptor, torchOperation.Descriptor);
        var emitted = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(onnxName, Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        AssertClose(expectedValues, ExecuteOnnxUnary(emitted.Value!, inputValues));
    }

    private static float[] AssertCompositeRoundTrip(OnnxModel model, string mappingName, params float[][] inputValues)
    {
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal(mappingName, operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("forward", generated.SourceCode(), StringComparison.Ordinal);
        return ExecuteGeneratedComposite(generated, inputValues, model);
    }

    private static float[] ExecuteGeneratedComposite(
        CompilerResult<IFile[]> generated,
        IReadOnlyList<float[]> inputValues,
        OnnxModel weightsModel
    )
    {
        var source = generated.SourceCode();
        var assemblyName = $"Onnxify.GeneratedComposite{Guid.NewGuid():N}";
        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))!
            .Split(Path.PathSeparator)
            .Select(static path => MetadataReference.CreateFromFile(path))
            .Append(MetadataReference.CreateFromFile(typeof(global::TorchSharp.torch.Tensor).Assembly.Location));
        var compilation = CSharpCompilation.Create(
            assemblyName,
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine,
            emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));

        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.GetMethod("forward") is not null);
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "composite-round-trip", weightsModel);
        var tensors = inputValues.Select(values => (IDisposable)global::TorchSharp.torch.tensor(
            values,
            [values.Length],
            dtype: global::TorchSharp.torch.ScalarType.Float32)).ToArray();
        try
        {
            var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!
                .Invoke(module, tensors.Cast<object>().ToArray())!;
            using (output)
            {
                return output.data<float>().ToArray();
            }
        }
        finally
        {
            foreach (var tensor in tensors)
            {
                tensor.Dispose();
            }
        }
    }

    private static OnnxModel CreateBinaryModel(string opType, long[] leftShape, long[] rightShape, long[]? outputShape = null)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<float>(leftShape.Select(static dimension => new OnnxDimension<long>(dimension))));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<float>(rightShape.Select(static dimension => new OnnxDimension<long>(dimension))));
        var output = model.Graph.AddOutput(
            "output",
            OnnxTensorType.Create<float>((outputShape ?? [2, 3]).Select(static dimension => new OnnxDimension<long>(dimension))));
        model.Graph.AddNode(opType.ToLowerInvariant(), opType, string.Empty, string.Empty, [left, right], [output], []);
        return model;
    }

    private static OnnxModel CreateUnaryModel(string opType)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([new OnnxDimension<long>(3)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([new OnnxDimension<long>(3)]));
        model.Graph.AddNode(opType.ToLowerInvariant(), opType, string.Empty, string.Empty, [input], [output], []);
        return model;
    }

    private static OnnxModel CreateModModel(long fmod)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var dimensions = new[] { new OnnxDimension<long>(2), new OnnxDimension<long>(3) };
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<float>(dimensions));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<float>([new OnnxDimension<long>(3)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>(dimensions));
        model.Graph.AddNode(new Onnxify.Mod(
            "mod",
            new Onnxify.ModInputOutputOptions
            {
                A = left,
                B = right,
                Fmod = fmod,
                C = output,
            }));
        return model;
    }

    private static OnnxModel CreateInt32ModModel()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<int>([2, 3]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<int>([3]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<int>([2, 3]));
        model.Graph.AddNode(new Onnxify.Mod(
            "mod",
            new Onnxify.ModInputOutputOptions
            {
                A = left,
                B = right,
                Fmod = 0,
                C = output,
            }));
        return model;
    }

    private static OnnxModel CreateWhereModel()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var matrixShape = new[] { new OnnxDimension<long>(2), new OnnxDimension<long>(3) };
        var condition = model.Graph.AddInput("condition", OnnxTensorType.Create<bool>(matrixShape));
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<float>(matrixShape));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<float>([new OnnxDimension<long>(3)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>(matrixShape));
        model.Graph.AddNode("where", "Where", string.Empty, string.Empty, [condition, left, right], [output], []);
        return model;
    }

    private static float[] ExecuteGeneratedModule(
        CompilerResult<IFile[]> generated,
        float[] leftValues,
        float[]? rightValues,
        OnnxModel? weightsModel = null
    )
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Arithmetic{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static d => d.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "arithmetic", weightsModel);
        var leftShape = leftValues.Length == 6 ? new long[] { 2, 3 } : [leftValues.Length];
        using var left = global::TorchSharp.torch.tensor(leftValues, leftShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var right = rightValues is null
            ? null
            : global::TorchSharp.torch.tensor(rightValues, rightValues.Length == 1 ? [] : [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(
            module,
            right is null ? [left] : [left, right])!;
        return output.data<float>().ToArray();
    }

    private static float[] ExecuteGeneratedBroadcastModule(
        CompilerResult<IFile[]> generated,
        float[] leftValues,
        long[] leftShape,
        float[] rightValues,
        long[] rightShape)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Broadcast{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "broadcast");
        using var left = global::TorchSharp.torch.tensor(leftValues, leftShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var right = global::TorchSharp.torch.tensor(rightValues, rightShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return output.data<float>().ToArray();
    }

    private static double[] ExecuteGeneratedDoubleBroadcastModule(CompilerResult<IFile[]> generated, double[] leftValues, double[] rightValues)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.DoubleBroadcast{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "double_broadcast");
        using var left = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var right = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return output.data<double>().ToArray();
    }

    private static double[] ExecuteOnnxDouble(OnnxModel model, double[] leftValues, double[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-double-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<double>(leftValues, [2, 1])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<double>(rightValues, [1, 2])),
            ]);
            return results.Single().AsTensor<double>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static double[] ExecuteOnnxDoubleBroadcast(OnnxModel model, double[] leftValues, double[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-double-broadcast-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<double>(leftValues, [2, 1])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<double>(rightValues, [1, 2])),
            ]);
            return results.Single().AsTensor<double>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static float[] ExecuteOnnxHalfBroadcast(OnnxModel model, float[] leftValues, float[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-float16-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var left = leftValues.Select(static value => (global::Microsoft.ML.OnnxRuntime.Float16)value).ToArray();
            var right = rightValues.Select(static value => (global::Microsoft.ML.OnnxRuntime.Float16)value).ToArray();
            var output = new global::Microsoft.ML.OnnxRuntime.Float16[4];
            using var leftBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance,
                left,
                TensorElementType.Float16,
                [2, 1],
                left.Length * sizeof(ushort));
            using var rightBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance,
                right,
                TensorElementType.Float16,
                [1, 2],
                right.Length * sizeof(ushort));
            using var outputBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance,
                output,
                TensorElementType.Float16,
                [2, 2],
                output.Length * sizeof(ushort));
            session.Run(["left", "right"], [leftBuffer, rightBuffer], ["output"], [outputBuffer]);
            return output.Select(static value => (float)value).ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static float[] ExecuteOnnxHalfUnary(OnnxModel model, float[] inputValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-float16-unary-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var input = inputValues.Select(static value => (global::Microsoft.ML.OnnxRuntime.Float16)value).ToArray();
            var output = new global::Microsoft.ML.OnnxRuntime.Float16[inputValues.Length];
            using var inputBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance, input, TensorElementType.Float16, [input.Length], input.Length * sizeof(ushort));
            using var outputBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance, output, TensorElementType.Float16, [output.Length], output.Length * sizeof(ushort));
            session.Run(["input"], [inputBuffer], ["output"], [outputBuffer]);
            return output.Select(static value => (float)value).ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static float[] ExecuteOnnxHalfThreeInputs(
        OnnxModel model,
        float[] inputValues,
        float[] firstValues,
        float[] secondValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-float16-ternary-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var input = inputValues.Select(static value => (global::Microsoft.ML.OnnxRuntime.Float16)value).ToArray();
            var first = firstValues.Select(static value => (global::Microsoft.ML.OnnxRuntime.Float16)value).ToArray();
            var second = secondValues.Select(static value => (global::Microsoft.ML.OnnxRuntime.Float16)value).ToArray();
            var output = new global::Microsoft.ML.OnnxRuntime.Float16[inputValues.Length];
            using var inputBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance, input, TensorElementType.Float16, [input.Length], input.Length * sizeof(ushort));
            using var firstBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance, first, TensorElementType.Float16, [first.Length], first.Length * sizeof(ushort));
            using var secondBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance, second, TensorElementType.Float16, [second.Length], second.Length * sizeof(ushort));
            using var outputBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance, output, TensorElementType.Float16, [output.Length], output.Length * sizeof(ushort));
            session.Run(
                ["input", "first", "second"],
                [inputBuffer, firstBuffer, secondBuffer],
                ["output"],
                [outputBuffer]);
            return output.Select(static value => (float)value).ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static int[] ExecuteOnnxInt32ThreeInputs(
        OnnxModel model,
        int[] inputValues,
        int[] firstValues,
        int[] secondValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-int32-ternary-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<int>(inputValues, [inputValues.Length])),
                NamedOnnxValue.CreateFromTensor("first", new DenseTensor<int>(firstValues, [firstValues.Length])),
                NamedOnnxValue.CreateFromTensor("second", new DenseTensor<int>(secondValues, [secondValues.Length])),
            ]);
            return results.Single().AsTensor<int>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static long[] ExecuteOnnxInt64ThreeInputs(
        OnnxModel model,
        long[] inputValues,
        long[] firstValues,
        long[] secondValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-int64-ternary-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<long>(inputValues, [inputValues.Length])),
                NamedOnnxValue.CreateFromTensor("first", new DenseTensor<long>(firstValues, [firstValues.Length])),
                NamedOnnxValue.CreateFromTensor("second", new DenseTensor<long>(secondValues, [secondValues.Length])),
            ]);
            return results.Single().AsTensor<long>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static int[] ExecuteOnnxInt32Broadcast(OnnxModel model, int[] leftValues, int[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-int32-fmod-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<int>(leftValues, [2, 1])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<int>(rightValues, [1, 2])),
            ]);
            return results.Single().AsTensor<int>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static int[] ExecuteGeneratedInt32BroadcastModule(CompilerResult<IFile[]> generated, int[] leftValues, int[] rightValues)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Int32Fmod{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "int32_fmod");
        using var left = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var right = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return output.data<int>().ToArray();
    }

    private static byte[] ExecuteOnnxUInt8Broadcast(OnnxModel model, byte[] leftValues, byte[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-uint8-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<byte>(leftValues, [2, 1])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<byte>(rightValues, [1, 2])),
            ]);
            return results.Single().AsTensor<byte>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static byte[] ExecuteGeneratedUInt8BroadcastModule(CompilerResult<IFile[]> generated, byte[] leftValues, byte[] rightValues)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.UInt8Arithmetic{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "uint8_arithmetic");
        using var left = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Byte);
        using var right = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Byte);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return output.data<byte>().ToArray();
    }

    private static float[] ExecuteOnnxFloatInt32Broadcast(OnnxModel model, float[] leftValues, int[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-pow-mixed-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<float>(leftValues, [2, 1])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<int>(rightValues, [1, 2])),
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

    private static float[] ExecuteGeneratedFloatInt32BroadcastModule(CompilerResult<IFile[]> generated, float[] leftValues, int[] rightValues)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.MixedPow{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "mixed_pow");
        using var left = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var right = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return output.data<float>().ToArray();
    }

    private static float[] ExecuteGeneratedHalfBroadcastModule(CompilerResult<IFile[]> generated, float[] leftValues, float[] rightValues)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Float16Binary{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "float16_binary");
        using var left = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var right = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return output.data<Half>().Select(static value => (float)value).ToArray();
    }

    private static long[] ExecuteOnnxInt64Broadcast(OnnxModel model, long[] leftValues, long[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-int64-fmod-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<long>(leftValues, [2, 1])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<long>(rightValues, [1, 2])),
            ]);
            return results.Single().AsTensor<long>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static long[] ExecuteGeneratedInt64BroadcastModule(CompilerResult<IFile[]> generated, long[] leftValues, long[] rightValues)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Int64Fmod{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "int64_fmod");
        using var left = global::TorchSharp.torch.tensor(leftValues, [2L, 1L], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var right = global::TorchSharp.torch.tensor(rightValues, [1L, 2L], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return output.data<long>().ToArray();
    }

    private static void AssertFmodIntegerType<T>(
        T[] leftValues,
        T[] rightValues,
        Func<T[], long[], global::TorchSharp.torch.Tensor> createTorchTensor,
        Func<global::TorchSharp.torch.Tensor, T[]> readTorchTensor)
        where T : unmanaged
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<T>([2, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<T>([1, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<T>([2, 2]));
        model.Graph.AddNode(new Onnxify.Mod("fmod", new Onnxify.ModInputOutputOptions
        {
            A = left, B = right, Fmod = 1, C = output,
        }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        Assert.Equal("Fmod", Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        using var leftTensor = createTorchTensor(leftValues, [2L, 1L]);
        using var rightTensor = createTorchTensor(rightValues, [1L, 2L]);
        using var expectedTensor = global::TorchSharp.torch.fmod(leftTensor, rightTensor);
        var expected = readTorchTensor(expectedTensor);
        Assert.Equal(expected, ExecuteGeneratedGenericBinaryModule(generated, leftValues, rightValues, createTorchTensor, readTorchTensor));
        Assert.Equal(expected, ExecuteOnnxTypedFmodBroadcast<T>(emitted.Value!, leftValues, rightValues));
    }

    private static T[] ExecuteOnnxTypedFmodBroadcast<T>(OnnxModel model, T[] leftValues, T[] rightValues)
        where T : unmanaged
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-fmod-{typeof(T).Name}-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<T>(leftValues, [2, 1])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<T>(rightValues, [1, 2])),
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

    private static T[] ExecuteGeneratedGenericBinaryModule<T>(
        CompilerResult<IFile[]> generated,
        T[] leftValues,
        T[] rightValues,
        Func<T[], long[], global::TorchSharp.torch.Tensor> createTorchTensor,
        Func<global::TorchSharp.torch.Tensor, T[]> readTorchTensor,
        OnnxModel? weightsModel = null)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Fmod{typeof(T).Name}{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, $"fmod_{typeof(T).Name}", weightsModel);
        using var left = createTorchTensor(leftValues, [2L, 1L]);
        using var right = createTorchTensor(rightValues, [1L, 2L]);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return readTorchTensor(output);
    }

    private static bool[] ExecuteOnnxDoubleBoolean(OnnxModel model, double[] leftValues, double[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-double-bool-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<double>(leftValues, [leftValues.Length])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<double>(rightValues, [rightValues.Length])),
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

    private static double[] ExecuteOnnxDoubleUnary(OnnxModel model, double[] inputValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-double-unary-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<double>(inputValues, [inputValues.Length]))]);
            return results.Single().AsTensor<double>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static void AssertCloseDouble(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
        {
            if (double.IsNaN(expected[index]))
            {
                Assert.True(double.IsNaN(actual[index]));
            }
            else
            {
                Assert.Equal(expected[index], actual[index], 12);
            }
        }
    }

    private static float[] ExecuteOnnx(OnnxModel model, float[] leftValues, long[] leftShape, float[] rightValues, long[] rightShape)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-{Guid.NewGuid():N}.onnx");
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

    private static float[] ExecuteOnnxScalar(OnnxModel model, float[] inputValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-arithmetic-scalar-{Guid.NewGuid():N}.onnx");
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

    private static float[] ExecuteOnnxThreeInputs(OnnxModel model, string[] inputNames, float[][] inputValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-pointwise-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var inputs = inputNames
                .Select((name, index) => NamedOnnxValue.CreateFromTensor(
                    name,
                    new DenseTensor<float>(inputValues[index], [inputValues[index].Length])))
                .ToArray();
            using var results = session.Run(inputs);
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

    private static float[] ExecuteGeneratedUnaryModule(CompilerResult<IFile[]> generated, float[] inputValues, OnnxModel? weightsModel = null)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.UnaryArithmetic{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static d => d.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "unary", weightsModel);
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<float>().ToArray();
    }

    private static float[] ExecuteGeneratedHalfUnaryModule(CompilerResult<IFile[]> generated, float[] inputValues, OnnxModel weightsModel)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Float16UnaryArithmetic{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "float16_round", weightsModel);
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<Half>().Select(static value => (float)value).ToArray();
    }

    private static float[] ExecuteGeneratedHalfThreeInputs(
        CompilerResult<IFile[]> generated,
        float[] inputValues,
        float[] firstValues,
        float[] secondValues,
        OnnxModel weightsModel)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Float16TernaryArithmetic{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "float16_ternary", weightsModel);
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var first = global::TorchSharp.torch.tensor(firstValues, [firstValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var second = global::TorchSharp.torch.tensor(secondValues, [secondValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input, first, second])!;
        return output.data<Half>().Select(static value => (float)value).ToArray();
    }

    private static int[] ExecuteGeneratedInt32ThreeInputs(
        CompilerResult<IFile[]> generated,
        int[] inputValues,
        int[] firstValues,
        int[] secondValues,
        OnnxModel weightsModel)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Int32TernaryArithmetic{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "int32_ternary", weightsModel);
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var first = global::TorchSharp.torch.tensor(firstValues, [firstValues.Length], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var second = global::TorchSharp.torch.tensor(secondValues, [secondValues.Length], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input, first, second])!;
        return output.data<int>().ToArray();
    }

    private static long[] ExecuteGeneratedInt64ThreeInputs(
        CompilerResult<IFile[]> generated,
        long[] inputValues,
        long[] firstValues,
        long[] secondValues,
        OnnxModel weightsModel)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Int64TernaryArithmetic{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "int64_ternary", weightsModel);
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var first = global::TorchSharp.torch.tensor(firstValues, [firstValues.Length], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var second = global::TorchSharp.torch.tensor(secondValues, [secondValues.Length], dtype: global::TorchSharp.torch.ScalarType.Int64);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input, first, second])!;
        return output.data<long>().ToArray();
    }

    private static double[] ExecuteGeneratedDoubleUnaryModule(CompilerResult<IFile[]> generated, double[] inputValues, OnnxModel weightsModel)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.DoubleUnaryArithmetic{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "double_unary", weightsModel);
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float64);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<double>().ToArray();
    }

    private static float[] ExecuteOnnxUnary(OnnxModel model, float[] inputValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-unary-arithmetic-{Guid.NewGuid():N}.onnx");
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

    private static int[] ExecuteOnnxInt32(OnnxModel model)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-mod-int32-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<int>(new[] { -5, 5, -5, 5, -2, 2 }, new[] { 2, 3 })),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<int>(new[] { 2, -2, 3 }, new[] { 3 })),
            ]);
            return results.Single().AsTensor<int>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static float[] ExecuteWhereModule(CompilerResult<IFile[]> generated, bool[] conditions, float[] leftValues, float[] rightValues)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Where{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "where");
        using var condition = global::TorchSharp.torch.tensor(conditions, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Bool);
        using var left = global::TorchSharp.torch.tensor(leftValues, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var right = global::TorchSharp.torch.tensor(rightValues, [3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [condition, left, right])!;
        return output.data<float>().ToArray();
    }

    private static float[] ExecuteWhereOnnx(OnnxModel model, bool[] conditions, float[] leftValues, float[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-where-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [
                NamedOnnxValue.CreateFromTensor("condition", new DenseTensor<bool>(conditions, [2, 3])),
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<float>(leftValues, [2, 3])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<float>(rightValues, [3])),
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

    private static long[] ExecuteGeneratedCastModule(CompilerResult<IFile[]> generated, float[] values)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Cast{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static item => item.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "cast");
        using var input = global::TorchSharp.torch.tensor(values, [values.Length], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<long>().ToArray();
    }

    private static long[] ExecuteOnnxCast(OnnxModel model, float[] values)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-cast-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var results = session.Run(
            [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(values, [values.Length]))]);
            return results.Single().AsTensor<long>().ToArray();
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

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics) =>
        string.Join(Environment.NewLine, diagnostics.Select(static diagnostic => $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
        {
            if (float.IsInfinity(expected[index]) || float.IsNaN(expected[index]))
            {
                Assert.Equal(expected[index], actual[index]);
                continue;
            }

            Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0f, 1e-5f * MathF.Max(1f, MathF.Abs(expected[index])));
        }
    }

    private static void AssertFloat16Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
        {
            if (float.IsInfinity(expected[index]) || float.IsNaN(expected[index]))
            {
                Assert.Equal(expected[index], actual[index]);
                continue;
            }

            // Allow one Float16 ULP at Logit outputs where ORT and TorchSharp round intermediates differently.
            Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0f, 0.0021f);
        }
    }

    private static float ErrorFunction(float value)
    {
        var term = value;
        var sum = value;
        for (var index = 1; index <= 24; index++)
        {
            term *= -(value * value) / index;
            sum += term / ((2 * index) + 1);
        }

        var result = (2f / MathF.Sqrt(MathF.PI)) * sum;
        return result;
    }
}
