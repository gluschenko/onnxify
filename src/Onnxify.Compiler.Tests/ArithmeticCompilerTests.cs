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
        var generatedCSharp = Compiler.GenerateCSharp(onnxTree.Value);
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
        Assert.Contains(expectedGeneratedForm, generatedCSharp.Value);
        AssertClose(expectedValues, ExecuteGeneratedModule(generatedCSharp.Value!, leftValues, rightValues));

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

        var generated = Compiler.GenerateCSharp(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal([3f, 4f, 5f], ExecuteGeneratedModule(generated.Value!, [1f, 2f, 3f], null));
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

    [Fact]
    public void DynamicScalarArithmeticIsRejectedWithACompilerDiagnostic()
    {
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, float scale) { return input * scale; }";
        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error
            && diagnostic.Code == CompilerDiagnosticCodes.Unsupported);
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
    public void TruncMappingLowersToRuntimeSupportedOnnxAndDeclaresItsDirection()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (trunc), exercised by ops_test.py::test_output_match_opinfo_.
        // ONNX Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (ModOpTest.Fmod_float_mixed_sign).
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.trunc(); }";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(tree.Value!.Operations));
        Assert.Equal("Trunc", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.ExportOnly, operation.Descriptor.Capability);

        var unsupportedImport = Compiler.GenerateCSharp(tree.Value);
        Assert.False(unsupportedImport.IsSuccess);
        Assert.Contains(unsupportedImport.Diagnostics, static diagnostic => diagnostic.Code == CompilerDiagnosticCodes.Unsupported);

        var emitted = Compiler.GenerateOnnx(tree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["Mod", "Sub"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        AssertClose([-1f, 0f, 2f], ExecuteOnnxUnary(emitted.Value, [-1.9f, 0.5f, 2.1f]));
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

        var generated = Compiler.GenerateCSharp(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertClose(expected, ExecuteGeneratedModule(generated.Value!, leftValues, rightValues));
    }

    [Fact]
    public void RemainderImportPreservesTorchSemanticsAndRejectsFmod()
    {
        var model = CreateModModel(fmod: 0);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var operation = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("Mod", operation.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);
        var generated = Compiler.GenerateCSharp(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("left.remainder(right)", generated.Value);

        var fmodModel = CreateModModel(fmod: 1);
        var unsupported = Compiler.CreateTreeFromOnnx(fmodModel);
        Assert.False(unsupported.IsSuccess);
        Assert.Contains(unsupported.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error
            && diagnostic.Message.Contains("fmod=1", StringComparison.Ordinal));
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
        var source = Compiler.GenerateCSharp(imported.Value);
        Assert.True(source.IsSuccess, FormatDiagnostics(source.Diagnostics));
        Assert.Contains("torch.where(condition, left, right)", source.Value);

        var conditions = new[] { true, false, true, false, true, false };
        var leftValues = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var rightValues = new[] { 10f, 20f, 30f };
        var expectedValues = new[] { 1f, 20f, 3f, 10f, 5f, 30f };
        AssertClose(expectedValues, ExecuteWhereModule(source.Value!, conditions, leftValues, rightValues));

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
        var generated = Compiler.GenerateCSharp(onnxTree.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("to_type(torch.ScalarType.Int64)", generated.Value);
        Assert.Equal([1L, -2L, 3L], ExecuteGeneratedCastModule(generated.Value!, [1.9f, -2.2f, 3.8f]));

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
        var generated = Compiler.GenerateCSharp(onnxTree.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        AssertClose(expectedValues, ExecuteGeneratedUnaryModule(generated.Value!, inputValues));

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

    private static OnnxModel CreateBinaryModel(string opType, long[] leftShape, long[] rightShape)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<float>(leftShape.Select(static dimension => new OnnxDimension<long>(dimension))));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<float>(rightShape.Select(static dimension => new OnnxDimension<long>(dimension))));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2, 3]));
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

    private static float[] ExecuteGeneratedModule(string source, float[] leftValues, float[]? rightValues)
    {
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
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "arithmetic")!;
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

    private static float[] ExecuteGeneratedUnaryModule(string source, float[] inputValues)
    {
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
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "unary")!;
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<float>().ToArray();
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

    private static float[] ExecuteWhereModule(string source, bool[] conditions, float[] leftValues, float[] rightValues)
    {
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
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "where")!;
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

    private static long[] ExecuteGeneratedCastModule(string source, float[] values)
    {
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
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "cast")!;
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
            Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0f, 1e-5f * MathF.Max(1f, MathF.Abs(expected[index])));
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
