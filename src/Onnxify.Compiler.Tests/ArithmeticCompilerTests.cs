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
        AssertClose(expectedValues, ExecuteGeneratedModule(generatedCSharp.SourceCode(), leftValues, rightValues));

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
        Assert.Equal([3f, 4f, 5f], ExecuteGeneratedModule(generated.SourceCode(), [1f, 2f, 3f], null));
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
        AssertClose(expected, ExecuteGeneratedModule(generated.SourceCode(), [1f, 2f, 3f], null));
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
        Assert.Equal(expected, ExecuteGeneratedBroadcastModule(generated.SourceCode(), leftValues, leftShape, rightValues, rightShape));
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
        Assert.Equal(expected, ExecuteGeneratedDoubleBroadcastModule(generated.SourceCode(), leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxDouble(model, leftValues, rightValues));

        var emitted = Compiler.GenerateOnnx(imported.Value!);

        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Add>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnxDouble(emitted.Value, leftValues, rightValues));
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
        AssertClose(expected, ExecuteGeneratedModule(generated.SourceCode(), leftValues, rightValues));
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
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("left.remainder(right)", generated.SourceCode());

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
        var source = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(source.IsSuccess, FormatDiagnostics(source.Diagnostics));
        Assert.Contains("torch.where(condition, left, right)", source.SourceCode());

        var conditions = new[] { true, false, true, false, true, false };
        var leftValues = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var rightValues = new[] { 10f, 20f, 30f };
        var expectedValues = new[] { 1f, 20f, 3f, 10f, 5f, 30f };
        AssertClose(expectedValues, ExecuteWhereModule(source.SourceCode(), conditions, leftValues, rightValues));

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
        Assert.Equal([1L, -2L, 3L], ExecuteGeneratedCastModule(generated.SourceCode(), [1.9f, -2.2f, 3.8f]));

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
    [InlineData("torch.deg2rad(input)", "Deg2Rad", new string[] { "Mul" }, new float[] { -180f, 0f, 90f }, new float[] { -MathF.PI, 0f, MathF.PI / 2f })]
    [InlineData("torch.rad2deg(input)", "Rad2Deg", new string[] { "Mul" }, new float[] { -MathF.PI, 0f, MathF.PI / 2f }, new float[] { -180f, 0f, 90f })]
    [InlineData("torch.rsqrt(input)", "Rsqrt", new string[] { "Sqrt", "Reciprocal" }, new float[] { 0.25f, 1f, 4f }, new float[] { 2f, 1f, 0.5f })]
    public void CompositeUnaryPointwiseMappingsLowerToOnnxAndMatchRuntime(
        string torchExpression,
        string mappingName,
        string[] expectedOperators,
        float[] inputValues,
        float[] expectedValues
    )
    {
        // TorchLib converter cases: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py (aten_exp2, aten_log1p, aten_log10, aten_log2, aten_deg2rad, aten_rad2deg, aten_rsqrt).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Pow_Float, MathOpTest.Log, MathOpTest.Exp_float).
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
            "Deg2Rad" => global::TorchSharp.torch.deg2rad(torchInput),
            "Rad2Deg" => global::TorchSharp.torch.rad2deg(torchInput),
            "Rsqrt" => global::TorchSharp.torch.rsqrt(torchInput),
            _ => throw new ArgumentOutOfRangeException(nameof(mappingName)),
        };
        AssertClose(torchOutput.data<float>().ToArray(), actualValues);

        AssertClose(actualValues, AssertCompositeRoundTrip(emitted.Value, mappingName, inputValues));
    }

    [Theory]
    [InlineData("torch.logaddexp(left, right)", "LogAddExp", new string[] { "Sub", "Abs", "Neg", "Exp", "Add", "Log", "Max", "Add", "IsInf", "IsInf", "Equal", "And", "And", "Where" })]
    [InlineData("torch.logaddexp2(left, right)", "LogAddExp2", new string[] { "Sub", "Abs", "Neg", "Pow", "Add", "Log", "Log", "Div", "Max", "Add", "IsInf", "IsInf", "Equal", "And", "And", "Where" })]
    public void CompositeBinaryPointwiseMappingsLowerToOnnxAndMatchRuntime(
        string torchExpression,
        string mappingName,
        string[] expectedOperators
    )
    {
        // TorchLib converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops/core.py (aten_logaddexp, aten_logaddexp2).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (MathOpTest.Exp_float, MathOpTest.Log, MathOpTest.Pow_Float, MathOpTest.Add_float).
        var leftValues = new[] { -2f, 0f, 100f, float.PositiveInfinity, float.NegativeInfinity };
        var rightValues = new[] { 2f, 1f, 100f, float.PositiveInfinity, float.NegativeInfinity };
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
        using var torchOutput = mappingName == "LogAddExp"
            ? global::TorchSharp.torch.logaddexp(left, right)
            : global::TorchSharp.torch.logaddexp2(left, right);
        AssertClose(torchOutput.data<float>().ToArray(), ExecuteOnnx(emitted.Value, leftValues, [5], rightValues, [5]));

        var onnxValues = ExecuteOnnx(emitted.Value, leftValues, [5], rightValues, [5]);
        AssertClose(onnxValues, AssertCompositeRoundTrip(emitted.Value, mappingName, leftValues, rightValues));
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
        AssertClose(expectedValues, ExecuteGeneratedUnaryModule(generated.SourceCode(), inputValues));

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
        return ExecuteGeneratedComposite(generated.SourceCode(), inputValues);
    }

    private static float[] ExecuteGeneratedComposite(string source, IReadOnlyList<float[]> inputValues)
    {
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
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "composite-round-trip")!;
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

    private static float[] ExecuteGeneratedBroadcastModule(
        string source,
        float[] leftValues,
        long[] leftShape,
        float[] rightValues,
        long[] rightShape)
    {
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
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "broadcast")!;
        using var left = global::TorchSharp.torch.tensor(leftValues, leftShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var right = global::TorchSharp.torch.tensor(rightValues, rightShape, dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return output.data<float>().ToArray();
    }

    private static double[] ExecuteGeneratedDoubleBroadcastModule(string source, double[] leftValues, double[] rightValues)
    {
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
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "double_broadcast")!;
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
            if (float.IsInfinity(expected[index]) || float.IsNaN(expected[index]))
            {
                Assert.Equal(expected[index], actual[index]);
                continue;
            }

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
