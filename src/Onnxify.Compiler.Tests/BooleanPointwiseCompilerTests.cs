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
        ["NotEqual", "ne", (Func<float, float, bool>)((left, right) => left != right)],
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
        // Torch converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (eq, ne, ge, gt, le, lt), exercised by ops_test.py::test_output_match_opinfo_. aten_ne is exactly Not(Equal).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc (Equal_float, Equal_multidirectional_broadcastAB, Not, Greater_9_float, GreaterOrEqual_12_float, Less, LessOrEqual).
        var leftValues = new[] { 1f, 4f, 3f, 5f, 2f, 6f };
        var rightValues = new[] { 1f, 3f, 7f };
        var expected = Broadcast(leftValues, rightValues, comparison);
        var model = CreateModel(onnxName, isLogical: false);

        var onnxTree = Compiler.CreateTreeFromOnnx(model);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(onnxTree.Value!.Operations));
        Assert.Equal(onnxName, onnxStep.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generatedSource = Compiler.GenerateSourceCode(onnxTree.Value);
        Assert.True(generatedSource.IsSuccess, FormatDiagnostics(generatedSource.Diagnostics));
        Assert.True(expected.SequenceEqual(ExecuteGeneratedModule(generatedSource, leftValues, rightValues)));

        var torchSource = CreateTorchSource(torchName);
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(torchSource));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations));
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);

        var emittedModel = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emittedModel.IsSuccess, FormatDiagnostics(emittedModel.Diagnostics));
        if (onnxName == "NotEqual")
        {
            Assert.Equal(["Equal", "Not"], emittedModel.Value!.Graph.Nodes.Select(static node => node.OpType));
        }
        else
        {
            Assert.Equal(onnxName, Assert.Single(emittedModel.Value!.Graph.Nodes).OpType);
        }
        Assert.True(expected.SequenceEqual(ExecuteOnnx(emittedModel.Value, leftValues, rightValues)));
    }

    [Theory]
    [InlineData("Greater")]
    [InlineData("GreaterOrEqual")]
    [InlineData("Less")]
    [InlineData("LessOrEqual")]
    public void OrderedComparisonsRejectFloat16CpuInputs(string opType)
    {
        var model = CreateTypedComparisonModel(opType, CompilerElementType.Float16, CompilerElementType.Float16);

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("does not register this element type", StringComparison.Ordinal));
    }

    [Fact]
    public void EqualRejectsMismatchedKnownInputTypes()
    {
        var model = CreateTypedComparisonModel("Equal", CompilerElementType.Float32, CompilerElementType.Int32);

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("requires matching input element types", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("Equal")]
    [InlineData("Greater")]
    [InlineData("GreaterOrEqual")]
    [InlineData("Less")]
    [InlineData("LessOrEqual")]
    public void IntegerComparisonMappingsPreserveInt32Values(string opType)
    {
        var leftValues = new[] { -4, 7, 3, 9 };
        var rightValues = new[] { -4, 6, 5, 10 };
        var model = CreateTypedComparisonModel(opType, CompilerElementType.Int32, CompilerElementType.Int32);

        var onnxTree = Compiler.CreateTreeFromOnnx(model);
        Assert.True(onnxTree.IsSuccess, FormatDiagnostics(onnxTree.Diagnostics));
        var generated = Compiler.GenerateSourceCode(onnxTree.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(onnxTree.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));

        var expected = (opType switch
        {
            "Equal" => leftValues.Zip(rightValues, static (left, right) => left == right),
            "Greater" => leftValues.Zip(rightValues, static (left, right) => left > right),
            "GreaterOrEqual" => leftValues.Zip(rightValues, static (left, right) => left >= right),
            "Less" => leftValues.Zip(rightValues, static (left, right) => left < right),
            "LessOrEqual" => leftValues.Zip(rightValues, static (left, right) => left <= right),
            _ => throw new ArgumentOutOfRangeException(nameof(opType)),
        }).ToArray();
        Assert.Equal(expected, ExecuteGeneratedInt32Comparison(generated, leftValues, rightValues));
        Assert.Equal(expected, ExecuteOnnxInt32Comparison(emitted.Value!, leftValues, rightValues));
    }

    [Theory]
    [InlineData("And")]
    [InlineData("Or")]
    [InlineData("Xor")]
    public void LogicalBinaryOperatorsRejectKnownNonBooleanInputs(string opType)
    {
        var model = CreateTypedComparisonModel(opType, CompilerElementType.Int32, CompilerElementType.Int32);

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("accepts Boolean inputs only", StringComparison.Ordinal));
    }

    [Fact]
    public void LogicalNotRejectsKnownNonBooleanInput()
    {
        var model = CreateUnaryModel(
            OnnxTensorType.Create<int>([new OnnxDimension<long>(4)]),
            new Onnxify.Not("logical_not", new Onnxify.NotInputOutputOptions
            {
                X = new OnnxEdge("input"),
                Y = new OnnxEdge("output"),
            }),
            OnnxTensorType.Create<bool>([new OnnxDimension<long>(4)]));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("accepts Boolean inputs only", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("Abs", "UInt16")]
    [InlineData("Abs", "Float16")]
    [InlineData("Neg", "Float16")]
    [InlineData("Neg", "UInt8")]
    [InlineData("Exp", "Int32")]
    [InlineData("Sqrt", "Float16")]
    [InlineData("Round", "Int32")]
    [InlineData("Sign", "Boolean")]
    [InlineData("Relu", "UInt8")]
    [InlineData("Sigmoid", "Int32")]
    [InlineData("Tanh", "Int32")]
    public void UnaryOperatorsRejectKnownTypesOutsideTorchAndCpuIntersection(string opType, string elementTypeName)
    {
        var model = CreateTypedUnaryModel(opType, elementTypeName);

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("intersection does not include this type", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("Erf")]
    [InlineData("Round")]
    public void UnaryFloat16KernelsWithExplicitCpuRegistrationsRemainImportable(string opType)
    {
        var model = CreateTypedUnaryModel(opType, "Float16");

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.True(result.IsSuccess, FormatDiagnostics(result.Diagnostics));
    }

    [Fact]
    public void ReluInt32ImportExportAndGeneratedTorchSharpPreserveValues()
    {
        var inputValues = new[] { -2, 0, 3, -4 };
        var expected = new[] { 0, 0, 3, 0 };
        var imported = Compiler.CreateTreeFromOnnx(CreateTypedUnaryModel("Relu", "Int32"));
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedInt32Unary(generated, "unary", inputValues));
        Assert.Equal(expected, ExecuteOnnxInt32Unary(emitted.Value!, inputValues));
    }

    [Theory]
    [InlineData("Condition", "Int32", "Float32")]
    [InlineData("Value", "Boolean", "Float16")]
    [InlineData("BranchMismatch", "Boolean", "Int32")]
    public void WhereRejectsKnownInputsOutsideItsCpuAndTorchIntersection(
        string invalidCase,
        string conditionType,
        string valueType)
    {
        var model = CreateTypedWhereModel(invalidCase, conditionType, valueType);

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic => diagnostic.Code == CompilerDiagnosticCodes.Unsupported);
    }

    [Fact]
    public void ClipRejectsFloatBoundsForKnownNonFloatInput()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<int>([4]));
        var minimum = model.Graph.AddInput("minimum", OnnxTensorType.Create<float>([1]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<int>([4]));
        model.Graph.AddNode(new Onnxify.Clip("clip", new Onnxify.ClipInputOutputOptions
        {
            Input = input,
            Min = minimum,
            Output = output,
        }));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("scalar bounds are Float32", StringComparison.Ordinal));
    }

    [Fact]
    public void WhereInt32ImportExportAndGeneratedTorchSharpPreserveValues()
    {
        var model = CreateTypedWhereModel("Valid", "Boolean", "Int32");
        var conditionValues = new[] { true, false, true, false };
        var xValues = new[] { 10, 20, 30, 40 };
        var yValues = new[] { -1, -2, -3, -4 };
        var expected = new[] { 10, -2, 30, -4 };

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedInt32Where(generated, conditionValues, xValues, yValues));
        Assert.Equal(expected, ExecuteOnnxInt32Where(emitted.Value!, conditionValues, xValues, yValues));
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

        var generatedSource = Compiler.GenerateSourceCode(onnxTree.Value);
        Assert.True(generatedSource.IsSuccess, FormatDiagnostics(generatedSource.Diagnostics));
        Assert.True(expected.SequenceEqual(ExecuteGeneratedModule(generatedSource, leftValues, rightValues)));

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

    [Fact]
    public void IsFiniteCompositeMappingIsBidirectionalAndMatchesOnnxRuntime()
    {
        // Torch converter: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py::aten_isfinite (And(Not(IsInf), Not(IsNaN))).
        // ONNX Runtime: tensor/isinf_test.cc and tensor/isnan_test.cc, plus math/element_wise_ops_test.cc (Not/And).
        var values = new[] { float.NegativeInfinity, -1f, 0f, float.PositiveInfinity, float.NaN };
        var expected = new[] { false, true, true, false, false };
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([new OnnxDimension<long>(5)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        var inf = model.Graph.AddValue("inf", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        var notInf = model.Graph.AddValue("not_inf", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        var nan = model.Graph.AddValue("nan", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        var notNan = model.Graph.AddValue("not_nan", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        model.Graph.AddNode(new Onnxify.IsInf("isinf", new Onnxify.IsInfInputOutputOptions { X = new OnnxEdge(input.Name), Y = inf }));
        model.Graph.AddNode(new Onnxify.Not("notinf", new Onnxify.NotInputOutputOptions { X = new OnnxEdge(inf.Name), Y = notInf }));
        model.Graph.AddNode(new Onnxify.IsNaN("isnan", new Onnxify.IsNaNInputOutputOptions { X = new OnnxEdge(input.Name), Y = nan }));
        model.Graph.AddNode(new Onnxify.Not("notnan", new Onnxify.NotInputOutputOptions { X = new OnnxEdge(nan.Name), Y = notNan }));
        model.Graph.AddNode(new Onnxify.And("and", new Onnxify.AndInputOutputOptions { A = new OnnxEdge(notInf.Name), B = new OnnxEdge(notNan.Name), C = new OnnxEdge(output.Name) }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var step = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("IsFinite", step.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, step.Descriptor.Capability);
        var source = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(source.IsSuccess, FormatDiagnostics(source.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedUnaryModule(source, values));

        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(
            "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return torch.isfinite(input); }"));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        Assert.Equal(step.Descriptor, Assert.IsType<CompilerOnnxStep>(Assert.Single(torchTree.Value!.Operations)).Descriptor);
        var emitted = Compiler.GenerateOnnx(torchTree.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["IsInf", "Not", "IsNaN", "Not", "And"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        Assert.Equal(expected, ExecuteOnnxUnary(emitted.Value, values));

        var doubleValues = new[] { double.NegativeInfinity, -1d, 0d, double.PositiveInfinity, double.NaN };
        var doubleModel = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var doubleInput = doubleModel.Graph.AddInput("input", OnnxTensorType.Create<double>([new OnnxDimension<long>(5)]));
        var doubleOutput = doubleModel.Graph.AddOutput("output", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        var doubleInf = doubleModel.Graph.AddValue("inf", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        var doubleNotInf = doubleModel.Graph.AddValue("not_inf", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        var doubleNan = doubleModel.Graph.AddValue("nan", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        var doubleNotNan = doubleModel.Graph.AddValue("not_nan", OnnxTensorType.Create<bool>([new OnnxDimension<long>(5)]));
        doubleModel.Graph.AddNode(new Onnxify.IsInf("isinf", new Onnxify.IsInfInputOutputOptions { X = new OnnxEdge(doubleInput.Name), Y = doubleInf }));
        doubleModel.Graph.AddNode(new Onnxify.Not("notinf", new Onnxify.NotInputOutputOptions { X = new OnnxEdge(doubleInf.Name), Y = doubleNotInf }));
        doubleModel.Graph.AddNode(new Onnxify.IsNaN("isnan", new Onnxify.IsNaNInputOutputOptions { X = new OnnxEdge(doubleInput.Name), Y = doubleNan }));
        doubleModel.Graph.AddNode(new Onnxify.Not("notnan", new Onnxify.NotInputOutputOptions { X = new OnnxEdge(doubleNan.Name), Y = doubleNotNan }));
        doubleModel.Graph.AddNode(new Onnxify.And("and", new Onnxify.AndInputOutputOptions { A = new OnnxEdge(doubleNotInf.Name), B = new OnnxEdge(doubleNotNan.Name), C = new OnnxEdge(doubleOutput.Name) }));

        var importedDouble = Compiler.CreateTreeFromOnnx(doubleModel);
        Assert.True(importedDouble.IsSuccess, FormatDiagnostics(importedDouble.Diagnostics));
        var generatedDouble = Compiler.GenerateSourceCode(importedDouble.Value!);
        Assert.True(generatedDouble.IsSuccess, FormatDiagnostics(generatedDouble.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedUnaryModule(generatedDouble, doubleValues));
        using (var eagerInput = global::TorchSharp.torch.tensor(doubleValues, [5L], dtype: global::TorchSharp.torch.ScalarType.Float64))
        using (var eagerOutput = global::TorchSharp.torch.isfinite(eagerInput))
        {
            Assert.Equal(expected, eagerOutput.data<bool>().ToArray());
        }
        Assert.Equal(expected, ExecuteOnnxUnary(doubleModel, doubleValues));
    }

    [Fact]
    public void IsFiniteFloat16ImportsGeneratesAndMatchesOnnxRuntimeAndTorchSharp()
    {
        var values = new[] { float.NegativeInfinity, -1f, 0f, float.PositiveInfinity, float.NaN };
        var expected = new[] { false, true, true, false, false };
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<Half>([new OnnxDimension<long>(values.Length)]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<bool>([new OnnxDimension<long>(values.Length)]));
        var inf = model.Graph.AddValue("inf", OnnxTensorType.Create<bool>([new OnnxDimension<long>(values.Length)]));
        var notInf = model.Graph.AddValue("not_inf", OnnxTensorType.Create<bool>([new OnnxDimension<long>(values.Length)]));
        var nan = model.Graph.AddValue("nan", OnnxTensorType.Create<bool>([new OnnxDimension<long>(values.Length)]));
        var notNan = model.Graph.AddValue("not_nan", OnnxTensorType.Create<bool>([new OnnxDimension<long>(values.Length)]));
        model.Graph.AddNode(new Onnxify.IsInf("isinf", new Onnxify.IsInfInputOutputOptions { X = new OnnxEdge(input.Name), Y = inf }));
        model.Graph.AddNode(new Onnxify.Not("notinf", new Onnxify.NotInputOutputOptions { X = new OnnxEdge(inf.Name), Y = notInf }));
        model.Graph.AddNode(new Onnxify.IsNaN("isnan", new Onnxify.IsNaNInputOutputOptions { X = new OnnxEdge(input.Name), Y = nan }));
        model.Graph.AddNode(new Onnxify.Not("notnan", new Onnxify.NotInputOutputOptions { X = new OnnxEdge(nan.Name), Y = notNan }));
        model.Graph.AddNode(new Onnxify.And("and", new Onnxify.AndInputOutputOptions { A = new OnnxEdge(notInf.Name), B = new OnnxEdge(notNan.Name), C = new OnnxEdge(output.Name) }));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var step = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("IsFinite", step.Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedHalfBooleanUnaryModule(generated, values, model));

        using var eagerInput = global::TorchSharp.torch.tensor(values, [values.Length], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerOutput = global::TorchSharp.torch.isfinite(eagerInput);
        Assert.Equal(expected, eagerOutput.data<bool>().ToArray());

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["IsInf", "Not", "IsNaN", "Not", "And"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        Assert.Equal(expected, ExecuteOnnxHalfBooleanUnary(emitted.Value, values));
    }

    [Fact]
    public void TorchSignbitRemainsUnsupportedBecausePointwiseComparisonLosesSignBit()
    {
        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return torch.signbit(input); }";
        var tree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));

        Assert.True(tree.IsSuccess, FormatDiagnostics(tree.Diagnostics));
        Assert.DoesNotContain(tree.Value!.Operations.OfType<CompilerOnnxStep>(),
            static operation => operation.Descriptor.Name == "Signbit");
        var emitted = Compiler.GenerateOnnx(tree.Value);
        Assert.False(emitted.IsSuccess);
        var diagnostic = Assert.Single(emitted.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);

        var values = new[]
        {
            -0f,
            0f,
            BitConverter.Int32BitsToSingle(unchecked((int)0xFFC00000)),
            BitConverter.Int32BitsToSingle(0x7FC00000),
        };
        using var eagerInput = global::TorchSharp.torch.tensor(values);
        using var eagerOutput = global::TorchSharp.torch.signbit(eagerInput);
        var expected = eagerOutput.data<bool>().ToArray();
        Assert.Equal([true, false, true, false], expected);
    }

    [Fact]
    public void IsCloseCompositeMappingMatchesDefaultTorchSharpSemantics()
    {
        // Torch converter: third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py::aten_isclose; corrected with Equal/Or for matching infinities.
        // ONNX Runtime: element_wise_ops_test.cc (Sub, Abs, Mul, Add, LessOrEqual, Equal, Or).
        var leftValues = new[] { 100f, 2f, float.NegativeInfinity, 10f, 0f, float.NaN };
        var rightValues = new[] { 100.0005f, 2.01f, float.NegativeInfinity, 10f, 1e-9f, float.NaN };
        var expected = new[] { true, false, true, true, true, false };
        var source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) { return left.isclose(right, 1e-5, 1e-8, false); }";
        var torchTree = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(torchTree.IsSuccess, FormatDiagnostics(torchTree.Diagnostics));
        var emitted = Compiler.GenerateOnnx(torchTree.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(["CastLike", "CastLike", "Sub", "Abs", "Abs", "Mul", "Add", "LessOrEqual", "Equal", "Or"], emitted.Value!.Graph.Nodes.Select(static node => node.OpType));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, leftValues, rightValues));

        var imported = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var step = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.Equal("IsClose", step.Descriptor.Name);
        Assert.Equal(CompilerOperationCapability.Bidirectional, step.Descriptor.Capability);
        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedModule(generated, leftValues, rightValues, emitted.Value));

        using var left = global::TorchSharp.torch.tensor(leftValues, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var right = global::TorchSharp.torch.tensor(rightValues, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var eager = left.isclose(right, 1e-5, 1e-8, false);
        Assert.Equal(expected, eager.data<bool>().ToArray());
    }

    [Fact]
    public void IsCloseFloat16UsesInputTypedTolerancesAndMatchesTorchSharpAndOnnxRuntime()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<Half>([4, 1]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<Half>([1, 4]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<bool>([4, 4]));
        model.Graph.AddNode(new OnnxNode(
            "isclose",
            "IsClose",
            "com.onnxify.compiler",
            string.Empty,
            [new OnnxEdge(left.Name), new OnnxEdge(right.Name)],
            [new OnnxEdge(output.Name)],
            []));

        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var emitted = Compiler.GenerateOnnx(imported.Value!);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.Equal(
            ["CastLike", "CastLike", "Sub", "Abs", "Abs", "Mul", "Add", "LessOrEqual", "Equal", "Or"],
            emitted.Value!.Graph.Nodes.Select(static node => node.OpType));

        var leftValues = new[] { (Half)100f, Half.NaN, (Half)0f, (Half)0f };
        var rightValues = new[] { (Half)100f, (Half)100.0625f, (Half)0f, Half.Epsilon };
        using var eagerLeft = global::TorchSharp.torch.tensor(leftValues.Select(static value => (float)value).ToArray(), [4L, 1L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerRight = global::TorchSharp.torch.tensor(rightValues.Select(static value => (float)value).ToArray(), [1L, 4L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var eagerOutput = eagerLeft.isclose(eagerRight, 1e-5, 1e-8, false);
        var expected = eagerOutput.data<bool>().ToArray();
        Assert.Equal(
            [true, false, false, false, false, false, false, false, false, false, true, false, false, false, true, false],
            expected);
        Assert.Equal(expected, ExecuteOnnxHalfBooleanBroadcast(emitted.Value, leftValues, rightValues));

        var roundTrip = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(roundTrip.IsSuccess, FormatDiagnostics(roundTrip.Diagnostics));
        Assert.Equal("IsClose", Assert.IsType<CompilerOnnxStep>(Assert.Single(roundTrip.Value!.Operations)).Descriptor.Name);
        var generated = Compiler.GenerateSourceCode(roundTrip.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedHalfBooleanBroadcast(generated, leftValues, rightValues, emitted.Value));
    }

    [Theory]
    [InlineData("IsFinite", true)]
    [InlineData("Heaviside", false)]
    [InlineData("XLogY", false)]
    [InlineData("Logit", false)]
    [InlineData("IsClose", true)]
    public void FloatingPointCompositesRejectIntegerInputs(string operationName, bool booleanOutput)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new OnnxDimension[] { new OnnxDimension<long>(2) };
        var inputCount = operationName is "IsFinite" or "Logit" ? 1 : 2;
        var inputs = Enumerable.Range(0, inputCount)
            .Select(index => model.Graph.AddInput($"input{index}", OnnxTensorType.Create<int>(shape)))
            .ToArray();
        var output = model.Graph.AddOutput(
            "output",
            booleanOutput ? OnnxTensorType.Create<bool>(shape) : OnnxTensorType.Create<int>(shape));
        model.Graph.AddNode(new OnnxNode(
            $"unsupported_{operationName.ToLowerInvariant()}",
            operationName,
            "com.onnxify.compiler",
            string.Empty,
            inputs.Select(static input => (IOnnxGraphEdge)new OnnxEdge(input.Name)),
            [new OnnxEdge(output.Name)],
            []));

        var result = Compiler.CreateTreeFromOnnx(model);

        Assert.False(result.IsSuccess);
        Assert.Contains(result.Diagnostics, diagnostic =>
            diagnostic.Code == CompilerDiagnosticCodes.Unsupported
            && diagnostic.Message.Contains("Float32/Float64", StringComparison.Ordinal));
    }

    [Fact]
    public void IsInfMappingIsBidirectionalAndMatchesOnnxRuntime()
    {
        // Torch converter case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (isinf), exercised by ops_test.py::test_output_match_opinfo_.
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/isinf_test.cc (IsInfTest.test_isinf_float10, IsInfTest.test_isinf_positive_float10, IsInfTest.test_isinf_negative_float10).
        var inputValues = new[] { float.NegativeInfinity, -1f, float.PositiveInfinity, float.NaN };
        var expected = new[] { true, false, true, false };
        var model = CreateUnaryModel(
            inputType: OnnxTensorType.Create<float>([new OnnxDimension<long>(4)]),
            node: new Onnxify.IsInf(
                "isinf",
                new Onnxify.IsInfInputOutputOptions
                {
                    X = new OnnxEdge("input"),
                    Y = new OnnxEdge("output"),
                }),
            outputType: OnnxTensorType.Create<bool>([new OnnxDimension<long>(4)]));

        VerifyUnaryBooleanMapping(
            model,
            "torch.isinf(input)",
            "IsInf",
            inputValues,
            expected);
    }

    [Fact]
    public void IsInfSignSpecificMappingsPreserveOnnxAttributes()
    {
        // Torch converter cases: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (isposinf, isneginf).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/isinf_test.cc (IsInfTest.test_isinf_positive_float10, IsInfTest.test_isinf_negative_float10).
        var inputValues = new[] { float.NegativeInfinity, -1f, float.PositiveInfinity, float.NaN };
        var cases = new[]
        {
            (Name: "isposinf", Call: "torch.isposinf(input)", Negative: 0L, Positive: 1L, Expected: new[] { false, false, true, false }),
            (Name: "isneginf", Call: "torch.isneginf(input)", Negative: 1L, Positive: 0L, Expected: new[] { true, false, false, false }),
        };

        foreach (var testCase in cases)
        {
            var model = CreateUnaryModel(
                inputType: OnnxTensorType.Create<float>([new OnnxDimension<long>(4)]),
                node: new Onnxify.IsInf(
                    testCase.Name,
                    new Onnxify.IsInfInputOutputOptions
                    {
                        X = new OnnxEdge("input"),
                        DetectNegative = testCase.Negative,
                        DetectPositive = testCase.Positive,
                        Y = new OnnxEdge("output"),
                    }),
                outputType: OnnxTensorType.Create<bool>([new OnnxDimension<long>(4)]));

            VerifyUnaryBooleanMapping(
                model,
                testCase.Call,
                "IsInf",
                inputValues,
                testCase.Expected);
        }
    }

    private static OnnxModel CreateUnaryModel(OnnxTensorType inputType, OnnxNode node, OnnxTensorType outputType)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        model.Graph.AddInput("input", inputType);
        model.Graph.AddOutput("output", outputType);
        model.Graph.AddNode(node);
        return model;
    }

    private static bool[] ExecuteOnnxHalfBooleanUnary(OnnxModel model, float[] inputValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-half-bool-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var input = inputValues.Select(static value => (global::Microsoft.ML.OnnxRuntime.Float16)value).ToArray();
            var output = new bool[inputValues.Length];
            using var inputBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance, input, TensorElementType.Float16, [input.Length], input.Length * sizeof(ushort));
            using var outputBuffer = FixedBufferOnnxValue.CreateFromMemory<bool>(
                OrtMemoryInfo.DefaultInstance, output, TensorElementType.Bool, [output.Length], output.Length * sizeof(bool));
            session.Run(["input"], [inputBuffer], ["output"], [outputBuffer]);
            return output;
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static bool[] ExecuteOnnxHalfBooleanBroadcast(OnnxModel model, Half[] leftValues, Half[] rightValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-half-bool-broadcast-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var left = leftValues.Select(static value => (global::Microsoft.ML.OnnxRuntime.Float16)(float)value).ToArray();
            var right = rightValues.Select(static value => (global::Microsoft.ML.OnnxRuntime.Float16)(float)value).ToArray();
            var output = new bool[leftValues.Length * rightValues.Length];
            using var leftBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance, left, TensorElementType.Float16, [left.Length, 1], left.Length * sizeof(ushort));
            using var rightBuffer = FixedBufferOnnxValue.CreateFromMemory<global::Microsoft.ML.OnnxRuntime.Float16>(
                OrtMemoryInfo.DefaultInstance, right, TensorElementType.Float16, [1, right.Length], right.Length * sizeof(ushort));
            using var outputBuffer = FixedBufferOnnxValue.CreateFromMemory<bool>(
                OrtMemoryInfo.DefaultInstance, output, TensorElementType.Bool, [leftValues.Length, rightValues.Length], output.Length * sizeof(bool));
            session.Run(["left", "right"], [leftBuffer, rightBuffer], ["output"], [outputBuffer]);
            return output;
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static bool[] ExecuteGeneratedHalfBooleanBroadcast(
        CompilerResult<IFile[]> generated,
        Half[] leftValues,
        Half[] rightValues,
        OnnxModel weightsModel)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.HalfBooleanBroadcast{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "isclose_half", weightsModel);
        using var left = global::TorchSharp.torch.tensor(leftValues.Select(static value => (float)value).ToArray(), [leftValues.Length, 1L], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var right = global::TorchSharp.torch.tensor(rightValues.Select(static value => (float)value).ToArray(), [1L, rightValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [left, right])!;
        return output.data<bool>().ToArray();
    }

    private static bool[] ExecuteGeneratedHalfBooleanUnaryModule(
        CompilerResult<IFile[]> generated,
        float[] inputValues,
        OnnxModel weightsModel)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.HalfBoolean{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "isfinite_half", weightsModel);
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Float16);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<bool>().ToArray();
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
        var generatedSource = Compiler.GenerateSourceCode(onnxTree.Value);
        Assert.True(generatedSource.IsSuccess, FormatDiagnostics(generatedSource.Diagnostics));
        Assert.Equal(expected, ExecuteGeneratedUnaryModule(generatedSource, inputValues));

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
        if (opType == "NotEqual")
        {
            var equal = model.Graph.AddValue("equal", OnnxTensorType.Create<bool>(leftShape));
            model.Graph.AddNode(new Onnxify.Equal("pointwise_equal", new Onnxify.EqualInputOutputOptions
            {
                A = leftEdge,
                B = rightEdge,
                C = equal,
            }));
            model.Graph.AddNode(new Onnxify.Not("pointwise_not", new Onnxify.NotInputOutputOptions
            {
                X = new OnnxEdge(equal.Name),
                Y = outputEdge,
            }));
        }
        else
        {
            var node = CreateNode(opType, leftEdge, rightEdge, outputEdge);
            model.Graph.AddNode(node);
        }
        return model;
    }

    private static OnnxModel CreateTypedComparisonModel(
        string opType,
        CompilerElementType leftElementType,
        CompilerElementType rightElementType)
    {
        static OnnxTensorType TensorType(CompilerElementType elementType, OnnxDimension[] dimensions) => elementType switch
        {
            CompilerElementType.Float16 => OnnxTensorType.Create<Half>(dimensions),
            CompilerElementType.Float32 => OnnxTensorType.Create<float>(dimensions),
            CompilerElementType.Int32 => OnnxTensorType.Create<int>(dimensions),
            CompilerElementType.Boolean => OnnxTensorType.Create<bool>(dimensions),
            _ => throw new ArgumentOutOfRangeException(nameof(elementType), elementType, "Unsupported comparison test fixture type."),
        };

        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var dimensions = new OnnxDimension[] { new OnnxDimension<long>(4) };
        var left = model.Graph.AddInput("left", TensorType(leftElementType, dimensions));
        var right = model.Graph.AddInput("right", TensorType(rightElementType, dimensions));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<bool>(dimensions));
        model.Graph.AddNode(CreateNode(opType, new OnnxEdge(left.Name), new OnnxEdge(right.Name), new OnnxEdge(output.Name)));
        return model;
    }

    private static OnnxModel CreateTypedUnaryModel(string opType, string elementTypeName)
    {
        static OnnxTensorType TensorType(string elementTypeName, OnnxDimension[] dimensions) => elementTypeName switch
        {
            "Boolean" => OnnxTensorType.Create<bool>(dimensions),
            "Float16" => OnnxTensorType.Create<Half>(dimensions),
            "Float32" => OnnxTensorType.Create<float>(dimensions),
            "Int32" => OnnxTensorType.Create<int>(dimensions),
            "UInt8" => OnnxTensorType.Create<byte>(dimensions),
            "UInt16" => OnnxTensorType.Create<ushort>(dimensions),
            _ => throw new ArgumentOutOfRangeException(nameof(elementTypeName), elementTypeName, "Unsupported unary test fixture type."),
        };

        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var dimensions = new OnnxDimension[] { new OnnxDimension<long>(4) };
        var inputType = TensorType(elementTypeName, dimensions);
        var input = model.Graph.AddInput("input", inputType);
        var output = model.Graph.AddOutput("output", inputType);
        var inputEdge = new OnnxEdge(input.Name);
        var outputEdge = new OnnxEdge(output.Name);
        var node = opType switch
        {
            "Abs" => (OnnxNode)new Onnxify.Abs("unary", new Onnxify.AbsInputOutputOptions { X = inputEdge, Y = outputEdge }),
            "Neg" => new Onnxify.Neg("unary", new Onnxify.NegInputOutputOptions { X = inputEdge, Y = outputEdge }),
            "Exp" => new Onnxify.Exp("unary", new Onnxify.ExpInputOutputOptions { Input = inputEdge, Output = outputEdge }),
            "Sqrt" => new Onnxify.Sqrt("unary", new Onnxify.SqrtInputOutputOptions { X = inputEdge, Y = outputEdge }),
            "Round" => new Onnxify.Round("unary", new Onnxify.RoundInputOutputOptions { X = inputEdge, Y = outputEdge }),
            "Erf" => new Onnxify.Erf("unary", new Onnxify.ErfInputOutputOptions { Input = inputEdge, Output = outputEdge }),
            "Relu" => new Onnxify.Relu("unary", new Onnxify.ReluInputOutputOptions { X = inputEdge, Y = outputEdge }),
            "Sigmoid" => new Onnxify.Sigmoid("unary", new Onnxify.SigmoidInputOutputOptions { X = inputEdge, Y = outputEdge }),
            "Tanh" => new Onnxify.Tanh("unary", new Onnxify.TanhInputOutputOptions { Input = inputEdge, Output = outputEdge }),
            "Sign" => new Onnxify.Sign("unary", new Onnxify.SignInputOutputOptions { Input = inputEdge, Output = outputEdge }),
            _ => throw new ArgumentOutOfRangeException(nameof(opType), opType, "Unsupported unary test fixture operator."),
        };
        model.Graph.AddNode(node);
        return model;
    }

    private static OnnxModel CreateTypedWhereModel(string invalidCase, string conditionTypeName, string valueTypeName)
    {
        static OnnxTensorType TensorType(string name, OnnxDimension[] dimensions) => name switch
        {
            "Boolean" => OnnxTensorType.Create<bool>(dimensions),
            "Float16" => OnnxTensorType.Create<Half>(dimensions),
            "Float32" => OnnxTensorType.Create<float>(dimensions),
            "Int32" => OnnxTensorType.Create<int>(dimensions),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unsupported Where test fixture type."),
        };

        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var shape = new OnnxDimension[] { new OnnxDimension<long>(4) };
        var condition = model.Graph.AddInput("condition", TensorType(conditionTypeName, shape));
        var x = model.Graph.AddInput("x", TensorType(valueTypeName, shape));
        var y = model.Graph.AddInput("y", TensorType(invalidCase == "BranchMismatch" ? "Float32" : valueTypeName, shape));
        var outputType = TensorType(invalidCase == "BranchMismatch" ? "Int32" : valueTypeName, shape);
        var output = model.Graph.AddOutput("output", outputType);
        model.Graph.AddNode(new Onnxify.Where("where", new Onnxify.WhereInputOutputOptions
        {
            Condition = new OnnxEdge(condition.Name),
            X = new OnnxEdge(x.Name),
            Y = new OnnxEdge(y.Name),
            Output = new OnnxEdge(output.Name),
        }));
        return model;
    }

    private static int[] ExecuteGeneratedInt32Where(
        CompilerResult<IFile[]> generated,
        bool[] conditionValues,
        int[] xValues,
        int[] yValues)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Int32Where{Guid.NewGuid():N}",
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
        using var condition = global::TorchSharp.torch.tensor(conditionValues, [conditionValues.Length], dtype: global::TorchSharp.torch.ScalarType.Bool);
        using var x = global::TorchSharp.torch.tensor(xValues, [xValues.Length], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var y = global::TorchSharp.torch.tensor(yValues, [yValues.Length], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [condition, x, y])!;
        return output.data<int>().ToArray();
    }

    private static int[] ExecuteOnnxInt32Where(OnnxModel model, bool[] conditionValues, int[] xValues, int[] yValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-int32-where-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var outputs = session.Run([
                NamedOnnxValue.CreateFromTensor("condition", new DenseTensor<bool>(conditionValues, [conditionValues.Length])),
                NamedOnnxValue.CreateFromTensor("x", new DenseTensor<int>(xValues, [xValues.Length])),
                NamedOnnxValue.CreateFromTensor("y", new DenseTensor<int>(yValues, [yValues.Length])),
            ]);
            return outputs.Single().AsTensor<int>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static int[] ExecuteOnnxInt32Unary(OnnxModel model, int[] inputValues)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-int32-unary-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var outputs = session.Run([
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<int>(inputValues, [inputValues.Length])),
            ]);
            return outputs.Single().AsTensor<int>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static int[] ExecuteGeneratedInt32Unary(CompilerResult<IFile[]> generated, string operationName, int[] inputValues)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.Int32Unary{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, operationName);
        using var input = global::TorchSharp.torch.tensor(inputValues, [inputValues.Length], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [input])!;
        return output.data<int>().ToArray();
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

    private static bool[] ExecuteGeneratedModule(CompilerResult<IFile[]> generated, float[] left, float[] right, OnnxModel? weightsModel = null)
    {
        var source = generated.SourceCode();
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
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "pointwise", weightsModel);
        using var leftTensor = global::TorchSharp.torch.tensor(left, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var rightTensor = global::TorchSharp.torch.tensor(
            right,
            right.Length == 6 ? [2L, 3L] : [3L],
            dtype: global::TorchSharp.torch.ScalarType.Float32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [leftTensor, rightTensor])!;
        return output.data<bool>().ToArray();
    }

    private static bool[] ExecuteGeneratedModule(CompilerResult<IFile[]> generated, bool[] left, bool[] right)
    {
        var source = generated.SourceCode();
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
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "pointwise");
        using var leftTensor = global::TorchSharp.torch.tensor(left, [2L, 3L], dtype: global::TorchSharp.torch.ScalarType.Bool);
        using var rightTensor = global::TorchSharp.torch.tensor(right, [3L], dtype: global::TorchSharp.torch.ScalarType.Bool);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [leftTensor, rightTensor])!;
        return output.data<bool>().ToArray();
    }

    private static bool[] ExecuteOnnx(OnnxModel model, float[] left, float[] right)
    {
        return ExecuteOnnx(
            model,
            new DenseTensor<float>(left, [2, 3]),
            right.Length == 6 ? new DenseTensor<float>(right, [2, 3]) : new DenseTensor<float>(right, [3]));
    }

    private static bool[] ExecuteOnnx(OnnxModel model, bool[] left, bool[] right)
    {
        return ExecuteOnnx(model, new DenseTensor<bool>(left, [2, 3]), new DenseTensor<bool>(right, [3]));
    }

    private static bool[] ExecuteOnnxInt32Comparison(OnnxModel model, int[] left, int[] right)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-int32-comparison-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            using var outputs = session.Run([
                NamedOnnxValue.CreateFromTensor("left", new DenseTensor<int>(left, [left.Length])),
                NamedOnnxValue.CreateFromTensor("right", new DenseTensor<int>(right, [right.Length])),
            ]);
            return outputs.Single().AsTensor<bool>().ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static bool[] ExecuteGeneratedInt32Comparison(CompilerResult<IFile[]> generated, int[] left, int[] right)
    {
        var source = generated.SourceCode();
        var compilation = CSharpCompilation.Create(
            $"Onnxify.BooleanPointwiseInt32{Guid.NewGuid():N}",
            [CSharpSyntaxTree.ParseText(source)],
            CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics.Where(static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetTypes().Single(static type => type.Name.Contains("TorchModule", StringComparison.Ordinal));
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "pointwise");
        using var leftTensor = global::TorchSharp.torch.tensor(left, [left.Length], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var rightTensor = global::TorchSharp.torch.tensor(right, [right.Length], dtype: global::TorchSharp.torch.ScalarType.Int32);
        using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, [leftTensor, rightTensor])!;
        return output.data<bool>().ToArray();
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

    private static bool[] ExecuteGeneratedUnaryModule<TInput>(CompilerResult<IFile[]> generated, TInput[] inputValues)
    {
        var source = generated.SourceCode();
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
        using var module = GeneratedModuleTestHelpers.CreateModule(moduleType, generated, "boolean_unary");
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
            double[] doubleValues => global::TorchSharp.torch.tensor(
                doubleValues,
                [doubleValues.Length],
                dtype: global::TorchSharp.torch.ScalarType.Float64),
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
            else if (inputValues is double[] doubleValues)
            {
                input = NamedOnnxValue.CreateFromTensor("input", new DenseTensor<double>(doubleValues, [doubleValues.Length]));
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
