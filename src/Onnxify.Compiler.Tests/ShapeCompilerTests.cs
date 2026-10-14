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

        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("torch.flatten(input, start_dim: 1)", generated.SourceCode());

        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values));

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
            var generated = Compiler.GenerateSourceCode(imported.Value!);
            Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));

            var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
            var expected = ExecuteOnnx(model, values);
            Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values));

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

        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("input.permute(new long[] { 1, 2, 0 })", generated.SourceCode());

        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values));

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

    [Fact]
    public void ReshapeWithStaticShapeInitializerRoundTripsAndPreservesRuntimeValues()
    {
        // ONNXScript cases: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (reshape, view).
        // ONNX Runtime cases: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/tensor_op_test.cc (TensorOpTest.Reshape, TensorOpTest.ReshapeWithInitializer).
        var model = CreateReshapeModel([6, 4]);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Reshape>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("input.reshape(new long[] { 6L, 4L })", generated.SourceCode());

        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.view(new long[] { 6, 4 }); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        Assert.IsType<Onnxify.Reshape>(torchStep.Node);
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);
        Assert.Single(scanned.Value.Initializers);

        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Reshape>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, values));
    }

    [Fact]
    public void ReshapeRejectsDynamicOrInvalidStaticShapes()
    {
        const string dynamicShapeSource = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input, global::TorchSharp.torch.Tensor shape) { return torch.reshape(input, shape); }";
        var dynamicShape = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(dynamicShapeSource));
        Assert.False(dynamicShape.IsSuccess);
        Assert.Contains(dynamicShape.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error);

        const string invalidShapeSource = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.reshape(new long[] { -1, -1 }); }";
        var invalidShape = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(invalidShapeSource));
        Assert.False(invalidShape.IsSuccess);
        Assert.Contains(invalidShape.Diagnostics, static diagnostic => diagnostic.Message.Contains("at most one", StringComparison.Ordinal));
    }

    [Fact]
    public void ReshapePreservesCopiedDimensionsAndInferredDimension()
    {
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/tensor_op_test.cc (TensorOpTest.Reshape_UnknownDimWithoutAllowZero).
        var model = CreateReshapeModel([0, -1], allowzero: 0, outputShape: [2, 12]);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));

        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("input.reshape(new long[] { input.shape[0], -1L })", generated.SourceCode());

        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values));
    }

    [Fact]
    public void UnsqueezeStaticAxisRoundTripsAndPreservesRuntimeValues()
    {
        // ONNXScript case: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (aten_unsqueeze).
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/unsqueeze_op_test.cc (UnsqueezeOpTest.Unsqueeze_1).
        var model = CreateUnsqueezeModel(1);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Unsqueeze>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("input.unsqueeze(1)", generated.SourceCode());

        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.unsqueeze(1); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        Assert.IsType<Onnxify.Unsqueeze>(torchStep.Node);
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);
        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Unsqueeze>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, values));
    }

    [Fact]
    public void SqueezeStaticAxisRoundTripsAndPreservesRuntimeValues()
    {
        // ONNXScript source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (squeeze and squeeze_dim).
        // ONNX Runtime case: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/squeeze_op_test.cc (SqueezeOpTest.Squeeze_1).
        var model = CreateSqueezeModel(1);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Squeeze>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("input.squeeze(1)", generated.SourceCode());

        var values = Enumerable.Range(0, 8).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values, [2, 1, 4]);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values, [2, 1, 4]));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.squeeze(1); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.False(scanned.IsSuccess);
        Assert.Contains(scanned.Diagnostics, static diagnostic => diagnostic.Severity == CompilerDiagnosticSeverity.Error
            && diagnostic.Message.Contains("known input rank", StringComparison.Ordinal));

        var descriptor = new CompilerTorchSharpModuleDescriptor(
            typeof(SqueezeFixture).Assembly.Location,
            typeof(SqueezeFixture).FullName!,
            methodName: nameof(SqueezeFixture.forward),
            methodMetadataToken: typeof(SqueezeFixture).GetMethod(nameof(SqueezeFixture.forward))!.MetadataToken,
            inputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "input",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerFixedDimension(2), new CompilerFixedDimension(1), new CompilerFixedDimension(4)])),
            ],
            outputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "output",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerFixedDimension(2), new CompilerFixedDimension(4)])),
            ]);
        var metadataScan = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(descriptor));
        Assert.True(metadataScan.IsSuccess, FormatDiagnostics(metadataScan.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(metadataScan.Value!.Operations));
        Assert.IsType<Onnxify.Squeeze>(torchStep.Node);
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);

        var emitted = Compiler.GenerateOnnx(metadataScan.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Squeeze>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, values, [2, 1, 4]));
    }

    [Fact]
    public void ExpandStaticShapeRoundTripsAndPreservesRuntimeValues()
    {
        // ONNXScript source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py, TorchLibOpInfo("expand", core_ops.aten_expand).
        // ONNX Runtime source: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/expand_test.cc, ExpandOpTest.Expand_3x1.
        var model = CreateExpandModel([2, 3, 4]);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Expand>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("ExpandWithRuntimeShape(input, new long[] { 2L, 3L, 4L })", generated.SourceCode());

        var values = Enumerable.Range(0, 12).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values, [1, 3, 4]);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values, [1, 3, 4]));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.expand(new long[] { 2, 3, 4 }); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        Assert.IsType<Onnxify.Expand>(torchStep.Node);
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);

        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Expand>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, values, [1, 3, 4]));
    }

    [Fact]
    public void TileStaticRepeatsRoundTripAndPreservesRuntimeValues()
    {
        // ONNXScript source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py, TorchLibOpInfo("repeat", core_ops.aten_repeat).
        // ONNX Runtime source: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/tile_op_test.cc, TensorOpTest.TileFloatType.
        var model = CreateTileModel([1, 2, 1]);
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Tile>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("input.repeat(new long[] { 1L, 2L, 1L })", generated.SourceCode());

        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.repeat(new long[] { 1, 2, 1 }); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        Assert.IsType<Onnxify.Tile>(torchStep.Node);
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);

        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Tile>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, values));
    }

    [Fact]
    public void SliceStaticBoundsRoundTripAndPreservesRuntimeValues()
    {
        // ONNXScript source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py, TorchLibOpInfo("slice", core_ops.aten_slice).
        // ONNX Runtime source: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/slice_op.test.cc, SliceTest.Slice2D_OneAxis.
        var model = CreateSliceModel();
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Slice>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("input.slice(1, 1, 3, 1)", generated.SourceCode());

        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, values);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) { return input.slice(1, 1, 3, 1); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        Assert.IsType<Onnxify.Slice>(torchStep.Node);
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);

        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Slice>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, values));
    }

    [Fact]
    public void ConcatStaticAxisRoundTripsAndPreservesRuntimeValues()
    {
        // ONNXScript source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py, TorchLibOpInfo("cat", core_ops.aten_cat).
        // ONNX Runtime source: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/concat_op_test.cc, ConcatOpTest.Concat2D_1.
        var model = CreateConcatModel();
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Concat>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("torch.cat(new global::TorchSharp.torch.Tensor[] { left, right }, dim: 1)", generated.SourceCode());

        var left = Enumerable.Range(0, 4).Select(static value => (float)value).ToArray();
        var right = Enumerable.Range(4, 4).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(model, [left, right], [[2, 2], [2, 2]]);
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), [left, right], [[2, 2], [2, 2]]));

        const string source = "public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor left, global::TorchSharp.torch.Tensor right) { return torch.cat(new global::TorchSharp.torch.Tensor[] { left, right }, 1); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        Assert.IsType<Onnxify.Concat>(torchStep.Node);
        Assert.Equal(onnxStep.Descriptor, torchStep.Descriptor);

        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Concat>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(expected, ExecuteOnnx(emitted.Value, [left, right], [[2, 2], [2, 2]]));
    }

    [Fact]
    public void SplitStaticSizesPreserveOutputOrderAndRuntimeValues()
    {
        // ONNXScript source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py, TorchLibOpInfo("split_with_sizes", core_ops.aten_split_with_sizes).
        // ONNX Runtime source: third_party/onnxruntime/onnxruntime/test/providers/cpu/tensor/split_op_test.cc, SplitOperatorTest.Axis0UnequalSplitFloat.
        var model = CreateSplitModel();
        var imported = Compiler.CreateTreeFromOnnx(model);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(imported.Value!.Operations));
        Assert.IsType<Onnxify.Split>(onnxStep.Node);
        Assert.Equal(["first", "second"], onnxStep.Outputs.Select(static output => output.Name));
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);

        var generated = Compiler.GenerateSourceCode(imported.Value);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("torch.split(input, new long[] { 1L, 3L }, dim: 1)", generated.SourceCode());
        Assert.Contains("var first = __first_outputs[0];", generated.SourceCode());
        Assert.Contains("var second = __first_outputs[1];", generated.SourceCode());

        var values = Enumerable.Range(0, 8).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnxOutputs(model, [values], [[2, 4]]);
        var generatedValues = ExecuteGeneratedOutputs(generated.SourceCode(), [values], [[2L, 4L]]);
        Assert.Equal(expected, generatedValues);

        const string source = "public (global::TorchSharp.torch.Tensor, global::TorchSharp.torch.Tensor) forward(global::TorchSharp.torch.Tensor input) { return torch.split(input, new long[] { 1, 3 }, 1); }";
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var torchStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        Assert.IsType<Onnxify.Split>(torchStep.Node);
        Assert.Equal(["output0", "output1"], torchStep.Outputs.Select(static output => output.Name));

        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        var emittedNode = Assert.IsType<Onnxify.Split>(Assert.Single(emitted.Value!.Graph.Nodes));
        Assert.Equal(2, emittedNode.Out.Count);
        Assert.Equal(expected, ExecuteOnnxOutputs(emitted.Value, [values], [[2, 4]]));
    }

    [Fact]
    public void UnflattenWithKnownRankLowersToReshape()
    {
        // ONNXScript source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py, TorchLibOpInfo("unflatten", core_ops.aten_unflatten).
        var descriptor = new CompilerTorchSharpModuleDescriptor(
            typeof(UnflattenFixture).Assembly.Location,
            typeof(UnflattenFixture).FullName!,
            methodName: nameof(UnflattenFixture.forward),
            methodMetadataToken: typeof(UnflattenFixture).GetMethod(nameof(UnflattenFixture.forward))!.MetadataToken,
            inputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "input",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerFixedDimension(2), new CompilerFixedDimension(4), new CompilerFixedDimension(3)])),
            ],
            outputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "output",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerFixedDimension(2), new CompilerFixedDimension(2), new CompilerFixedDimension(2), new CompilerFixedDimension(3)])),
            ]);
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(descriptor));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var onnxStep = Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations));
        var reshape = Assert.IsType<Onnxify.Reshape>(onnxStep.Node);
        Assert.Equal(CompilerOperationCapability.Bidirectional, onnxStep.Descriptor.Capability);
        Assert.Contains(scanned.Value.Initializers, member => member.Name == reshape.Shape.Name);

        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Reshape>(Assert.Single(emitted.Value!.Graph.Nodes));
        var values = Enumerable.Range(0, 24).Select(static value => (float)value).ToArray();
        var expected = ExecuteOnnx(emitted.Value, values, [2, 4, 3]);
        Assert.Equal(values, expected);

        var imported = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Equal(expected, ExecuteGenerated(generated.SourceCode(), values, [2, 4, 3]));
    }

    [Theory]
    [InlineData(nameof(SplitFormsFixture.Chunk))]
    [InlineData(nameof(SplitFormsFixture.Unbind))]
    public void FixedOutputChunkAndUnbindLowerToSplit(string methodName)
    {
        // ONNXScript source: ops_test_data.py, TorchLibOpInfo("chunk", core_ops.aten_chunk) and TorchLibOpInfo("unbind", core_ops.aten_unbind).
        // ONNX Runtime source: split_op_test.cc, SplitOperatorTest.Axis0UnequalSplitFloat.
        var descriptor = new CompilerTorchSharpModuleDescriptor(
            typeof(SplitFormsFixture).Assembly.Location,
            typeof(SplitFormsFixture).FullName!,
            methodName: methodName,
            methodMetadataToken: typeof(SplitFormsFixture).GetMethod(methodName)!.MetadataToken,
            inputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "input",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerFixedDimension(2), new CompilerFixedDimension(4)])),
            ],
            outputs:
            [
                new CompilerTorchSharpValueDescriptor("first", new CompilerTensorType(CompilerElementType.Float32, methodName == nameof(SplitFormsFixture.Chunk)
                    ? [new CompilerFixedDimension(2), new CompilerFixedDimension(2)]
                    : [new CompilerFixedDimension(1), new CompilerFixedDimension(4)])),
                new CompilerTorchSharpValueDescriptor("second", new CompilerTensorType(CompilerElementType.Float32, methodName == nameof(SplitFormsFixture.Chunk)
                    ? [new CompilerFixedDimension(2), new CompilerFixedDimension(2)]
                    : [new CompilerFixedDimension(1), new CompilerFixedDimension(4)])),
            ]);
        var scanned = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(descriptor));
        Assert.True(scanned.IsSuccess, FormatDiagnostics(scanned.Diagnostics));
        var split = Assert.IsType<Onnxify.Split>(Assert.IsType<CompilerOnnxStep>(Assert.Single(scanned.Value!.Operations)).Node);
        Assert.Equal(["first", "second"], split.Out.Select(static output => output.Name));

        var emitted = Compiler.GenerateOnnx(scanned.Value);
        Assert.True(emitted.IsSuccess, FormatDiagnostics(emitted.Diagnostics));
        Assert.IsType<Onnxify.Split>(Assert.Single(emitted.Value!.Graph.Nodes));
        var imported = Compiler.CreateTreeFromOnnx(emitted.Value);
        Assert.True(imported.IsSuccess, FormatDiagnostics(imported.Diagnostics));
        var generated = Compiler.GenerateSourceCode(imported.Value!);
        Assert.True(generated.IsSuccess, FormatDiagnostics(generated.Diagnostics));
        Assert.Contains("torch.split", generated.SourceCode());
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

    private static OnnxModel CreateReshapeModel(long[] shape, long? allowzero = 1, long[]? outputShape = null)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([2, 3, 4]));
        var output = model.Graph.AddOutput(
            "output",
            OnnxTensorType.Create<float>((outputShape ?? shape).Select(static dimension => new OnnxDimension<long>(dimension))));
        var shapeTensor = model.Graph.AddTensor("shape", [shape.Length], shape);
        model.Graph.AddNode(new Onnxify.Reshape(
            "reshape",
            new Onnxify.ReshapeInputOutputOptions
            {
                Data = input,
                Shape = shapeTensor,
                Allowzero = allowzero,
                Reshaped = output,
            }));
        return model;
    }

    private static OnnxModel CreateUnsqueezeModel(long axis)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([2, 3, 4]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2, 1, 3, 4]));
        var axes = model.Graph.AddTensor("axes", [1], [axis]);
        model.Graph.AddNode(new Onnxify.Unsqueeze(
            "unsqueeze",
            new Onnxify.UnsqueezeInputOutputOptions
            {
                Data = input,
                Axes = axes,
                Expanded = output,
            }));
        return model;
    }

    private static OnnxModel CreateSqueezeModel(long axis)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([2, 1, 4]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2, 4]));
        var axes = model.Graph.AddTensor("axes", [1], [axis]);
        model.Graph.AddNode(new Onnxify.Squeeze(
            "squeeze",
            new Onnxify.SqueezeInputOutputOptions
            {
                Data = input,
                Axes = axes,
                Squeezed = output,
            }));
        return model;
    }

    private static OnnxModel CreateExpandModel(long[] shape)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([1, 3, 4]));
        var output = model.Graph.AddOutput(
            "output",
            OnnxTensorType.Create<float>(shape.Select(static dimension => new OnnxDimension<long>(dimension))));
        var targetShape = model.Graph.AddTensor("shape", [shape.Length], shape);
        model.Graph.AddNode(new Onnxify.Expand(
            "expand",
            new Onnxify.ExpandInputOutputOptions
            {
                Input = input,
                Shape = targetShape,
                Output = output,
            }));
        return model;
    }

    private static OnnxModel CreateTileModel(long[] repeats)
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([2, 3, 4]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2, 6, 4]));
        var repeatsTensor = model.Graph.AddTensor("repeats", [repeats.Length], repeats);
        model.Graph.AddNode(new Onnxify.Tile(
            "tile",
            new Onnxify.TileInputOutputOptions
            {
                Input = input,
                Repeats = repeatsTensor,
                Output = output,
            }));
        return model;
    }

    private static OnnxModel CreateSliceModel()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([2, 3, 4]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2, 2, 4]));
        var starts = model.Graph.AddTensor("starts", [1], [1L]);
        var ends = model.Graph.AddTensor("ends", [1], [3L]);
        var axes = model.Graph.AddTensor("axes", [1], [1L]);
        var steps = model.Graph.AddTensor("steps", [1], [1L]);
        model.Graph.AddNode(new Onnxify.Slice(
            "slice",
            new Onnxify.SliceInputOutputOptions
            {
                Data = input,
                Starts = starts,
                Ends = ends,
                Axes = axes,
                Steps = steps,
                Output = output,
            }));
        return model;
    }

    private static OnnxModel CreateConcatModel()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var left = model.Graph.AddInput("left", OnnxTensorType.Create<float>([2, 2]));
        var right = model.Graph.AddInput("right", OnnxTensorType.Create<float>([2, 2]));
        var output = model.Graph.AddOutput("output", OnnxTensorType.Create<float>([2, 4]));
        model.Graph.AddNode(new Onnxify.Concat(
            "concat",
            new Onnxify.ConcatInputOutputOptions
            {
                In = [left, right],
                Axis = 1,
                ConcatResult = output,
            }));
        return model;
    }

    private static OnnxModel CreateSplitModel()
    {
        var model = OnnxModel.Create(new OnnxModelCreationOptions { Opset = 25 });
        var input = model.Graph.AddInput("input", OnnxTensorType.Create<float>([2, 4]));
        var first = model.Graph.AddOutput("first", OnnxTensorType.Create<float>([2, 1]));
        var second = model.Graph.AddOutput("second", OnnxTensorType.Create<float>([2, 3]));
        var sizes = model.Graph.AddTensor("split_sizes", [2], [1L, 3L]);
        model.Graph.AddNode(new Onnxify.Split(
            "split",
            new Onnxify.SplitInputOutputOptions
            {
                Input = input,
                InputSplit = sizes,
                Axis = 1,
                Out = [first, second],
            }));
        return model;
    }

    private static float[] ExecuteGenerated(string source, float[] values, long[]? shape = null)
    {
        return ExecuteGenerated(source, [values], [shape ?? [2L, 3L, 4L]]);
    }

    private static float[] ExecuteGenerated(string source, float[][] values, long[][] shapes)
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
        var inputs = values
            .Select((inputValues, index) => global::TorchSharp.torch.tensor(
                inputValues,
                shapes[index],
                dtype: global::TorchSharp.torch.ScalarType.Float32))
            .ToArray();
        try
        {
            using var output = (global::TorchSharp.torch.Tensor)moduleType.GetMethod("forward")!.Invoke(module, inputs)!;
            return output.data<float>().ToArray();
        }
        finally
        {
            foreach (var input in inputs)
            {
                input.Dispose();
            }
        }
    }

    private static float[] ExecuteOnnx(OnnxModel model, float[] values, int[]? shape = null)
    {
        return ExecuteOnnx(model, [values], [shape ?? [2, 3, 4]]);
    }

    private static float[] ExecuteOnnx(OnnxModel model, float[][] values, int[][] shapes)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-flatten-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var inputNames = model.Graph.Inputs.Select(static input => input.Name).ToArray();
            using var results = session.Run(values.Select((inputValues, index) => NamedOnnxValue.CreateFromTensor(
                inputNames[index],
                new DenseTensor<float>(inputValues, shapes[index]))).ToArray());
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

    private static float[][] ExecuteOnnxOutputs(OnnxModel model, float[][] values, int[][] shapes)
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-split-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new InferenceSession(path);
            var inputNames = model.Graph.Inputs.Select(static input => input.Name).ToArray();
            using var results = session.Run(values.Select((inputValues, index) => NamedOnnxValue.CreateFromTensor(
                inputNames[index],
                new DenseTensor<float>(inputValues, shapes[index]))).ToArray());
            return model.Graph.Outputs
                .Select(outputName => results.Single(result => result.Name == outputName.Name).AsTensor<float>().ToArray())
                .ToArray();
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static float[][] ExecuteGeneratedOutputs(string source, float[][] values, long[][] shapes)
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
        using var module = (IDisposable)Activator.CreateInstance(moduleType, "split")!;
        var inputs = values
            .Select((inputValues, index) => global::TorchSharp.torch.tensor(
                inputValues,
                shapes[index],
                dtype: global::TorchSharp.torch.ScalarType.Float32))
            .ToArray();
        try
        {
            var result = moduleType.GetMethod("forward")!.Invoke(module, inputs)!;
            var resultType = result.GetType();
            return [
                ((global::TorchSharp.torch.Tensor)resultType.GetField("Item1")!.GetValue(result)!).data<float>().ToArray(),
                ((global::TorchSharp.torch.Tensor)resultType.GetField("Item2")!.GetValue(result)!).data<float>().ToArray(),
            ];
        }
        finally
        {
            foreach (var input in inputs)
            {
                input.Dispose();
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

    private sealed class SqueezeFixture
    {
        public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
        {
            return input.squeeze(1);
        }
    }

    private sealed class UnflattenFixture
    {
        public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
        {
            return input.unflatten(1, new long[] { 2, 2 });
        }
    }

    private sealed class SplitFormsFixture
    {
        public global::TorchSharp.torch.Tensor[] Chunk(global::TorchSharp.torch.Tensor input)
        {
            return input.chunk(2, 1);
        }

        public global::TorchSharp.torch.Tensor[] Unbind(global::TorchSharp.torch.Tensor input)
        {
            return input.unbind(0);
        }
    }
}
