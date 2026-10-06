using Onnxify.Compiler;
using Onnxify.TorchSharp;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using System.Runtime.Loader;

namespace Onnxify.Compiler.Tests;

public sealed class CSharpCompilerTests
{
    [Fact]
    public void SourceTextIsScannedIntoImmutableSyntaxRepresentation()
    {
        var source = new CSharpTorchSharpSource(
            """
            public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
            {
                var values = new[] { 1, 2 };
                var selected = values[0];
                if (true)
                {
                    selected = selected + 1;
                }
                foreach (var item in values)
                {
                    selected = selected + item;
                }
                return input;
            }
            """);

        var first = Compiler.CreateTreeFromTorchSharp(source);
        var second = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(source.SourceText));

        Assert.True(first.IsSuccess, string.Join(Environment.NewLine, first.Diagnostics.Select(x => x.Message)));
        Assert.True(second.IsSuccess, string.Join(Environment.NewLine, second.Diagnostics.Select(x => x.Message)));
        Assert.Equal(first.Value, second.Value);
        Assert.NotNull(first.Value!.SyntaxBody);
        Assert.Equal(2, first.Value.IntermediateValues.Count);
        Assert.Contains(first.Value.SyntaxBody!.Statements, x => x is CompilerStaticIfStatement);
        Assert.Contains(first.Value.SyntaxBody.Statements, x => x is CompilerStaticForeachStatement);
    }

    [Fact]
    public void BackendGeneratesTorchSharpSourceFromSyntaxTree()
    {
        var treeResult = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource("return input + 1f;"));

        Assert.True(treeResult.IsSuccess);
        var generated = Compiler.GenerateCSharp(
            treeResult.Value!,
            new CompilerCSharpGenerationOptions
            {
                Namespace = "Generated.Tests",
                ClassName = "IdentityLikeModule",
                ModuleName = "IdentityLikeModule",
            });

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("torch.nn.Module", generated.Value);
        Assert.Contains("public override", generated.Value);
        Assert.Contains("return (input + 1f);", generated.Value);
    }

    [Fact]
    public void TorchSharpReluLowersThroughSharedMappingToOnnx()
    {
        // Source: third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py (nn.functional.relu).
        // Runtime semantics: third_party/onnxruntime/onnxruntime/test/providers/cpu/activation/activation_op_test.cc (Relu).
        var imported = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource("return torch.nn.functional.relu(input);"));

        Assert.True(imported.IsSuccess, string.Join(" | ", imported.Diagnostics.Select(x => x.Message)));
        var operation = Assert.IsType<CompilerOperation>(imported.Value!.Operations.Single());
        Assert.Equal(CompilerOperationCapability.Bidirectional, operation.Descriptor.Capability);
        Assert.Equal("Relu", operation.Descriptor.Name);
        Assert.Null(imported.Value.SyntaxBody);

        var generated = Compiler.GenerateCSharp(imported.Value);
        Assert.True(generated.IsSuccess, string.Join(" | ", generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("torch.nn.functional.relu(input)", generated.Value);

        var emitted = Compiler.GenerateOnnx(imported.Value);
        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        Assert.Equal("Relu", emitted.Value!.Graph.Nodes.Single().OpType);

        var path = Path.Combine(Path.GetTempPath(), $"onnxify-compiler-relu-source-{Guid.NewGuid():N}.onnx");
        var inputValues = new[] { -3f, -1f, 0.25f, 2f };
        try
        {
            emitted.Value.Save(path, overwrite: true);
            using var session = new global::Microsoft.ML.OnnxRuntime.InferenceSession(path);
            using var results = session.Run(
            [
                global::Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
                    "input",
                    new global::Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(inputValues, [4])),
            ]);
            var runtimeValues = results.Single().AsTensor<float>().ToArray();

            using var torchInput = global::TorchSharp.torch.tensor(
                inputValues,
                [4L],
                dtype: global::TorchSharp.torch.ScalarType.Float32);
            using var torchOutput = global::TorchSharp.torch.nn.functional.relu(torchInput);
            Assert.Equal(torchOutput.data<float>().ToArray(), runtimeValues);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public void UnsupportedReluOverloadReportsAnAnalyzeDiagnostic()
    {
        var imported = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource("return torch.nn.functional.relu(input, input);"));

        Assert.False(imported.IsSuccess);
        var diagnostic = Assert.Single(imported.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Equal(CompilerDiagnosticStage.Analyze, diagnostic.Stage);
        Assert.Equal(CompilerDiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("one tensor input", diagnostic.Message);
    }

    [Fact]
    public void GeneratedSourceIsCompilableCSharp()
    {
        var treeResult = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource("return input;"));
        Assert.True(treeResult.IsSuccess);

        var generated = Compiler.GenerateCSharp(treeResult.Value!);
        Assert.True(generated.IsSuccess);

        var compilation = CSharpCompilation.Create(
            "Onnxify.Generated.CompilerTest",
            [CSharpSyntaxTree.ParseText(generated.Value!)],
            GetCompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var output = new MemoryStream();
        var emit = compilation.Emit(output);

        Assert.True(
            emit.Success,
            string.Join(Environment.NewLine, emit.Diagnostics.Where(x => x.Severity == DiagnosticSeverity.Error)));
    }

    [Fact]
    public void GeneratedSourceReconstructsAMinimalExecutableModule()
    {
        var treeResult = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource("return input;"));
        Assert.True(treeResult.IsSuccess);
        var generated = Compiler.GenerateCSharp(treeResult.Value!);
        Assert.True(generated.IsSuccess);

        var compilation = CSharpCompilation.Create(
            "Onnxify.Generated.CompilerExecutionTest",
            [CSharpSyntaxTree.ParseText(generated.Value!)],
            GetCompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(
            emit.Success,
            string.Join(Environment.NewLine, emit.Diagnostics.Where(x => x.Severity == DiagnosticSeverity.Error)));

        assemblyBytes.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(assemblyBytes);
        var moduleType = assembly.GetType("Onnxify.Generated.GeneratedTorchModule");
        Assert.NotNull(moduleType);
        using var module = (IDisposable)Activator.CreateInstance(moduleType!, "generated")!;
        using var input = global::TorchSharp.torch.tensor(
            new float[] { 1f, 2f },
            new long[] { 2L });
        var output = (global::TorchSharp.torch.Tensor)moduleType!
            .GetMethod("forward")!
            .Invoke(module, [input])!;
        using (output)
        {
            Assert.Equal(input.shape.ToArray(), output.shape.ToArray());
        }
    }

    [Fact]
    public void HelperMethodsLowerToBlocksAndOnnxCalls()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    return Project(input);
                }

                private global::TorchSharp.torch.Tensor Project(global::TorchSharp.torch.Tensor value)
                {
                    return value;
                }
                """));

        Assert.True(result.IsSuccess, string.Join(Environment.NewLine, result.Diagnostics.Select(x => x.Message)));
        var tree = Assert.IsType<CompilerComputationTree>(result.Value);
        var helper = Assert.Single(tree.Blocks);
        Assert.Equal("Project", helper.Name);
        var moduleCall = Assert.IsType<CompilerModuleCall>(Assert.Single(tree.Operations));
        Assert.Equal("Project", moduleCall.TargetBlock);
        Assert.Null(tree.SyntaxBody);

        var generated = Compiler.GenerateCSharp(tree);

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("private global::TorchSharp.torch.Tensor Project", generated.Value);
        Assert.Contains("var output = Project(input);", generated.Value);

        var emitted = Compiler.GenerateOnnx(tree);
        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        Assert.Equal("Identity", Assert.Single(emitted.Value!.Graph.Nodes).OpType);

        var path = Path.Combine(Path.GetTempPath(), $"onnxify-compiler-helper-{Guid.NewGuid():N}.onnx");
        var inputValues = new[] { -1.5f, 0.25f, 2f };
        try
        {
            emitted.Value.Save(path, overwrite: true);
            using var session = new global::Microsoft.ML.OnnxRuntime.InferenceSession(path);
            using var results = session.Run(
            [
                global::Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
                    "input",
                    new global::Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(inputValues, [3])),
            ]);

            Assert.Equal(inputValues, results.Single().AsTensor<float>().ToArray());
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public void NestedHelperCallsInlineToOnnxAndPreserveMappingSemantics()
    {
        // The helper tree is intentionally composed only from the already verified ReLU mapping (OXY-024).
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    return Outer(input);
                }

                private global::TorchSharp.torch.Tensor Outer(global::TorchSharp.torch.Tensor value)
                {
                    return Inner(value);
                }

                private global::TorchSharp.torch.Tensor Inner(global::TorchSharp.torch.Tensor value)
                {
                    return torch.nn.functional.relu(value);
                }
                """));

        Assert.True(result.IsSuccess, string.Join(" | ", result.Diagnostics.Select(x => x.Message)));
        var tree = Assert.IsType<CompilerComputationTree>(result.Value);
        Assert.Equal(["Outer", "Inner"], tree.Blocks.Select(block => block.Name));
        Assert.Null(tree.SyntaxBody);
        Assert.Equal("Outer", Assert.IsType<CompilerModuleCall>(Assert.Single(tree.Operations)).TargetBlock);

        var emitted = Compiler.GenerateOnnx(tree);
        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        Assert.Equal("Relu", Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        AssertOnnxOutput(emitted.Value, [-1f, 0.5f, 2f], [0f, 0.5f, 2f]);
    }

    [Fact]
    public void HelperScalarArgumentBindsToActivationAttribute()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    return ApplySlope(input, 0.25f);
                }

                private global::TorchSharp.torch.Tensor ApplySlope(global::TorchSharp.torch.Tensor value, float slope)
                {
                    return torch.nn.functional.leaky_relu(value, slope);
                }
                """));

        Assert.True(result.IsSuccess, string.Join(" | ", result.Diagnostics.Select(x => x.Message)));
        var tree = Assert.IsType<CompilerComputationTree>(result.Value);
        var moduleCall = Assert.IsType<CompilerModuleCall>(Assert.Single(tree.Operations));
        Assert.Equal(2, moduleCall.Arguments.Count);
        Assert.IsType<CompilerLiteralExpression>(moduleCall.Arguments[1]);

        var generated = Compiler.GenerateCSharp(tree);
        Assert.True(generated.IsSuccess, string.Join(" | ", generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("ApplySlope(input, 0.25f)", generated.Value);
        Assert.Contains("float slope", generated.Value);

        var emitted = Compiler.GenerateOnnx(tree);
        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        var node = Assert.Single(emitted.Value!.Graph.Nodes);
        Assert.Equal("LeakyRelu", node.OpType);
        Assert.Equal(0.25f, Assert.IsType<OnnxAttribute<float>>(node.Attributes.Single()).Value);
        AssertOnnxOutput(emitted.Value, [-2f, 0.5f, 2f], [-0.5f, 0.5f, 2f]);
    }

    [Fact]
    public void TupleDeconstructionLowersHelperReturnsInOrder()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public (global::TorchSharp.torch.Tensor, global::TorchSharp.torch.Tensor) forward(global::TorchSharp.torch.Tensor input)
                {
                    var (first, second) = Split(input);
                    return (first, second);
                }

                private (global::TorchSharp.torch.Tensor, global::TorchSharp.torch.Tensor) Split(global::TorchSharp.torch.Tensor value)
                {
                    return (value, value);
                }
                """));

        Assert.True(result.IsSuccess, string.Join(" | ", result.Diagnostics.Select(x => x.Message)));
        var tree = Assert.IsType<CompilerComputationTree>(result.Value);
        Assert.Null(tree.SyntaxBody);
        var call = Assert.IsType<CompilerModuleCall>(Assert.Single(tree.Operations));
        Assert.Equal("Split", call.TargetBlock);
        Assert.Equal(["output0", "output1"], call.Outputs.Select(output => output.Name));

        var generated = Compiler.GenerateCSharp(tree);
        Assert.True(generated.IsSuccess, string.Join(" | ", generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("var (output0, output1) = Split(input);", generated.Value);

        var emitted = Compiler.GenerateOnnx(tree);
        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        Assert.Equal(["Identity", "Identity"], emitted.Value!.Graph.Nodes.Select(node => node.OpType));

        var path = Path.Combine(Path.GetTempPath(), $"onnxify-compiler-tuple-{Guid.NewGuid():N}.onnx");
        var inputValues = new[] { -1f, 0.5f, 2f };
        try
        {
            emitted.Value.Save(path, overwrite: true);
            using var session = new global::Microsoft.ML.OnnxRuntime.InferenceSession(path);
            using var results = session.Run(
            [
                global::Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
                    "input",
                    new global::Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(inputValues, [inputValues.Length])),
            ]);

            Assert.Equal(["output0", "output1"], results.Select(output => output.Name).OrderBy(name => name));
            foreach (var output in results)
            {
                Assert.Equal(inputValues, output.AsTensor<float>().ToArray());
            }
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public void StaticallySelectedIfBranchLowersThroughSharedMapping()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    if (true)
                    {
                        return torch.nn.functional.relu(input);
                    }
                    else
                    {
                        return torch.nn.functional.tanh(input);
                    }
                }
                """));

        Assert.True(result.IsSuccess, string.Join(" | ", result.Diagnostics.Select(x => x.Message)));
        var tree = Assert.IsType<CompilerComputationTree>(result.Value);
        Assert.Null(tree.SyntaxBody);
        Assert.Equal("Relu", Assert.IsType<CompilerOperation>(Assert.Single(tree.Operations)).Descriptor.Name);

        var emitted = Compiler.GenerateOnnx(tree);
        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        Assert.Equal("Relu", Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        AssertOnnxOutput(emitted.Value, [-1f, 0.5f, 2f], [0f, 0.5f, 2f]);
    }

    [Fact]
    public void NoGradScopeIsTransparentWhenLoweringToOnnx()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    using (torch.no_grad())
                    {
                        return torch.nn.functional.relu(input);
                    }
                }
                """));

        Assert.True(result.IsSuccess, string.Join(" | ", result.Diagnostics.Select(x => x.Message)));
        var tree = Assert.IsType<CompilerComputationTree>(result.Value);
        Assert.Null(tree.SyntaxBody);
        Assert.Equal("Relu", Assert.IsType<CompilerOperation>(Assert.Single(tree.Operations)).Descriptor.Name);

        var emitted = Compiler.GenerateOnnx(tree);
        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        Assert.Equal("Relu", Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        AssertOnnxOutput(emitted.Value, [-1f, 0.5f, 2f], [0f, 0.5f, 2f]);
    }

    [Fact]
    public void LiteralArrayForeachIsPreservedForTorchSharpGeneration()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    var values = new[] { 1f, 2f };
                    var output = input;
                    foreach (var value in values)
                    {
                        output = output + value;
                    }
                    return output;
                }
                """));

        Assert.True(result.IsSuccess, string.Join(" | ", result.Diagnostics.Select(x => x.Message)));
        var tree = Assert.IsType<CompilerComputationTree>(result.Value);
        Assert.Contains(tree.SyntaxBody!.Statements, statement => statement is CompilerStaticForeachStatement);

        var generated = Compiler.GenerateCSharp(tree);
        Assert.True(generated.IsSuccess, string.Join(" | ", generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("foreach (var value in values)", generated.Value);
    }

    private static void AssertOnnxOutput(
        OnnxModel model,
        float[] inputValues,
        float[] expectedValues
    )
    {
        var path = Path.Combine(Path.GetTempPath(), $"onnxify-compiler-static-{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var session = new global::Microsoft.ML.OnnxRuntime.InferenceSession(path);
            using var results = session.Run(
            [
                global::Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
                    "input",
                    new global::Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(inputValues, [inputValues.Length])),
            ]);
            var actualValues = results.Single().AsTensor<float>().ToArray();
            Assert.Equal(expectedValues, actualValues);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public void ModuleCallsAreEmittedThroughReusableBlocks()
    {
        var builder = new CompilerComputationTreeBuilder("module-call");
        builder.AddInput(new CompilerValue("input", new CompilerTensorType(CompilerElementType.Float32, null)));
        builder.AddOutput(new CompilerValue("output", new CompilerTensorType(CompilerElementType.Float32, null)));
        builder.AddBlock(
            new CompilerComputationBlock(
                "Project",
                [new CompilerValueReference("value")],
                [new CompilerValueReference("result")],
                new CompilerBlockStatement(
                [
                    new CompilerReturnStatement(new CompilerReferenceExpression("value")),
                ])));
        builder.AddOperation(
            new CompilerModuleCall(
                "project-call",
                "Project",
                [new CompilerValueReference("input")],
                [new CompilerValueReference("output")]));

        var generated = Compiler.GenerateCSharp(builder.Build());

        Assert.True(generated.IsSuccess, string.Join(Environment.NewLine, generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("var output = Project(input);", generated.Value);
        Assert.Contains("return output;", generated.Value);
    }

    [Fact]
    public void DynamicIfIsReportedAsAnAnalyzeErrorWithCSharpSpan()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    if (input is not null)
                    {
                        return input;
                    }
                    return input;
                }
                """));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Equal(CompilerDiagnosticStage.Analyze, diagnostic.Stage);
        Assert.Equal(CompilerDiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal(CompilerSourceSpanKind.CSharp, diagnostic.Span!.Kind);
        Assert.NotNull(diagnostic.Context);
    }

    [Fact]
    public void DynamicForeachCollectionIsReportedAsAnAnalyzeError()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    foreach (var item in GetValues(input))
                    {
                        input = input;
                    }
                    return input;
                }
                """));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Equal(CompilerDiagnosticStage.Analyze, diagnostic.Stage);
        Assert.Contains("compile-time array", diagnostic.Message);
        Assert.NotNull(diagnostic.Span);
    }

    [Fact]
    public void RecursiveCallsAreReportedWithCallerAndSourceSpan()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    return forward(input);
                }
                """));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Equal(CompilerDiagnosticStage.Analyze, diagnostic.Stage);
        Assert.Equal("forward", diagnostic.Context!.Caller);
        Assert.NotNull(diagnostic.Span);
    }

    [Fact]
    public void DynamicModuleForwardDispatchIsReportedExplicitly()
    {
        var result = Compiler.CreateTreeFromTorchSharp(
            new CSharpTorchSharpSource(
                """
                public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
                {
                    return child.forward(input);
                }
                """));

        Assert.False(result.IsSuccess);
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(CompilerDiagnosticCodes.Unsupported, diagnostic.Code);
        Assert.Equal(CompilerDiagnosticStage.Analyze, diagnostic.Stage);
        Assert.Contains("Dynamic module forward dispatch", diagnostic.Message);
        Assert.NotNull(diagnostic.Span);
        Assert.Equal("forward", diagnostic.Context!.Caller);
    }

    [Fact]
    public void CompilerDescriptorIsImmutableAndStructurallyComparable()
    {
        var descriptor = new CompilerTorchSharpModuleDescriptor(
            typeof(CSharpCompilerTests).Assembly.Location,
            typeof(CSharpCompilerTests).FullName!,
            inputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "input",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerSymbolicDimension("batch")])),
            ],
            outputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "output",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerSymbolicDimension("batch")])),
            ],
            stateMembers:
            [
                new CompilerTorchSharpStateMemberDescriptor(
                    "threshold",
                    CompilerStateMemberKind.Initializer,
                    new CompilerScalarType(CompilerElementType.Float32),
                    new CompilerFloatingPointLiteral(CompilerElementType.Float32, 1.5)),
            ]);

        var equivalent = new CompilerTorchSharpModuleDescriptor(
            typeof(CSharpCompilerTests).Assembly.Location,
            typeof(CSharpCompilerTests).FullName!,
            inputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "input",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerSymbolicDimension("batch")])),
            ],
            outputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "output",
                    new CompilerTensorType(
                        CompilerElementType.Float32,
                        [new CompilerSymbolicDimension("batch")])),
            ],
            stateMembers:
            [
                new CompilerTorchSharpStateMemberDescriptor(
                    "threshold",
                    CompilerStateMemberKind.Initializer,
                    new CompilerScalarType(CompilerElementType.Float32),
                    new CompilerFloatingPointLiteral(CompilerElementType.Float32, 1.5)),
            ]);

        Assert.Equal(descriptor, equivalent);
        Assert.Equal(descriptor.GetHashCode(), equivalent.GetHashCode());
    }

    [Fact]
    public void CompiledModuleDescriptorCanBeDecompiledWithoutTorchSharpInCompilerApi()
    {
        var descriptor = new CompilerTorchSharpModuleDescriptor(
            typeof(DescriptorFixture).Assembly.Location,
            typeof(DescriptorFixture).FullName!,
            methodName: nameof(DescriptorFixture.forward),
            methodMetadataToken: typeof(DescriptorFixture).GetMethod(nameof(DescriptorFixture.forward))!.MetadataToken,
            inputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "input",
                    new CompilerTensorType(CompilerElementType.Float32, null)),
            ],
            outputs:
            [
                new CompilerTorchSharpValueDescriptor(
                    "output",
                    new CompilerTensorType(CompilerElementType.Float32, null)),
            ]);

        var result = Compiler.CreateTreeFromTorchSharp(new CSharpTorchSharpSource(descriptor));

        Assert.True(result.IsSuccess, string.Join(Environment.NewLine, result.Diagnostics.Select(x => x.Message)));
        Assert.NotNull(result.Value!.SyntaxBody);
        Assert.Contains(result.Value.SyntaxBody!.Statements, x => x is CompilerReturnStatement);
    }

    [Fact]
    public void TorchSharpAdapterExposesACompilerNeutralDescriptor()
    {
        using var module = new AdapterFixture();

        var source = TorchSharpCompilerAdapter.CreateSource(module);

        Assert.Equal(CompilerSourceKind.CSharpTorchSharp, source.Kind);
        Assert.NotNull(source.Module);
        Assert.Equal(typeof(AdapterFixture).Assembly.Location, source.Module!.AssemblyPath);
        Assert.Equal(nameof(AdapterFixture.forward), source.Module.MethodName);

        var result = Compiler.CreateTreeFromTorchSharp(source);

        Assert.True(result.IsSuccess, string.Join(Environment.NewLine, result.Diagnostics.Select(x => x.Message)));
        Assert.Single(result.Value!.Inputs);
    }

    [Fact]
    public void TorchSharpAdapterAndCompilerFlattenNestedUserModules()
    {
        using var module = new ParentAdapterFixture();

        var source = TorchSharpCompilerAdapter.CreateSource(module);
        var child = Assert.Single(source.Module!.ChildModules);
        Assert.NotNull(child.Module);
        Assert.Single(child.Module!.ChildModules);
        Assert.NotNull(child.Module.ChildModules[0].Module);

        var imported = Compiler.CreateTreeFromTorchSharp(source);
        Assert.True(imported.IsSuccess, string.Join(" | ", imported.Diagnostics.Select(x => x.Message)));
        var tree = Assert.IsType<CompilerComputationTree>(imported.Value);
        Assert.Contains(tree.Blocks, block => block.Name == "_child__forward");
        Assert.Contains(tree.Blocks, block => block.Name == "_child___inner__forward");
        var childCall = Assert.IsType<CompilerModuleCall>(Assert.Single(tree.Operations));
        Assert.Equal("_child__forward", childCall.TargetBlock);

        var generated = Compiler.GenerateCSharp(tree);
        Assert.True(generated.IsSuccess, string.Join(" | ", generated.Diagnostics.Select(x => x.Message)));
        Assert.Contains("_child__forward(input)", generated.Value);
        Assert.Contains("_child___inner__forward(input)", generated.Value);

        var emitted = Compiler.GenerateOnnx(tree);
        Assert.True(emitted.IsSuccess, string.Join(" | ", emitted.Diagnostics.Select(x => x.Message)));
        Assert.Equal("Relu", Assert.Single(emitted.Value!.Graph.Nodes).OpType);
        AssertOnnxOutput(emitted.Value, [-1f, 0.5f, 2f], [0f, 0.5f, 2f]);
    }

    private sealed class DescriptorFixture
    {
        public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input)
        {
            return input;
        }
    }

    private sealed class AdapterFixture
        : global::TorchSharp.torch.nn.Module<
            global::TorchSharp.torch.Tensor,
            global::TorchSharp.torch.Tensor>
    {
        public AdapterFixture()
            : base(nameof(AdapterFixture))
        {
        }

        public override global::TorchSharp.torch.Tensor forward(
            global::TorchSharp.torch.Tensor input)
        {
            return input;
        }
    }

    private sealed class ParentAdapterFixture
        : global::TorchSharp.torch.nn.Module<
            global::TorchSharp.torch.Tensor,
            global::TorchSharp.torch.Tensor>
    {
        private readonly ChildAdapterFixture _child;

        public ParentAdapterFixture()
            : base(nameof(ParentAdapterFixture))
        {
            _child = new ChildAdapterFixture();
            RegisterComponents();
        }

        public override global::TorchSharp.torch.Tensor forward(
            global::TorchSharp.torch.Tensor input
        )
        {
            return _child.forward(input);
        }
    }

    private sealed class ChildAdapterFixture
        : global::TorchSharp.torch.nn.Module<
            global::TorchSharp.torch.Tensor,
            global::TorchSharp.torch.Tensor>
    {
        private readonly GrandchildAdapterFixture _inner;

        public ChildAdapterFixture()
            : base(nameof(ChildAdapterFixture))
        {
            _inner = new GrandchildAdapterFixture();
            RegisterComponents();
        }

        public override global::TorchSharp.torch.Tensor forward(
            global::TorchSharp.torch.Tensor input
        )
        {
            return _inner.forward(input);
        }
    }

    private sealed class GrandchildAdapterFixture
        : global::TorchSharp.torch.nn.Module<
            global::TorchSharp.torch.Tensor,
            global::TorchSharp.torch.Tensor>
    {
        public GrandchildAdapterFixture()
            : base(nameof(GrandchildAdapterFixture))
        {
        }

        public override global::TorchSharp.torch.Tensor forward(
            global::TorchSharp.torch.Tensor input
        )
        {
            return global::TorchSharp.torch.nn.functional.relu(input);
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
}
