using Onnxify;
using Onnxify.Compiler;
using Onnxify.ModelGenerator;
using Onnxify.TorchSharp;

namespace Onnxify.Compiler.Tests;

public sealed class CompilerBoundaryTests
{
    [Fact]
    public void Onnx_source_preserves_core_model_and_source_kind()
    {
        var model = OnnxModel.Create();
        var source = new OnnxCompilerSource(model);

        Assert.Same(model, source.Model);
        Assert.Equal(CompilerSourceKind.Onnx, source.Kind);
    }

    [Fact]
    public void CSharp_source_preserves_source_text_and_source_kind()
    {
        const string sourceText = "return input;";
        var source = new CSharpTorchSharpSource(sourceText);

        Assert.Equal(sourceText, source.SourceText);
        Assert.Equal(CompilerSourceKind.CSharpTorchSharp, source.Kind);
    }

    [Fact]
    public void Compiler_sinks_report_typed_target_kinds()
    {
        ICompilerSink<OnnxGraph> onnxSink = new OnnxCompilerSink();
        ICompilerSink<string> csharpSink = new CSharpCompilerSink();

        Assert.Equal(CompilerTargetKind.OnnxGraph, onnxSink.Kind);
        Assert.Equal(CompilerTargetKind.CSharp, csharpSink.Kind);
    }

    [Fact]
    public void Compiler_session_contract_returns_diagnostic_results()
    {
        ICompilerSession session = new TestCompilerSession();

        var treeResult = session.CreateTree(new CSharpTorchSharpSource("return input;"));
        var outputResult = session.Generate(treeResult.Value!, new CSharpCompilerSink());

        Assert.True(treeResult.IsSuccess);
        Assert.Empty(treeResult.Diagnostics);
        Assert.True(outputResult.IsSuccess);
        Assert.Equal("generated", outputResult.Value);
    }

    [Fact]
    public void Compiler_session_contract_supports_onnx_output()
    {
        ICompilerSession session = new TestCompilerSession();

        var treeResult = session.CreateTree(new OnnxCompilerSource(OnnxModel.Create()));
        var outputResult = session.Generate(treeResult.Value!, new OnnxCompilerSink());

        Assert.True(outputResult.IsSuccess);
        Assert.NotNull(outputResult.Value);
    }

    [Fact]
    public void Consumer_projects_can_use_the_compiler_boundary()
    {
        ICompilerSource onnxSource = new OnnxCompilerSource(OnnxModel.Create());
        ICompilerSource torchSource = new CSharpTorchSharpSource("return input;");

        Assert.Equal(CompilerSourceKind.Onnx, onnxSource.Kind);
        Assert.Equal(CompilerSourceKind.CSharpTorchSharp, torchSource.Kind);
    }

    [Fact]
    public void Onnx_source_rejects_null_model()
    {
        Assert.Throws<ArgumentNullException>(() => new OnnxCompilerSource(null!));
    }

    [Fact]
    public void CSharp_source_rejects_empty_source_text()
    {
        Assert.Throws<ArgumentException>(() => new CSharpTorchSharpSource(string.Empty));
    }

    [Fact]
    public void Representative_computation_tree_has_deep_structural_equality()
    {
        var first = CreateRepresentativeTree();
        var second = CreateRepresentativeTree();

        Assert.Equal(first, second);
        Assert.Equal(first.GetHashCode(), second.GetHashCode());
        Assert.Equal(CompilerElementType.Float32, Assert.IsType<CompilerTensorType>(first.Inputs[0].Type).ElementType);
        Assert.Equal(CompilerDimensionKind.Symbolic, Assert.IsType<CompilerSymbolicDimension>(Assert.IsType<CompilerTensorType>(first.Inputs[0].Type).Dimensions![1]).Kind);
        Assert.Equal(CompilerDimensionKind.Unknown, Assert.IsType<CompilerUnknownDimension>(Assert.IsType<CompilerTensorType>(first.Inputs[0].Type).Dimensions![2]).Kind);
        Assert.Single(first.Parameters);
        Assert.Single(first.Buffers);
        Assert.Single(first.Initializers);
        Assert.Equal(2, first.Operations.Count);
        Assert.NotNull(first.SyntaxBody);
    }

    [Fact]
    public void Compiler_types_and_literals_preserve_supported_shapes()
    {
        var dimensions = new CompilerDimension[]
        {
            new CompilerFixedDimension(2),
            new CompilerSymbolicDimension("batch"),
            new CompilerUnknownDimension("runtime"),
        };
        var tensorType = new CompilerTensorType(CompilerElementType.Float32, dimensions, "activation");
        var equivalentTensorType = new CompilerTensorType(
            CompilerElementType.Float32,
            [
                new CompilerFixedDimension(2),
                new CompilerSymbolicDimension("batch"),
                new CompilerUnknownDimension("runtime"),
            ],
            "activation");

        Assert.Equal(equivalentTensorType, tensorType);
        Assert.Null(new CompilerTensorType(CompilerElementType.Float32, null).Dimensions);
        Assert.Empty(new CompilerTensorType(CompilerElementType.Float32, []).Dimensions!);

        var supportedTypes = new CompilerType[]
        {
            new CompilerScalarType(CompilerElementType.Int32),
            tensorType,
            new CompilerOptionalType(tensorType),
            new CompilerSequenceType(tensorType),
            new CompilerTupleType([new CompilerScalarType(CompilerElementType.Int32), tensorType]),
            new CompilerMapType(CompilerElementType.String, tensorType),
            new CompilerSparseTensorType(CompilerElementType.Float32, dimensions),
            new CompilerOpaqueType("custom", "resource"),
        };

        Assert.Equal(8, supportedTypes.Length);
        Assert.Equal(CompilerElementType.String, Assert.IsType<CompilerMapType>(supportedTypes[5]).KeyType);

        var tensorLiteral = new CompilerTensorLiteral(
            CompilerElementType.Float32,
            dimensions,
            [new CompilerFloatingPointLiteral(CompilerElementType.Float32, 1.5)],
            new CompilerExternalTensorData("weights.bin", offset: 4, length: 8, checksum: "sha256:abc"));
        var equivalentTensorLiteral = new CompilerTensorLiteral(
            CompilerElementType.Float32,
            [
                new CompilerFixedDimension(2),
                new CompilerSymbolicDimension("batch"),
                new CompilerUnknownDimension("runtime"),
            ],
            [new CompilerFloatingPointLiteral(CompilerElementType.Float32, 1.5)],
            new CompilerExternalTensorData("weights.bin", offset: 4, length: 8, checksum: "sha256:abc"));
        var literals = new CompilerLiteral[]
        {
            new CompilerBooleanLiteral(true),
            new CompilerSignedIntegerLiteral(CompilerElementType.Int64, -1),
            new CompilerUnsignedIntegerLiteral(CompilerElementType.UInt64, 1),
            new CompilerFloatingPointLiteral(CompilerElementType.Float64, 1.5),
            new CompilerComplexLiteral(CompilerElementType.Complex128, 1, -2),
            new CompilerStringLiteral("value"),
            tensorLiteral,
            new CompilerArrayLiteral([new CompilerStringLiteral("a"), new CompilerStringLiteral("b")]),
            new CompilerTupleLiteral([new CompilerBooleanLiteral(false), new CompilerSignedIntegerLiteral(CompilerElementType.Int32, 2)]),
        };

        Assert.Equal(equivalentTensorLiteral, tensorLiteral);
        Assert.Equal(9, literals.Length);
        Assert.Equal("alpha", new CompilerAttribute("alpha", literals[0]).Name);
    }

    [Fact]
    public void Compiler_syntax_tree_preserves_ordered_statements_and_structural_equality()
    {
        var body = CreateSyntaxBody();
        var equivalent = CreateSyntaxBody();

        Assert.Equal(body, equivalent);
        Assert.Equal(body.GetHashCode(), equivalent.GetHashCode());
        Assert.Equal("values", Assert.IsType<CompilerDeclarationStatement>(body.Statements[0]).Name);
        Assert.IsType<CompilerStaticIfStatement>(body.Statements[1]);
        Assert.IsType<CompilerStaticForeachStatement>(body.Statements[2]);
    }

    [Fact]
    public void Operator_descriptors_expose_all_capability_classifications()
    {
        Assert.Equal(0, (int)CompilerOperationCapability.Unsupported);
        Assert.Equal(1, (int)CompilerOperationCapability.Bidirectional);
        Assert.Equal(2, (int)CompilerOperationCapability.ExportOnly);
        Assert.Equal(3, (int)CompilerOperationCapability.ImportOnly);
        Assert.Equal(4, Enum.GetValues<CompilerOperationCapability>().Length);
    }

    [Fact]
    public void Diagnostics_preserve_stage_severity_span_and_context()
    {
        var diagnostic = CreateUnsupportedDiagnostic();
        var copy = CreateUnsupportedDiagnostic();

        Assert.Equal(copy, diagnostic);
        Assert.Equal(CompilerDiagnosticStage.Analyze, diagnostic.Stage);
        Assert.Equal(CompilerDiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal("forward", diagnostic.Context!.Caller);
        Assert.Equal("torch.add", diagnostic.Context.Callee);
        Assert.Equal(CompilerSourceSpanKind.CSharp, diagnostic.Span!.Kind);
    }

    [Fact]
    public void Diagnostics_expose_unsupported_ambiguous_and_lossy_mapping_codes()
    {
        var diagnostics = new[]
        {
            new CompilerDiagnostic(CompilerDiagnosticCodes.Unsupported, "Unsupported", CompilerDiagnosticStage.Analyze, CompilerDiagnosticSeverity.Error),
            new CompilerDiagnostic(CompilerDiagnosticCodes.Ambiguous, "Ambiguous", CompilerDiagnosticStage.Analyze, CompilerDiagnosticSeverity.Error),
            new CompilerDiagnostic(CompilerDiagnosticCodes.Lossy, "Lossy", CompilerDiagnosticStage.Normalize, CompilerDiagnosticSeverity.Warning),
        };

        var result = new CompilerResult<string>("value", diagnostics);

        Assert.Equal(
            [CompilerDiagnosticCodes.Unsupported, CompilerDiagnosticCodes.Ambiguous, CompilerDiagnosticCodes.Lossy],
            result.Diagnostics.Select(static diagnostic => diagnostic.Code));
        Assert.True(result.HasErrors);
        Assert.False(result.IsSuccess);
    }

    [Fact]
    public void Compiler_result_distinguishes_warning_only_from_errors()
    {
        var warning = new CompilerDiagnostic(
            CompilerDiagnosticCodes.Lossy,
            "A conversion loses metadata.",
            CompilerDiagnosticStage.Normalize,
            CompilerDiagnosticSeverity.Warning);
        var error = new CompilerDiagnostic(
            CompilerDiagnosticCodes.Ambiguous,
            "The mapping is ambiguous.",
            CompilerDiagnosticStage.Analyze,
            CompilerDiagnosticSeverity.Error);

        var warningResult = new CompilerResult<string>("generated", [warning]);
        var errorResult = CompilerResult<string>.Failure([error]);

        Assert.True(warningResult.IsSuccess);
        Assert.False(warningResult.HasErrors);
        Assert.False(errorResult.IsSuccess);
        Assert.True(errorResult.HasErrors);
        Assert.Null(errorResult.Value);
    }

    [Fact]
    public void Tree_builder_rejects_duplicate_names_and_unknown_references()
    {
        var builder = new CompilerComputationTreeBuilder();
        var type = new CompilerTensorType(CompilerElementType.Float32, [new CompilerFixedDimension(1)]);
        builder.AddInput(new CompilerValue("input", type));

        Assert.Throws<ArgumentException>(() => builder.AddInput(new CompilerValue("input", type)));
        Assert.Throws<ArgumentException>(() => builder.AddOperation(new CompilerOperation(
            "add",
            new CompilerOperatorDescriptor("Add", capability: CompilerOperationCapability.Bidirectional),
            [new CompilerValueReference("missing")],
            [new CompilerValueReference("input")])));
    }

    [Fact]
    public void Compiler_assembly_has_no_consumer_project_reference()
    {
        var references = typeof(CompilerComputationTree).Assembly
            .GetReferencedAssemblies()
            .Select(static x => x.Name)
            .Where(static x => x is not null)
            .ToArray();

        Assert.DoesNotContain("Onnxify.ModelGenerator", references);
        Assert.DoesNotContain("Onnxify.TorchSharp", references);
    }

    private static CompilerComputationTree CreateRepresentativeTree()
    {
        var tensorType = new CompilerTensorType(
            CompilerElementType.Float32,
            [
                new CompilerFixedDimension(1),
                new CompilerSymbolicDimension("sequence"),
                new CompilerUnknownDimension(),
            ],
            denotation: "activation");
        var scalarType = new CompilerScalarType(CompilerElementType.Float32);
        var output = new CompilerValue("output", tensorType);
        var hidden = new CompilerValue("hidden", tensorType);
        var input = new CompilerValue("input", tensorType);
        var weightsType = new CompilerTensorType(
            CompilerElementType.Float32,
            [new CompilerFixedDimension(2), new CompilerFixedDimension(2)]);
        var weights = new CompilerStateMember(
            "weights",
            CompilerStateMemberKind.Initializer,
            weightsType,
            new CompilerTensorLiteral(
                CompilerElementType.Float32,
                [new CompilerFixedDimension(2), new CompilerFixedDimension(2)],
                [
                    new CompilerFloatingPointLiteral(CompilerElementType.Float32, 1),
                    new CompilerFloatingPointLiteral(CompilerElementType.Float32, 0),
                    new CompilerFloatingPointLiteral(CompilerElementType.Float32, 0),
                    new CompilerFloatingPointLiteral(CompilerElementType.Float32, 1),
                ]));

        var builder = new CompilerComputationTreeBuilder("representative");
        builder.AddInput(input);
        builder.AddOutput(output);
        builder.AddIntermediateValue(hidden);
        builder.AddStateMember(new CompilerStateMember("parameter", CompilerStateMemberKind.Parameter, scalarType));
        builder.AddStateMember(new CompilerStateMember("buffer", CompilerStateMemberKind.Buffer, tensorType));
        builder.AddStateMember(weights);
        builder.AddBlock(new CompilerComputationBlock(
            "identity_block",
            [new CompilerValueReference("block_input")],
            [new CompilerValueReference("block_output")],
            new CompilerBlockStatement(
            [
                new CompilerReturnStatement(new CompilerReferenceExpression("block_input")),
            ])));
        builder.AddOperation(new CompilerOperation(
            "add",
            new CompilerOperatorDescriptor(
                "Add",
                capability: CompilerOperationCapability.Bidirectional,
                constraints: ["broadcasting"]),
            [new CompilerValueReference("input"), new CompilerValueReference("weights")],
            [new CompilerValueReference("hidden")],
            [new CompilerAttribute(
                "alpha",
                new CompilerFloatingPointLiteral(CompilerElementType.Float32, 1))]));
        builder.AddOperation(new CompilerModuleCall(
            "identity",
            "identity_block",
            [new CompilerValueReference("hidden")],
            [new CompilerValueReference("output")]));
        builder.SetSyntaxBody(new CompilerBlockStatement(
        [
            new CompilerDeclarationStatement(
                "constant",
                new CompilerLiteralExpression(new CompilerSignedIntegerLiteral(CompilerElementType.Int64, 1))),
            new CompilerReturnStatement(new CompilerReferenceExpression("output")),
        ]));
        builder.AddMetadata("purpose", "structural-test");
        return builder.Build();
    }

    private static CompilerBlockStatement CreateSyntaxBody()
    {
        return new CompilerBlockStatement(
        [
            new CompilerDeclarationStatement("values", new CompilerArrayExpression(
            [
                new CompilerLiteralExpression(new CompilerSignedIntegerLiteral(CompilerElementType.Int64, 1)),
                new CompilerLiteralExpression(new CompilerSignedIntegerLiteral(CompilerElementType.Int64, 2)),
            ])),
            new CompilerStaticIfStatement(
                new CompilerLiteralExpression(new CompilerBooleanLiteral(true)),
                new CompilerAssignmentStatement(
                    new CompilerReferenceExpression("x"),
                    new CompilerIndexerExpression(
                        new CompilerReferenceExpression("values"),
                        new CompilerLiteralExpression(new CompilerSignedIntegerLiteral(CompilerElementType.Int64, 0)))),
                new CompilerReturnStatement(new CompilerReferenceExpression("fallback"))),
            new CompilerStaticForeachStatement(
                "item",
                new CompilerReferenceExpression("values"),
                new CompilerExpressionStatement(new CompilerInvocationExpression(
                    new CompilerReferenceExpression("consume"),
                    [new CompilerReferenceExpression("item")]))),
            new CompilerReturnStatement(new CompilerTupleExpression(
            [
                new CompilerReferenceExpression("x"),
                new CompilerReferenceExpression("values"),
            ])),
        ]);
    }

    private static CompilerDiagnostic CreateUnsupportedDiagnostic()
    {
        return new CompilerDiagnostic(
            CompilerDiagnosticCodes.Unsupported,
            "The operation is not supported in this direction.",
            CompilerDiagnosticStage.Analyze,
            CompilerDiagnosticSeverity.Error,
            new CompilerSourceSpan(
                CompilerSourceSpanKind.CSharp,
                "model.cs",
                10,
                4,
                2,
                3,
                2,
                7,
                "forward",
                "torch.add"),
            new CompilerDiagnosticContext("forward", "torch.add"));
    }

    private sealed class TestCompilerSession : ICompilerSession
    {
        public CompilerResult<ICompilerTree> CreateTree(ICompilerSource source)
        {
            if (source is null)
            {
                throw new ArgumentNullException(nameof(source));
            }
            return CompilerResult<ICompilerTree>.Success(new TestCompilerTree(source));
        }

        public CompilerResult<TOutput> Generate<TOutput>(ICompilerTree tree, ICompilerSink<TOutput> sink)
        {
            if (tree is null)
            {
                throw new ArgumentNullException(nameof(tree));
            }

            if (sink is null)
            {
                throw new ArgumentNullException(nameof(sink));
            }

            if (sink.Kind == CompilerTargetKind.CSharp && typeof(TOutput) == typeof(string))
            {
                return CompilerResult<TOutput>.Success((TOutput)(object)"generated");
            }

            if (sink.Kind == CompilerTargetKind.OnnxGraph && typeof(TOutput) == typeof(OnnxGraph))
            {
                return CompilerResult<TOutput>.Success((TOutput)(object)OnnxModel.Create().Graph);
            }

            return CompilerResult<TOutput>.Failure(
            [
                new CompilerDiagnostic(
                    CompilerDiagnosticCodes.Unsupported,
                    "The test session does not support this compiler sink.",
                    CompilerDiagnosticStage.Emit,
                    CompilerDiagnosticSeverity.Error),
            ]);
        }
    }

    private sealed class TestCompilerTree(ICompilerSource source) : ICompilerTree
    {
        public ICompilerSource Source { get; } = source;
    }

    // These methods are compile-only consumer smoke checks. Calling the ModelGenerator method
    // would load its Roslyn analyzer dependencies into this runtime test process.
    private static ICompilerSource CreateModelGeneratorSource(OnnxModelGenerator consumer)
    {
        _ = consumer;
        return new OnnxCompilerSource(OnnxModel.Create());
    }

    private static ICompilerSource CreateTorchSharpSource(TorchTensorDataType consumer)
    {
        _ = consumer;
        return new CSharpTorchSharpSource("return input;");
    }
}
