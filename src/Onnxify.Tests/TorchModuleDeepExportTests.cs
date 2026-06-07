using Onnxify.TorchSharp;
using static TorchSharp.torch;
using TorchModule = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>;

using static TorchSharp.torch.nn.functional;

namespace Onnxify.Tests;

public sealed class TorchModuleDeepExportTests
{
    [Fact]
    public void DeepExport_ForLstmStyleForward_EmitsGraphFromForwardAst()
    {
        using var module = new DeepExportTestModule();
        module.eval();

        var model = module.ExportOnnxModel(
            input: OnnxTensorType.Create<long>(["batch_size", "seq_len"]),
            output: OnnxTensorType.Create<float>(["batch_size", 3]),
            options: new OnnxModelCreationOptions
            {
                Opset = 22,
            }
        );

        Assert.Collection(
            model.Graph.Inputs,
            input => Assert.Equal("input", input.Name)
        );

        Assert.Collection(
            model.Graph.Outputs,
            output => Assert.Equal("output", output.Name)
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Gather");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "LSTM");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "MatMul");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "ReduceSum");
        Assert.Equal("Identity", model.Graph.Nodes.Last().OpType);
    }

    [Fact]
    public void DeepExport_ForMiniGptStyleForward_EmitsRecursiveTransformerGraph()
    {
        using var module = new DeepExportMiniGptModule();
        module.eval();

        var model = module.ExportOnnxModel(
            input: OnnxTensorType.Create<long>(["batch", module.MaxSequenceLength]),
            output: OnnxTensorType.Create<float>(["batch", module.MaxSequenceLength, module.VocabularySize]),
            options: new OnnxModelCreationOptions
            {
                Opset = 22,
            }
        );

        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<long>>(),
            initializer => initializer.Shape.SequenceEqual([module.MaxSequenceLength])
                && initializer.Value.SequenceEqual(Enumerable.Range(0, module.MaxSequenceLength).Select(static x => (long)x))
        );
        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<float>>(),
            initializer => initializer.Shape.SequenceEqual([module.MaxSequenceLength, module.MaxSequenceLength])
                && initializer.Value.Contains(-10_000f)
        );
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Gather");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Reshape");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Transpose");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "MatMul");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Softmax");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Add");
        Assert.Equal("Identity", model.Graph.Nodes.Last().OpType);
    }

    [Fact]
    public void DeepExport_ForLocalHelperProjection_InlinesHelperAndMaterializesTransposedWeight()
    {
        using var module = new DeepExportHelperProjectionModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<long>(["batch", 2]),
            output: OnnxTensorType.Create<float>(["batch", 2, module.VocabularySize])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Gather");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "MatMul");
        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<float>>(),
            initializer => initializer.Shape.SequenceEqual([module.HiddenSize, module.VocabularySize])
        );
        Assert.Equal("Identity", model.Graph.Nodes.Last().OpType);
    }

    [Fact]
    public void DeepExport_ForArangeStartStepWithNamedMetadata_EmitsExpectedInitializer()
    {
        using var module = new DeepExportArangeStartStepModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([1]),
            output: OnnxTensorType.Create<float>([1, 4, module.EmbeddingDimensions])
        );

        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<long>>(),
            initializer => initializer.Shape.SequenceEqual([4])
                && initializer.Value.SequenceEqual([1L, 3L, 5L, 7L])
        );
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Unsqueeze");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Gather");
    }

    [Fact]
    public void DeepExport_ForFullTriuWithNamedMetadata_EmitsExpectedInitializerValues()
    {
        using var module = new DeepExportFullTriuModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([3, 3]),
            output: OnnxTensorType.Create<float>([3, 3])
        );

        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<float>>(),
            initializer => initializer.Shape.SequenceEqual([3, 3])
                && initializer.Value.SequenceEqual([0f, -2f, -2f, 0f, 0f, -2f, 0f, 0f, 0f])
        );
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Add");
    }

    [Fact]
    public void DeepExport_ForSymbolicSliceEndOnIntermediate_DoesNotCollapseEndToZero()
    {
        using var module = new DeepExportSymbolicSliceEndModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([4, 4]),
            output: OnnxTensorType.Create<float>([4, 4])
        );

        var sliceEnds = model.Graph.Initializers
            .OfType<OnnxTensor<long>>()
            .Where(initializer => initializer.Name.Contains("_ends", StringComparison.Ordinal))
            .Select(initializer => initializer.Value.Single())
            .ToArray();

        Assert.Contains(4L, sliceEnds);
        Assert.Contains(long.MaxValue, sliceEnds);
        Assert.DoesNotContain(0L, sliceEnds);
        Assert.Equal(2, model.Graph.Nodes.Count(node => node.OpType == "Slice"));
    }

    [Fact]
    public void DeepExport_ForValidationStatement_IgnoresRuntimeGuard()
    {
        using var module = new DeepExportValidationGuardModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>(["batch", 3]),
            output: OnnxTensorType.Create<float>(["batch", 2])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "MatMul");
        Assert.Equal("Identity", model.Graph.Nodes.Last().OpType);
    }

    [Fact]
    public void DeepExport_ForNonValidationVoidHelper_ThrowsClearNotSupported()
    {
        using var module = new DeepExportUnsupportedVoidHelperModule();

        var exception = Assert.Throws<NotSupportedException>(
            () => ExportDeep(
                module,
                input: OnnxTensorType.Create<float>([2, 2]),
                output: OnnxTensorType.Create<float>([2, 2])
            )
        );

        Assert.Contains("did not return", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DeepExport_ForNestedScalarArithmetic_FoldsConstantsForFactoriesAndTensorScale()
    {
        using var module = new DeepExportNestedScalarArithmeticModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([1]),
            output: OnnxTensorType.Create<float>([1, 3, module.EmbeddingDimensions])
        );

        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<long>>(),
            initializer => initializer.Shape.SequenceEqual([3])
                && initializer.Value.SequenceEqual([3L, 6L, 9L])
        );
        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<float>>(),
            initializer => initializer.Shape.SequenceEqual([])
                && initializer.Value.Single() == 2f
        );
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Mul");
    }

    [Fact]
    public void DeepExport_ForTensorScalarArithmetic_EmitsScalarAddSubAndDiv()
    {
        using var module = new DeepExportTensorScalarArithmeticModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([2, 2]),
            output: OnnxTensorType.Create<float>([2, 2])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Add");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Sub");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Div");
        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<float>>(),
            initializer => initializer.Shape.SequenceEqual([])
                && initializer.Value.Single() == 8f
        );
        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<float>>(),
            initializer => initializer.Shape.SequenceEqual([])
                && initializer.Value.Single() == 2f
        );
    }

    [Fact]
    public void DeepExport_ForCatCollectionExpression_ResolvesInlineArrayTensorList()
    {
        using var module = new DeepExportCatCollectionExpressionModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>(["batch", 2, 3]),
            output: OnnxTensorType.Create<float>(["batch", 4, 3])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Concat");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Add");
    }

    [Fact]
    public void DeepExport_ForViewCollectionExpression_ResolvesInlineArrayShape()
    {
        using var module = new DeepExportViewCollectionExpressionModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([6]),
            output: OnnxTensorType.Create<float>([2, 3])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Reshape");
    }

    [Fact]
    public void DeepExport_ForPermuteCollectionExpression_ResolvesInlineArrayPermutation()
    {
        using var module = new DeepExportPermuteCollectionExpressionModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([2, 3, 4]),
            output: OnnxTensorType.Create<float>([2, 4, 3])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Transpose");
    }

    [Fact]
    public void DeepExport_ForExplicitNewLongArrayShape_ResolvesTorchSharpExamplesViewSyntax()
    {
        using var module = new DeepExportExplicitArrayShapeModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([2, 2, 2]),
            output: OnnxTensorType.Create<float>([2, 4])
        );

        Assert.Equal(2, model.Graph.Nodes.Count(node => node.OpType == "Reshape"));
        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<long>>(),
            initializer => initializer.Value.SequenceEqual([-1L, 4L])
        );
    }

    [Fact]
    public void DeepExport_ForFunctionalActivationCalls_ExportsTorchSharpExamplesSyntax()
    {
        using var module = new DeepExportFunctionalActivationModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([2, 3]),
            output: OnnxTensorType.Create<float>([2, 3])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Relu");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Sigmoid");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "LogSoftmax");
    }

    [Fact]
    public void DeepExport_ForTupleReturningLocalHelper_DeconstructsVaeStyleSyntax()
    {
        using var module = new DeepExportTupleHelperModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([2, 4]),
            output: OnnxTensorType.Create<float>([2, 3])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Relu");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Exp");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Add");
    }

    [Fact]
    public void DeepExport_ForPaddingMaskChain_ExportsNotEqualUnsqueezeCastSumAndClamp()
    {
        using var module = new DeepExportPaddingMaskModule();

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<long>(["batch_size", "seq_len"]),
            output: OnnxTensorType.Create<float>(["batch_size", 3])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Equal");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Not");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Unsqueeze");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Cast");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "ReduceSum");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Max");
        Assert.Contains(model.Graph.Nodes, node => node.OpType == "Div");
        Assert.Contains(
            model.Graph.Initializers.OfType<OnnxTensor<long>>(),
            initializer => initializer.Shape.SequenceEqual([1])
                && initializer.Value.SequenceEqual([-1L])
        );
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(true, 2)]
    public void DeepExport_ForStaticNullConditionalExpression_ExportsSelectedBranch(
        bool hasOptionalProjection,
        int expectedMatMulCount
    )
    {
        using var module = new DeepExportNullConditionalModule(hasOptionalProjection);

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([2, 3]),
            output: OnnxTensorType.Create<float>([2, 2])
        );

        Assert.Equal(expectedMatMulCount, model.Graph.Nodes.Count(node => node.OpType == "MatMul"));
        Assert.Equal("Identity", model.Graph.Nodes.Last().OpType);
    }

    [Theory]
    [InlineData(true, "Add")]
    [InlineData(false, "Mul")]
    public void DeepExport_ForStaticBoolConditionalExpression_ExportsSelectedBranch(
        bool useAdd,
        string expectedOperator
    )
    {
        using var module = new DeepExportBoolConditionalModule(useAdd);

        var model = ExportDeep(
            module,
            input: OnnxTensorType.Create<float>([2, 2]),
            output: OnnxTensorType.Create<float>([2, 2])
        );

        Assert.Contains(model.Graph.Nodes, node => node.OpType == expectedOperator);
    }

    [Fact]
    public void DeepExport_ForLoopStatement_ThrowsUnsupportedStatement()
    {
        using var module = new DeepExportLoopModule();

        var exception = Assert.Throws<NotSupportedException>(
            () => ExportDeep(
                module,
                input: OnnxTensorType.Create<float>([2, 2]),
                output: OnnxTensorType.Create<float>([2, 2])
            )
        );

        Assert.Contains("Unsupported forward statement", exception.Message, StringComparison.Ordinal);
        Assert.Contains("ForStatement", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DeepExport_ForDynamicIfElseStatement_ThrowsUnsupportedStatement()
    {
        using var module = new DeepExportDynamicIfElseModule();

        var exception = Assert.Throws<NotSupportedException>(
            () => ExportDeep(
                module,
                input: OnnxTensorType.Create<float>([2, 2]),
                output: OnnxTensorType.Create<float>([2, 2])
            )
        );

        Assert.Contains("If statement condition must be statically resolvable", exception.Message, StringComparison.Ordinal);
    }

    private static OnnxModel ExportDeep(
        TorchModule module,
        OnnxTensorType input,
        OnnxTensorType output
    )
    {
        module.eval();
        return module.ExportOnnxModel(
            input: input,
            output: output,
            options: new OnnxModelCreationOptions
            {
                Opset = 22,
            }
        );
    }

    private sealed class DeepExportHelperProjectionModule : TorchModule
    {
        private const int VOCABULARY_SIZE = 7;
        private const int HIDDEN_SIZE = 3;

        private readonly global::TorchSharp.Modules.Embedding _embedding;

        public DeepExportHelperProjectionModule()
            : base(nameof(DeepExportHelperProjectionModule))
        {
            _embedding = nn.Embedding(VOCABULARY_SIZE, HIDDEN_SIZE);

            RegisterComponents();
        }

        public int VocabularySize => VOCABULARY_SIZE;

        public int HiddenSize => HIDDEN_SIZE;

        public override Tensor forward(Tensor tokens)
        {
            var hidden = EmbedTokens(tokens);
            return Project(hidden);
        }

        private Tensor EmbedTokens(Tensor tokens)
        {
            return _embedding.forward(tokens);
        }

        private Tensor Project(Tensor hiddenStates)
        {
            var weight = _embedding.weight
                ?? throw new InvalidOperationException("Embedding weights are not initialized.");

            using var transposed = weight.transpose(0, 1);
            return matmul(hiddenStates, transposed);
        }
    }

    private sealed class DeepExportArangeStartStepModule : TorchModule
    {
        private const int EMBEDDING_DIMENSIONS = 2;

        private readonly global::TorchSharp.Modules.Embedding _embedding;

        public DeepExportArangeStartStepModule()
            : base(nameof(DeepExportArangeStartStepModule))
        {
            _embedding = nn.Embedding(10, EMBEDDING_DIMENSIONS);

            RegisterComponents();
        }

        public int EmbeddingDimensions => EMBEDDING_DIMENSIONS;

        public override Tensor forward(Tensor input)
        {
            using var ids = global::TorchSharp.torch.arange(
                1L,
                8L,
                2L,
                dtype: ScalarType.Int64,
                device: input.device
            ).unsqueeze(0);

            return _embedding.forward(ids);
        }
    }

    private sealed class DeepExportFullTriuModule : TorchModule
    {
        public DeepExportFullTriuModule()
            : base(nameof(DeepExportFullTriuModule))
        { }

        public override Tensor forward(Tensor input)
        {
            using var mask = global::TorchSharp.torch.triu(
                global::TorchSharp.torch.full(
                    [3L, 3L],
                    -2f,
                    dtype: input.dtype,
                    device: input.device
                ),
                diagonal: 1
            );

            return input + mask;
        }
    }

    private sealed class DeepExportSymbolicSliceEndModule : TorchModule
    {
        public DeepExportSymbolicSliceEndModule()
            : base(nameof(DeepExportSymbolicSliceEndModule))
        { }

        public override Tensor forward(Tensor input)
        {
            var sequenceLength = input.shape[1];

            using var mask = global::TorchSharp.torch.triu(
                global::TorchSharp.torch.full(
                    [4L, 4L],
                    -3f,
                    dtype: input.dtype,
                    device: input.device
                ),
                diagonal: 1
            )
            .slice(0, 0, sequenceLength, 1)
            .slice(1, 0, sequenceLength, 1);

            return input + mask;
        }
    }

    private sealed class DeepExportValidationGuardModule : TorchModule
    {
        private readonly global::TorchSharp.Modules.Linear _linear;

        public DeepExportValidationGuardModule()
            : base(nameof(DeepExportValidationGuardModule))
        {
            _linear = nn.Linear(3, 2);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            ValidateInputShape(input);
            return _linear.forward(input);
        }

        private static void ValidateInputShape(Tensor input)
        {
            if (input.shape.Length != 2)
            {
                throw new ArgumentException("Expected rank-2 input.", nameof(input));
            }
        }
    }

    private sealed class DeepExportUnsupportedVoidHelperModule : TorchModule
    {
        public DeepExportUnsupportedVoidHelperModule()
            : base(nameof(DeepExportUnsupportedVoidHelperModule))
        { }

        public override Tensor forward(Tensor input)
        {
            Touch(input);
            return input;
        }

        private static void Touch(Tensor input)
        { }
    }

    private sealed class DeepExportNestedScalarArithmeticModule : TorchModule
    {
        private const int EMBEDDING_DIMENSIONS = 2;

        private readonly long _longBase = 1L;
        private readonly float _floatBase = 1f;
        private readonly global::TorchSharp.Modules.Embedding _embedding;

        public DeepExportNestedScalarArithmeticModule()
            : base(nameof(DeepExportNestedScalarArithmeticModule))
        {
            _embedding = nn.Embedding(12, EMBEDDING_DIMENSIONS);

            RegisterComponents();
        }

        public int EmbeddingDimensions => EMBEDDING_DIMENSIONS;

        public override Tensor forward(Tensor input)
        {
            var start = ((_longBase + 2L) * 2L) - 3L;
            var end = ((_longBase + 4L) * 2L) + 1L;
            var step = ((_longBase + 7L) / 4L) + 1L;
            var scale = (((_floatBase + 2f) * 4f) - 6f) / 3f;

            using var ids = arange(
                start,
                end,
                step,
                dtype: ScalarType.Int64,
                device: input.device
            ).unsqueeze(0);

            return _embedding.forward(ids) * scale;
        }
    }

    private sealed class DeepExportTensorScalarArithmeticModule : TorchModule
    {
        private readonly float _base = 1f;

        public DeepExportTensorScalarArithmeticModule()
            : base(nameof(DeepExportTensorScalarArithmeticModule))
        { }

        public override Tensor forward(Tensor input)
        {
            var offset = ((_base + 2f) * 3f) - 1f;
            var subtract = _base + 1f;
            var divisor = (_base + 5f) / 3f;

            return ((input + offset) - subtract) / divisor;
        }
    }

    private sealed class DeepExportCatCollectionExpressionModule : TorchModule
    {
        public DeepExportCatCollectionExpressionModule()
            : base(nameof(DeepExportCatCollectionExpressionModule))
        { }

        public override Tensor forward(Tensor input)
        {
            var shifted = input + 1f;

            return cat([input, shifted], dim: 1);
        }
    }

    private sealed class DeepExportViewCollectionExpressionModule : TorchModule
    {
        public DeepExportViewCollectionExpressionModule()
            : base(nameof(DeepExportViewCollectionExpressionModule))
        { }

        public override Tensor forward(Tensor input)
        {
            return input.view([2, 3]);
        }
    }

    private sealed class DeepExportPermuteCollectionExpressionModule : TorchModule
    {
        public DeepExportPermuteCollectionExpressionModule()
            : base(nameof(DeepExportPermuteCollectionExpressionModule))
        { }

        public override Tensor forward(Tensor input)
        {
            return input.permute([0, 2, 1]);
        }
    }

    private sealed class DeepExportExplicitArrayShapeModule : TorchModule
    {
        public DeepExportExplicitArrayShapeModule()
            : base(nameof(DeepExportExplicitArrayShapeModule))
        { }

        public override Tensor forward(Tensor input)
        {
            var reshaped = input.reshape(-1, 4);

            return reshaped.view(new long[] { reshaped.shape[0], 4 });
        }
    }

    private sealed class DeepExportFunctionalActivationModule : TorchModule
    {
        private readonly global::TorchSharp.Modules.Linear _linear;

        public DeepExportFunctionalActivationModule()
            : base(nameof(DeepExportFunctionalActivationModule))
        {
            _linear = nn.Linear(3, 3);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = relu(_linear.forward(input));
            x = global::TorchSharp.torch.sigmoid(x);
            return log_softmax(x, dim: 1);
        }
    }

    private sealed class DeepExportTupleHelperModule : TorchModule
    {
        private readonly global::TorchSharp.Modules.Linear _input;
        private readonly global::TorchSharp.Modules.Linear _mu;
        private readonly global::TorchSharp.Modules.Linear _logVar;

        public DeepExportTupleHelperModule()
            : base(nameof(DeepExportTupleHelperModule))
        {
            _input = nn.Linear(4, 5);
            _mu = nn.Linear(5, 3);
            _logVar = nn.Linear(5, 3);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var (mu, logvar) = Encode(input);
            return mu + global::TorchSharp.torch.exp(0.5f * logvar);
        }

        private (Tensor Mu, Tensor LogVar) Encode(Tensor input)
        {
            var hidden = relu(_input.forward(input));
            return (_mu.forward(hidden), _logVar.forward(hidden));
        }
    }

    private sealed class DeepExportPaddingMaskModule : TorchModule
    {
        private readonly global::TorchSharp.Modules.Embedding _embedding;
        private readonly global::TorchSharp.Modules.Linear _head;

        public DeepExportPaddingMaskModule()
            : base(nameof(DeepExportPaddingMaskModule))
        {
            _embedding = nn.Embedding(8, 4, padding_idx: 0);
            _head = nn.Linear(4, 3);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var emb = _embedding.forward(input);
            var logitsPerToken = _head.forward(emb);
            var mask = input.ne(0)
                .unsqueeze(-1)
                .to_type(logitsPerToken.dtype);
            var maskedLogits = logitsPerToken * mask;
            var lengths = mask.sum(1).clamp(min: 1);

            return maskedLogits.sum(1) / lengths;
        }
    }

    private sealed class DeepExportNullConditionalModule : TorchModule
    {
        private readonly global::TorchSharp.Modules.Linear? _optionalProjection;
        private readonly global::TorchSharp.Modules.Linear _outputProjection;

        public DeepExportNullConditionalModule(bool hasOptionalProjection)
            : base(nameof(DeepExportNullConditionalModule))
        {
            _optionalProjection = hasOptionalProjection
                ? nn.Linear(3, 3)
                : null;
            _outputProjection = nn.Linear(3, 2);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = _optionalProjection == null
                ? input
                : _optionalProjection.forward(input);

            return _outputProjection.forward(x);
        }
    }

    private sealed class DeepExportBoolConditionalModule : TorchModule
    {
        private readonly bool _useAdd;

        public DeepExportBoolConditionalModule(bool useAdd)
            : base(nameof(DeepExportBoolConditionalModule))
        {
            _useAdd = useAdd;
        }

        public override Tensor forward(Tensor input)
        {
            return _useAdd
                ? input + 1f
                : input * 2f;
        }
    }

    private sealed class DeepExportLoopModule : TorchModule
    {
        public DeepExportLoopModule()
            : base(nameof(DeepExportLoopModule))
        { }

        public override Tensor forward(Tensor input)
        {
            var x = input;
            for (var index = 0; index < 2; index++)
            {
                x = x + input;
            }

            return x;
        }
    }

    private sealed class DeepExportDynamicIfElseModule : TorchModule
    {
        public DeepExportDynamicIfElseModule()
            : base(nameof(DeepExportDynamicIfElseModule))
        { }

        public override Tensor forward(Tensor input)
        {
            if (DateTime.UtcNow.Ticks > 0)
            {
                return input + input;
            }

            return input * input;
        }
    }

    private sealed class DeepExportTestModule : TorchModule
    {
        private readonly global::TorchSharp.Modules.Embedding _embedding;
        private readonly global::TorchSharp.Modules.LSTM _lstm;
        private readonly global::TorchSharp.Modules.Linear _linear;

        public DeepExportTestModule()
            : base(nameof(DeepExportTestModule))
        {
            _embedding = nn.Embedding(8, 4);
            _lstm = nn.LSTM(4, 5, numLayers: 1, bidirectional: true, batchFirst: true);
            _linear = nn.Linear(10, 3);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = _embedding.forward(input);
            var (lstm, _, _) = _lstm.forward(x);
            var linear = _linear.forward(lstm);
            return sum(linear, 1);
        }
    }

    private sealed class DeepExportMiniGptModule : TorchModule
    {
        private const int VOCABULARY_SIZE = 16;
        private const int MAX_SEQUENCE_LENGTH = 4;
        private const int ATTENTION_HEADS = 2;
        private const int ATTENTION_DIMENSIONS = 8;

        private readonly global::TorchSharp.Modules.Embedding _tokenEmbedding;
        private readonly global::TorchSharp.Modules.Embedding _positionEmbedding;
        private readonly DeepExportTransformerBlock _block;
        private readonly global::TorchSharp.Modules.LayerNorm _outputNorm;

        public DeepExportMiniGptModule()
            : base(nameof(DeepExportMiniGptModule))
        {
            _tokenEmbedding = nn.Embedding(VOCABULARY_SIZE, ATTENTION_DIMENSIONS);
            _positionEmbedding = nn.Embedding(MAX_SEQUENCE_LENGTH, ATTENTION_DIMENSIONS);
            _block = new DeepExportTransformerBlock(ATTENTION_HEADS, ATTENTION_DIMENSIONS, MAX_SEQUENCE_LENGTH);
            _outputNorm = nn.LayerNorm([ATTENTION_DIMENSIONS]);

            RegisterComponents();
        }

        public int VocabularySize => VOCABULARY_SIZE;

        public int MaxSequenceLength => MAX_SEQUENCE_LENGTH;

        public override Tensor forward(Tensor tokens)
        {
            ValidateInputShape(tokens);

            using var positions = CreatePositionIds(tokens.shape[0], tokens.device);

            var x = _tokenEmbedding.forward(tokens) + _positionEmbedding.forward(positions);
            x = _block.forward(x);
            x = _outputNorm.forward(x);
            return ComputeLogits(x);
        }

        private Tensor ComputeLogits(Tensor hiddenStates)
        {
            using var tiedWeight = _tokenEmbedding.weight!.transpose(0, 1);
            return matmul(hiddenStates, tiedWeight);
        }

        private Tensor CreatePositionIds(long batchSize, Device device)
        {
            return arange(MAX_SEQUENCE_LENGTH, dtype: ScalarType.Int64, device: device)
                .unsqueeze(0)
                .expand(batchSize, MAX_SEQUENCE_LENGTH);
        }

        private static void ValidateInputShape(Tensor tokens)
        {
            if (tokens.shape.Length != 2)
            {
                throw new ArgumentException("Expected token ids with rank 2.", nameof(tokens));
            }
        }
    }

    private sealed class DeepExportTransformerBlock : TorchModule
    {
        private readonly global::TorchSharp.Modules.LayerNorm _attentionNorm;
        private readonly DeepExportCausalSelfAttention _attention;
        private readonly global::TorchSharp.Modules.LayerNorm _feedForwardNorm;
        private readonly global::TorchSharp.Modules.Linear _feedForward;

        public DeepExportTransformerBlock(
            int headCount,
            int attentionDimensions,
            int maxContextLength
        ) : base(nameof(DeepExportTransformerBlock))
        {
            _attentionNorm = nn.LayerNorm([attentionDimensions]);
            _attention = new DeepExportCausalSelfAttention(headCount, attentionDimensions, maxContextLength);
            _feedForwardNorm = nn.LayerNorm([attentionDimensions]);
            _feedForward = nn.Linear(attentionDimensions, attentionDimensions);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = x + _attention.forward(_attentionNorm.forward(x));
            x = x + _feedForward.forward(_feedForwardNorm.forward(x));
            return x;
        }
    }

    private sealed class DeepExportCausalSelfAttention : TorchModule
    {
        private readonly int _headCount;
        private readonly int _headDimension;
        private readonly int _attentionDimensions;
        private readonly int _maxContextLength;
        private readonly float _scale;
        private readonly global::TorchSharp.Modules.Linear _attention;
        private readonly global::TorchSharp.Modules.Linear _projection;

        public DeepExportCausalSelfAttention(
            int headCount,
            int attentionDimensions,
            int maxContextLength
        ) : base(nameof(DeepExportCausalSelfAttention))
        {
            _headCount = headCount;
            _headDimension = attentionDimensions / headCount;
            _attentionDimensions = attentionDimensions;
            _maxContextLength = maxContextLength;
            _scale = 1.0f / MathF.Sqrt(_headDimension);
            _attention = nn.Linear(attentionDimensions, attentionDimensions * 3);
            _projection = nn.Linear(attentionDimensions, attentionDimensions);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var batchSize = x.shape[0];
            var sequenceLength = x.shape[1];

            var qkv = _attention.forward(x);
            qkv = qkv
                .view([batchSize, sequenceLength, 3, _headCount, _headDimension])
                .permute(2, 0, 3, 1, 4);

            var query = qkv[0];
            var key = qkv[1];
            var value = qkv[2];

            using var causalMask =
                triu(
                    full([_maxContextLength, _maxContextLength], -10_000f, dtype: x.dtype, device: x.device),
                    diagonal: 1
                )
                .slice(0, 0, sequenceLength, 1)
                .slice(1, 0, sequenceLength, 1)
                .unsqueeze(0)
                .unsqueeze(0);

            var attentionScores = matmul(query, key.transpose(2, 3)) * _scale;
            attentionScores = attentionScores + causalMask;
            var attentionWeights = softmax(attentionScores, dim: -1);
            var attentionOutput = matmul(attentionWeights, value);

            attentionOutput = attentionOutput
                .permute(0, 2, 1, 3)
                .contiguous()
                .view([batchSize, sequenceLength, _attentionDimensions]);

            return _projection.forward(attentionOutput);
        }
    }
}
