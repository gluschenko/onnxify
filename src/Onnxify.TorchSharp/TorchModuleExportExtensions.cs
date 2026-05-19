using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using ICSharpCode.Decompiler;
using ICSharpCode.Decompiler.CSharp;
using ICSharpCode.Decompiler.CSharp.Syntax;
using ICSharpCode.Decompiler.Metadata;

namespace Onnxify.TorchSharp;

public static class TorchModuleExportExtensions
{
    public static OnnxModel Export(
        this TorchModule module,
        OnnxTensorType input,
        OnnxTensorType output,
        OnnxModelCreationOptions options
    )
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(options);

        var onnxModel = OnnxModel.Create(options);
        var graph = onnxModel.Graph;

        var inputName = graph.NextName("input");
        var outputName = graph.NextName("output");

        var inputEdge = graph.AddInput(inputName, input);
        graph.AddOutput(outputName, output);

        var forward = DecompileForward(module);

        // Interpret the decompiled forward body as a small data-flow program.
        // Only tensor-producing inference patterns are lowered; unsupported C# fails loudly.
        var result = ExportForwardBody(module, graph, inputEdge, forward);
        var outputEdge = graph.AddEdge(outputName);

        graph.Identity(
            name: graph.NextName("output_identity"),
            options: new IdentityInputOutputOptions
            {
                Input = result,
                Output = outputEdge,
            }
        );

        return onnxModel;
    }

    private static IOnnxGraphEdge ExportForwardBody(
        TorchModule module,
        OnnxGraph graph,
        IOnnxGraphEdge input,
        MethodDeclaration forward
    )
    {
        var context = new ForwardExportContext(module, graph);

        // The public overload currently supports single-input Module<Tensor, Tensor>.
        // Every decompiled forward parameter is therefore treated as an alias of the same graph input.
        context.Values["input"] = new ExportValue(input);

        foreach (var parameter in forward.Parameters)
        {
            context.Values.TryAdd(parameter.Name, new ExportValue(input));
        }

        IOnnxGraphEdge? result = null;
        foreach (var statement in forward.Body.Statements)
        {
            result = ExportStatement(context, statement) ?? result;
        }

        return result
            ?? throw new NotSupportedException(
                $"Method '{forward.Name}' did not return a supported ONNX graph edge."
            );
    }

    private static IOnnxGraphEdge? ExportStatement(
        ForwardExportContext context,
        Statement statement
    )
    {
        if (TryExportDeconstructionStatement(context, statement))
        {
            return null;
        }

        switch (statement)
        {
            case VariableDeclarationStatement variableDeclaration:
                ExportVariableDeclaration(context, variableDeclaration);
                return null;

            case ExpressionStatement { Expression: AssignmentExpression assignment }:
                AssignValue(context, assignment.Left, ExportExpression(context, assignment.Right));
                return null;

            case ExpressionStatement { Expression: InvocationExpression invocation }:
                ExportStatementInvocation(context, invocation);
                return null;

            // Decompiler emits "using var tensor = ..." as a UsingStatement that contains
            // the remaining forward body; disposal is irrelevant for the synthesized ONNX graph.
            case UsingStatement usingStatement:
                return ExportUsingStatement(context, usingStatement);

            case ReturnStatement returnStatement:
                return ExportExpression(context, returnStatement.Expression).GetRequiredEdge(statement);

            default:
                throw new NotSupportedException(
                    $"Unsupported forward statement '{statement.GetType().Name}': {statement}"
                );
        }
    }

    private static IOnnxGraphEdge? ExportUsingStatement(
        ForwardExportContext context,
        UsingStatement usingStatement
    )
    {
        if (usingStatement.ResourceAcquisition is VariableDeclarationStatement declaration)
        {
            ExportVariableDeclaration(context, declaration);
        }
        else if (usingStatement.ResourceAcquisition is Expression resourceExpression)
        {
            ExportExpression(context, resourceExpression);
        }

        IOnnxGraphEdge? result = null;
        foreach (var nestedStatement in GetNestedStatements(usingStatement.EmbeddedStatement))
        {
            result = ExportStatement(context, nestedStatement) ?? result;
        }

        return result;
    }

    private static void ExportStatementInvocation(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        // Runtime validation guards affect eager execution only. The exported model already
        // gets its input contract from the caller-provided OnnxTensorType.
        if (invocation.Target is IdentifierExpression identifier
            && IsValidationMethodName(identifier.Identifier))
        {
            return;
        }

        ExportExpression(context, invocation);
    }

    private static bool IsValidationMethodName(string methodName)
    {
        return methodName.StartsWith("Validate", StringComparison.Ordinal);
    }

    private static void ExportVariableDeclaration(
        ForwardExportContext context,
        VariableDeclarationStatement declaration
    )
    {
        foreach (var variable in declaration.Variables)
        {
            if (variable.Initializer.IsNull)
            {
                continue;
            }

            // Mini GPT-style attention builds a causal mask at runtime from torch.full/triu/slice.
            // For export this is a deterministic initializer, so materialize it directly instead
            // of lowering the whole mask-construction expression tree.
            context.Values[variable.Name] = IsCausalMaskInitializer(context, variable)
                ? ExportCausalMask(context)
                : ExportExpression(context, variable.Initializer);
        }
    }

    private static bool IsCausalMaskInitializer(
        ForwardExportContext context,
        VariableInitializer variable
    )
    {
        return (string.Equals(variable.Name, "causalMask", StringComparison.Ordinal)
                || variable.Initializer.ToString().Contains("torch.triu", StringComparison.Ordinal))
            && TryGetMemberValue(context.RootModule, "_maxContextLength", out _);
    }

    private static bool TryExportDeconstructionStatement(
        ForwardExportContext context,
        Statement statement
    )
    {
        // Tuple deconstruction may survive as source syntax for some modules, while recurrent
        // modules return wrapper objects such as LSTMOutput. Map tuple slots onto ONNX outputs.
        var text = statement.ToString().Trim();
        var match = Regex.Match(
            text,
            @"^(?:var\s+)?\((?<names>[^)]*)\)\s*=\s*(?<value>.*);$",
            RegexOptions.CultureInvariant
        );

        if (!match.Success)
        {
            return false;
        }

        var invocation = statement.Descendants.OfType<InvocationExpression>().SingleOrDefault()
            ?? throw new NotSupportedException($"Unsupported deconstruction initializer: {statement}");

        var value = ExportExpression(context, invocation);
        var names = match.Groups["names"].Value
            .Split(',')
            .Select(static x => x.Trim())
            .ToArray();

        for (var index = 0; index < names.Length; index++)
        {
            if (string.IsNullOrWhiteSpace(names[index]) || names[index] == "_")
            {
                continue;
            }

            context.Values[names[index]] = GetTupleElement(value, index, statement);
        }

        return true;
    }

    private static ExportValue ExportExpression(
        ForwardExportContext context,
        Expression expression
    )
    {
        switch (expression)
        {
            case IdentifierExpression identifier:
                if (context.Values.TryGetValue(identifier.Identifier, out var value))
                {
                    return value;
                }

                // Decompilation often turns private fields such as "_scale" into plain identifiers.
                // Resolve those against the current module instance when no local value exists.
                if (TryGetMemberValue(context.RootModule, identifier.Identifier, out var memberValue))
                {
                    return new ExportValue(memberValue);
                }

                throw new NotSupportedException($"Unknown forward value '{identifier.Identifier}'.");

            case InvocationExpression invocation:
                return ExportInvocation(context, invocation);

            case MemberReferenceExpression memberReference:
                return ExportMemberReference(context, memberReference);

            case BinaryOperatorExpression binaryOperator:
                return ExportBinaryOperator(context, binaryOperator);

            case IndexerExpression indexer:
                return ExportIndexer(context, indexer);

            case ParenthesizedExpression parenthesized:
                return ExportExpression(context, parenthesized.Expression);

            case CastExpression cast:
                return ExportExpression(context, cast.Expression);

            case PrimitiveExpression primitive:
                return new ExportValue(primitive.Value);

            // C# collection expressions like .view([b, s, h]) can decompile into compiler-generated
            // InlineArray temporaries. Represent the temporary as a mutable shape/permutation builder.
            case DefaultValueExpression defaultValue
                when TryCreateInlineArrayBuilder(defaultValue, out var builder):
                return new ExportValue(builder);

            default:
                throw new NotSupportedException(
                    $"Unsupported forward expression '{expression.GetType().Name}': {expression}"
                );
        }
    }

    private static ExportValue ExportInvocation(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        if (invocation.Target is IdentifierExpression { Identifier: "CreatePositionIds" })
        {
            return ExportCreatePositionIds(context);
        }

        // This is a model-local helper in MiniGpt2LikeModel, not a Torch operator. Lower it as
        // a tied output projection using the token embedding weights.
        if (invocation.Target is IdentifierExpression { Identifier: "ComputeLogits" })
        {
            return ExportComputeLogits(context, invocation);
        }

        if (invocation.Target is not MemberReferenceExpression memberReference)
        {
            throw new NotSupportedException($"Unsupported invocation target: {invocation}");
        }

        if (string.Equals(memberReference.MemberName, "forward", StringComparison.Ordinal))
        {
            return ExportModuleForwardCall(context, memberReference, invocation);
        }

        // Tensor instance methods are lowered here because they are not nn.Module calls and
        // therefore do not go through TorchModuleExtensions.Export dispatch.
        if (IsTensorMethod(memberReference.MemberName))
        {
            return ExportTensorMethodInvocation(context, memberReference, invocation);
        }

        if (string.Equals(memberReference.MemberName, "CreatePositionIds", StringComparison.Ordinal)
            || invocation.Target is IdentifierExpression { Identifier: "CreatePositionIds" })
        {
            return ExportCreatePositionIds(context);
        }

        if (string.Equals(memberReference.MemberName, "sum", StringComparison.Ordinal)
            && IsTorchReference(memberReference.Target))
        {
            return ExportTorchSum(context, invocation);
        }

        if (string.Equals(memberReference.MemberName, "matmul", StringComparison.Ordinal)
            && IsTorchReference(memberReference.Target))
        {
            return ExportTorchMatMul(context, invocation);
        }

        if (string.Equals(memberReference.MemberName, "softmax", StringComparison.Ordinal)
            && IsTorchReference(memberReference.Target))
        {
            return ExportTorchSoftmax(context, invocation);
        }

        throw new NotSupportedException($"Unsupported forward invocation: {invocation}");
    }

    private static bool IsTensorMethod(string methodName)
    {
        return methodName is "view" or "reshape" or "permute" or "transpose" or "contiguous" or "unsqueeze" or "slice";
    }

    private static ExportValue ExportTensorMethodInvocation(
        ForwardExportContext context,
        MemberReferenceExpression memberReference,
        InvocationExpression invocation
    )
    {
        var input = ExportExpression(context, memberReference.Target).GetRequiredEdge(memberReference.Target);

        return memberReference.MemberName switch
        {
            "view" or "reshape" => new ExportValue(
                context.Graph.ExportView(input, ResolveLongArguments(context, invocation.Arguments).ToArray())
            ),
            "permute" => new ExportValue(
                context.Graph.Transpose(
                    name: context.Graph.NextName("transpose"),
                    options: new TransposeInputOptions
                    {
                        Data = input,
                        Perm = ResolveLongArguments(context, invocation.Arguments).ToArray(),
                    }
                )
            ),
            "transpose" => new ExportValue(
                ExportTranspose(context, input, invocation)
            ),
            "contiguous" => new ExportValue(input),
            "unsqueeze" => new ExportValue(
                context.Graph.ExportUnsqueeze(input, ResolveLongArguments(context, invocation.Arguments).Single())
            ),
            "slice" => new ExportValue(
                context.Graph.ExportSlice(
                    input,
                    dim: ResolveLongArgument(context, invocation.Arguments.ElementAtOrDefault(0), defaultValue: 0),
                    start: ResolveLongArgument(context, invocation.Arguments.ElementAtOrDefault(1), defaultValue: 0),
                    end: ResolveLongArgument(context, invocation.Arguments.ElementAtOrDefault(2), defaultValue: long.MaxValue),
                    step: ResolveLongArgument(context, invocation.Arguments.ElementAtOrDefault(3), defaultValue: 1)
                )
            ),
            _ => throw new NotSupportedException($"Unsupported tensor method invocation: {invocation}"),
        };
    }

    private static IOnnxGraphEdge ExportTranspose(
        ForwardExportContext context,
        IOnnxGraphEdge input,
        InvocationExpression invocation
    )
    {
        var arguments = ResolveLongArguments(context, invocation.Arguments).ToArray();
        if (arguments.Length != 2)
        {
            throw new NotSupportedException($"transpose requires two dimensions: {invocation}");
        }

        // The exporter does not run full shape inference yet. GPT attention transposes rank-4
        // tensors, so rank 4 is the conservative floor when only dimension ids are visible.
        var rank = Math.Max(arguments.Max(static x => Math.Abs(x)), 0) + 1;
        rank = Math.Max(rank, 4);
        var permutation = Enumerable.Range(0, checked((int)rank)).Select(static x => (long)x).ToArray();
        var dim0 = NormalizeAxis(arguments[0], rank);
        var dim1 = NormalizeAxis(arguments[1], rank);
        (permutation[dim0], permutation[dim1]) = (permutation[dim1], permutation[dim0]);

        return context.Graph.Transpose(
            name: context.Graph.NextName("transpose"),
            options: new TransposeInputOptions
            {
                Data = input,
                Perm = permutation,
            }
        );
    }

    private static ExportValue ExportIndexer(
        ForwardExportContext context,
        IndexerExpression indexer
    )
    {
        var argument = indexer.Arguments.SingleOrDefault()
            ?? throw new NotSupportedException($"Only single-argument indexers are supported: {indexer}");

        if (indexer.Target is MemberReferenceExpression { MemberName: "shape" } shapeReference)
        {
            var dimensionIndex = Convert.ToInt64(ExportExpression(context, argument).Value);

            // Torch shape reads feed static reshape templates in the supported patterns.
            // ONNX Reshape uses 0 to copy the corresponding input dimension.
            return new ExportValue(new ShapeDimensionReference(checked((int)dimensionIndex)));
        }

        // qkv[0], qkv[1], qkv[2] in attention are tensor slices along the first axis.
        var target = ExportExpression(context, indexer.Target).GetRequiredEdge(indexer.Target);
        var index = Convert.ToInt64(ExportExpression(context, argument).Value);
        var name = context.Graph.NextName("gather");
        var indexTensor = context.Graph.AddTensor($"{name}_index", [], [index]);

        return new ExportValue(
            context.Graph.Gather(
                name: name,
                options: new GatherInputOptions
                {
                    Data = target,
                    Indices = indexTensor,
                    Axis = 0,
                }
            )
        );
    }

    private static ExportValue ExportBinaryOperator(
        ForwardExportContext context,
        BinaryOperatorExpression binaryOperator
    )
    {
        var left = ExportExpression(context, binaryOperator.Left);
        var right = ExportExpression(context, binaryOperator.Right);

        return binaryOperator.Operator switch
        {
            BinaryOperatorType.Add => new ExportValue(
                context.Graph.ExportAdd(
                    left.GetRequiredEdge(binaryOperator.Left),
                    right.GetRequiredEdge(binaryOperator.Right)
                )
            ),
            BinaryOperatorType.Multiply => new ExportValue(
                // ONNX Mul expects both operands as graph edges; scalar literals become scalar initializers.
                TryGetGraphEdge(left, out var leftEdge) && TryGetScalar(right, out var rightScalar)
                    ? context.Graph.ExportMul(leftEdge, AddScalar(context.Graph, "mul", rightScalar))
                    : TryGetGraphEdge(right, out var rightEdge) && TryGetScalar(left, out var leftScalar)
                        ? context.Graph.ExportMul(rightEdge, AddScalar(context.Graph, "mul", leftScalar))
                        : context.Graph.ExportMul(
                            left.GetRequiredEdge(binaryOperator.Left),
                            right.GetRequiredEdge(binaryOperator.Right)
                        )
            ),
            _ => throw new NotSupportedException($"Unsupported binary operator: {binaryOperator}"),
        };
    }

    private static ExportValue ExportModuleForwardCall(
        ForwardExportContext context,
        MemberReferenceExpression target,
        InvocationExpression invocation
    )
    {
        if (invocation.Arguments.Count < 1
            || invocation.Arguments.Skip(1).Any(static x => x is not NullReferenceExpression))
        {
            throw new NotSupportedException(
                $"Only module.forward(input) and module.forward(input, null, ...) calls are supported by deep export: {invocation}"
            );
        }

        // First prefer concrete module exporters; if no exporter exists, recursively decompile
        // user-defined Module<Tensor, Tensor> children such as transformer blocks.
        var torchModule = ResolveTorchModule(context.RootModule, target.Target);
        var input = ExportExpression(context, invocation.Arguments.First()).GetRequiredEdge(invocation);
        var output = InvokeModuleExport(torchModule, context.Graph, input);
        return new ExportValue(output);
    }

    private static ExportValue ExportTorchSum(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        if (invocation.Arguments.Count < 1 || invocation.Arguments.Count > 3)
        {
            throw new NotSupportedException($"Unsupported torch.sum argument count: {invocation}");
        }

        var data = ExportExpression(context, invocation.Arguments.ElementAt(0)).GetRequiredEdge(invocation);
        var axis = invocation.Arguments.Count >= 2
            ? Convert.ToInt64(ExportExpression(context, invocation.Arguments.ElementAt(1)).Value)
            : throw new NotSupportedException("Deep export requires torch.sum dimension to be specified.");

        var keepdims = invocation.Arguments.Count >= 3
            && Convert.ToBoolean(ExportExpression(context, invocation.Arguments.ElementAt(2)).Value);

        var name = context.Graph.NextName("sum");
        var output = context.Graph.ReduceSum(
            name: name,
            options: new ReduceSumInputOptions
            {
                Data = data,
                Axes = context.Graph.AddTensor<long>($"{name}_axes", [1], [axis]),
                Keepdims = keepdims ? 1 : 0,
            }
        );

        return new ExportValue(output);
    }

    private static ExportValue ExportTorchMatMul(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        if (invocation.Arguments.Count != 2)
        {
            throw new NotSupportedException($"Unsupported torch.matmul argument count: {invocation}");
        }

        return new ExportValue(
            context.Graph.MatMul(
                name: context.Graph.NextName("matmul"),
                options: new MatMulInputOptions
                {
                    A = ExportExpression(context, invocation.Arguments.ElementAt(0)).GetRequiredEdge(invocation),
                    B = ExportExpression(context, invocation.Arguments.ElementAt(1)).GetRequiredEdge(invocation),
                }
            )
        );
    }

    private static ExportValue ExportTorchSoftmax(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        if (invocation.Arguments.Count < 1 || invocation.Arguments.Count > 2)
        {
            throw new NotSupportedException($"Unsupported torch.softmax argument count: {invocation}");
        }

        var axis = invocation.Arguments.Count == 2
            ? ResolveLongArgument(context, invocation.Arguments.ElementAt(1))
            : -1;

        return new ExportValue(
            context.Graph.Softmax(
                name: context.Graph.NextName("softmax"),
                options: new SoftmaxInputOptions
                {
                    Input = ExportExpression(context, invocation.Arguments.ElementAt(0)).GetRequiredEdge(invocation),
                    Axis = axis,
                }
            )
        );
    }

    private static ExportValue ExportCreatePositionIds(ForwardExportContext context)
    {
        var maxSequenceLength = Convert.ToInt64(GetRequiredMemberValue(context.RootModule, "MaxSequenceLength"));
        var name = context.Graph.NextName("position_ids");

        // The eager method expands by batch size, but ONNX broadcasting lets a [1, seq] constant
        // feed Embedding/Gather for any runtime batch.
        return new ExportValue(
            context.Graph.AddTensor(
                name: name,
                shape: [1, maxSequenceLength],
                value: Enumerable.Range(0, checked((int)maxSequenceLength))
                    .Select(static x => (long)x)
                    .ToArray()
            )
        );
    }

    private static ExportValue ExportCausalMask(ForwardExportContext context)
    {
        var sequenceLength = Convert.ToInt64(GetRequiredMemberValue(context.RootModule, "_maxContextLength"));
        var mask = new float[sequenceLength * sequenceLength];
        for (var row = 0; row < sequenceLength; row++)
        {
            for (var column = 0; column < sequenceLength; column++)
            {
                mask[(row * sequenceLength) + column] = column <= row ? 0f : -10_000f;
            }
        }

        return new ExportValue(
            context.Graph.AddTensor(
                name: context.Graph.NextName("causal_mask"),
                shape: [1, 1, sequenceLength, sequenceLength],
                value: mask
            )
        );
    }

    private static ExportValue ExportComputeLogits(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        if (invocation.Arguments.Count != 1)
        {
            throw new NotSupportedException($"Unsupported ComputeLogits argument count: {invocation}");
        }

        // GPT output projection ties lm_head to token embedding weights. ONNX MatMul needs the
        // transposed [hidden, vocab] form as an initializer.
        var hiddenStates = ExportExpression(context, invocation.Arguments.Single()).GetRequiredEdge(invocation);
        var embedding = GetRequiredMemberValue(context.RootModule, "_tokenEmbedding");
        var weight = GetRequiredMemberValue(embedding, "weight") as global::TorchSharp.torch.Tensor
            ?? throw new NotSupportedException("ComputeLogits export requires token embedding weights.");

        var shape = weight.shape.ToArray();
        if (shape.Length != 2)
        {
            throw new NotSupportedException($"Token embedding weight must be rank-2. Got rank {shape.Length}.");
        }

        var transposedWeight = Transpose2D(
            TorchHelper.GetFloatData(weight),
            rows: checked((int)shape[0]),
            columns: checked((int)shape[1])
        );

        var name = context.Graph.NextName("lm_head");
        var initializer = context.Graph.AddTensor(
            name: $"{name}_w",
            shape: [shape[1], shape[0]],
            value: transposedWeight
        );

        return new ExportValue(
            context.Graph.MatMul(
                name: name,
                options: new MatMulInputOptions
                {
                    A = hiddenStates,
                    B = initializer,
                }
            )
        );
    }

    private static ExportValue ExportMemberReference(
        ForwardExportContext context,
        MemberReferenceExpression memberReference
    )
    {
        var target = ExportExpression(context, memberReference.Target);
        var value = target.Value
            ?? throw new NotSupportedException($"Cannot access member '{memberReference.MemberName}' on an ONNX edge.");

        if (TryGetTupleItemIndex(memberReference.MemberName, out var itemIndex))
        {
            return GetTupleElement(new ExportValue(value), itemIndex, memberReference);
        }

        return new ExportValue(GetRequiredMemberValue(value, memberReference.MemberName));
    }

    private static void AssignValue(
        ForwardExportContext context,
        Expression target,
        ExportValue value
    )
    {
        if (target is IdentifierExpression identifier)
        {
            context.Values[identifier.Identifier] = value;
            return;
        }

        if (target is IndexerExpression indexer
            && indexer.Target is IdentifierExpression targetIdentifier
            && context.Values.TryGetValue(targetIdentifier.Identifier, out var inlineArrayValue)
            && inlineArrayValue.Value is InlineArrayBuilder builder)
        {
            // Source-level collection expressions may appear as normal indexer assignments
            // when the decompiler can keep the shape literal close to C# syntax.
            var index = checked((int)ResolveLongArgument(context, indexer.Arguments.Single()));
            builder.Values[index] = value;
            return;
        }

        if (TryAssignInlineArrayElementRef(context, target, value))
        {
            return;
        }

        throw new NotSupportedException($"Unsupported assignment target: {target}");
    }

    private static bool TryAssignInlineArrayElementRef(
        ForwardExportContext context,
        Expression target,
        ExportValue value
    )
    {
        var targetText = target.ToString();
        if (!targetText.Contains("InlineArrayElementRef", StringComparison.Ordinal))
        {
            return false;
        }

        // Newer C# collection expressions can decompile to calls into PrivateImplementationDetails,
        // for example InlineArrayElementRef(ref buffer, 0) = batchSize.
        var match = Regex.Match(
            targetText,
            @"ref\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*(?<index>\d+)",
            RegexOptions.CultureInvariant
        );

        if (!match.Success)
        {
            return false;
        }

        var name = match.Groups["name"].Value;
        var index = int.Parse(match.Groups["index"].Value);
        if (context.Values.TryGetValue(name, out var inlineArrayValue)
            && inlineArrayValue.Value is InlineArrayBuilder builder)
        {
            builder.Values[index] = value;
            return true;
        }

        return false;
    }

    private static object ResolveTorchModule(
        TorchModule root,
        Expression expression
    )
    {
        return ResolveMemberExpression(root, expression);
    }

    private static object ResolveMemberExpression(
        object root,
        Expression expression
    )
    {
        return expression switch
        {
            IdentifierExpression identifier => GetRequiredMemberValue(root, identifier.Identifier),
            MemberReferenceExpression { Target: ThisReferenceExpression, MemberName: var memberName } =>
                GetRequiredMemberValue(root, memberName),
            MemberReferenceExpression memberReference =>
                GetRequiredMemberValue(ResolveMemberExpression(root, memberReference.Target), memberReference.MemberName),
            ThisReferenceExpression => root,
            _ => throw new NotSupportedException($"Unsupported module reference: {expression}"),
        };
    }

    private static object InvokeModuleExport(
        object module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        const BindingFlags FLAGS = BindingFlags.Public | BindingFlags.Static;

        var exportMethod = typeof(TorchModuleExtensions)
            .GetMethods(FLAGS)
            .Where(static x => string.Equals(x.Name, "Export", StringComparison.Ordinal))
            .Select(x => new
            {
                Method = x,
                Parameters = x.GetParameters(),
            })
            .Where(x => x.Parameters.Length == 3
                && x.Parameters[1].ParameterType == typeof(OnnxGraph)
                && x.Parameters[2].ParameterType == typeof(IOnnxGraphEdge)
                && x.Parameters[0].ParameterType.IsAssignableFrom(module.GetType())
                && x.Parameters[0].ParameterType != typeof(TorchModule))
            .OrderByDescending(x => GetInheritanceDistance(module.GetType(), x.Parameters[0].ParameterType))
            .Select(x => x.Method)
            .FirstOrDefault()
            ?? TryDeepExportModule(module, graph, input);

        if (exportMethod is null)
        {
            throw new NotSupportedException(
                $"No TorchModuleExtensions.Export overload was found for '{module.GetType().FullName}'."
            );
        }

        if (exportMethod is IOnnxGraphEdge deepExportOutput)
        {
            return deepExportOutput;
        }

        return ((MethodInfo)exportMethod).Invoke(null, [module, graph, input])
            ?? throw new InvalidOperationException(
                $"Export overload '{exportMethod}' returned null for '{module.GetType().FullName}'."
            );
    }

    private static object? TryDeepExportModule(
        object module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        // Concrete TorchSharp module types (Linear, LayerNorm, LSTM, ...) should use explicit
        // exporters. This fallback is only for user modules whose forward can be lowered.
        return module is TorchModule torchModule
            ? ExportForwardBody(torchModule, graph, input, DecompileForward(torchModule))
            : null;
    }

    private static MethodDeclaration DecompileForward(TorchModule module)
    {
        var assemblyPath = module.GetType().Assembly.Location;

        // Pin to C# 10 so collection expressions from newer compilers become explicit
        // temporaries that the inline-array builder below can recognize consistently.
        var settings = new DecompilerSettings(LanguageVersion.CSharp10_0)
        {
            ThrowOnAssemblyResolveErrors = false,
        };

        var decompiler = new CSharpDecompiler(
            fileName: assemblyPath,
            settings: settings
        );

        var method = module.GetType().GetMethod("forward")
            ?? throw new InvalidOperationException(
                $"Could not find forward method on '{module.GetType().FullName}'."
            );

        var metadataToken = MetadataTokenHelpers.EntityHandleOrNil(method.MetadataToken);
        var syntaxTree = decompiler.Decompile(metadataToken);

        return syntaxTree
            .Descendants
            .OfType<MethodDeclaration>()
            .SingleOrDefault(x => string.Equals(x.Name, method.Name, StringComparison.Ordinal))
            ?? throw new InvalidOperationException(
                $"Could not find decompiled method '{method.Name}' on '{module.GetType().FullName}'."
            );
    }

    private static IEnumerable<Statement> GetNestedStatements(Statement statement)
    {
        if (statement is BlockStatement block)
        {
            return block.Statements;
        }

        return [statement];
    }

    private static bool TryGetGraphEdge(ExportValue value, out IOnnxGraphEdge edge)
    {
        if (value.Value is IOnnxGraphEdge graphEdge)
        {
            edge = graphEdge;
            return true;
        }

        edge = null!;
        return false;
    }

    private static bool TryGetScalar(ExportValue value, out float scalar)
    {
        if (value.Value is null)
        {
            scalar = default;
            return false;
        }

        if (value.Value is IConvertible)
        {
            scalar = Convert.ToSingle(value.Value);
            return true;
        }

        scalar = default;
        return false;
    }

    private static IReadOnlyList<long> ResolveLongArguments(
        ForwardExportContext context,
        IEnumerable<Expression> arguments
    )
    {
        var argumentList = arguments.ToArray();
        if (argumentList.Length == 1)
        {
            return ResolveLongArray(context, argumentList[0]);
        }

        return argumentList
            .Select(argument => ResolveLongArgument(context, argument))
            .ToArray();
    }

    private static IReadOnlyList<long> ResolveLongArray(
        ForwardExportContext context,
        Expression expression
    )
    {
        switch (expression)
        {
            case ArrayCreateExpression arrayCreate:
                return arrayCreate.Initializer.Elements
                    .OfType<Expression>()
                    .Select(element => ResolveLongArgument(context, element))
                    .ToArray();

            case ArrayInitializerExpression arrayInitializer:
                return arrayInitializer.Elements
                    .OfType<Expression>()
                    .Select(element => ResolveLongArgument(context, element))
                    .ToArray();

            default:
                var value = ExportExpression(context, expression).Value;
                if (value is InlineArrayBuilder inlineArray)
                {
                    // Resolve compiler-generated inline arrays back into ordinary long[] shape
                    // or permutation values consumed by Reshape/Transpose wrappers.
                    return inlineArray.Values
                        .Select(item => item is null
                            ? throw new NotSupportedException($"Inline array '{expression}' has unassigned elements.")
                            : ConvertExportValueToLong(item.Value, expression))
                        .ToArray();
                }

                return [ConvertExportValueToLong(new ExportValue(value), expression)];
        }
    }

    private static long ResolveLongArgument(
        ForwardExportContext context,
        Expression? expression,
        long defaultValue = 0
    )
    {
        if (expression is null)
        {
            return defaultValue;
        }

        var value = ExportExpression(context, expression).Value;
        return ConvertExportValueToLong(new ExportValue(value), expression, defaultValue);
    }

    private static long ConvertExportValueToLong(
        ExportValue value,
        AstNode source,
        long defaultValue = 0
    )
    {
        return value switch
        {
            // ShapeDimensionReference represents x.shape[i]. In supported reshape templates,
            // a 0 means "copy this dimension from the input" in ONNX Reshape.
            { Value: ShapeDimensionReference } => 0,
            { Value: null } => defaultValue,
            { Value: IConvertible convertible } => Convert.ToInt64(convertible),
            _ => throw new NotSupportedException($"Expression '{source}' did not produce an integer value."),
        };
    }

    private static int NormalizeAxis(long axis, long rank)
    {
        var normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank)
        {
            throw new NotSupportedException($"Axis {axis} is outside rank {rank}.");
        }

        return checked((int)normalized);
    }

    private static IOnnxGraphEdge AddScalar(
        OnnxGraph graph,
        string prefix,
        float value
    )
    {
        return graph.AddTensor(
            name: graph.NextName($"{prefix}_scalar"),
            shape: [],
            value: [value]
        );
    }

    private static bool TryCreateInlineArrayBuilder(
        DefaultValueExpression expression,
        out InlineArrayBuilder builder
    )
    {
        var typeText = expression.Type.ToString();
        var match = Regex.Match(typeText, @"InlineArray(?<length>\d+)<", RegexOptions.CultureInvariant);
        if (match.Success && int.TryParse(match.Groups["length"].Value, out var length))
        {
            builder = new InlineArrayBuilder(length);
            return true;
        }

        builder = null!;
        return false;
    }

    private static float[] Transpose2D(
        float[] input,
        int rows,
        int columns
    )
    {
        var output = new float[input.Length];
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                output[(column * rows) + row] = input[(row * columns) + column];
            }
        }

        return output;
    }

    private static int GetInheritanceDistance(Type type, Type candidate)
    {
        if (type == candidate)
        {
            return int.MaxValue;
        }

        var distance = 0;
        var current = type;
        while (current is not null)
        {
            if (current == candidate)
            {
                return int.MaxValue - distance;
            }

            current = current.BaseType;
            distance++;
        }

        return 0;
    }

    private static ExportValue GetTupleElement(
        ExportValue value,
        int index,
        AstNode source
    )
    {
        if (value.Value is null)
        {
            throw new NotSupportedException($"Cannot deconstruct ONNX edge result from: {source}");
        }

        var memberName = index switch
        {
            0 => "Y",
            1 => "YH",
            2 => "YC",
            _ => $"Item{index + 1}",
        };

        if (TryGetMemberValue(value.Value, memberName, out var memberValue))
        {
            return new ExportValue(memberValue);
        }

        if (value.Value is ITuple tuple && index < tuple.Length)
        {
            return new ExportValue(tuple[index]);
        }

        throw new NotSupportedException($"Cannot read deconstruction item {index} from '{value.Value.GetType().FullName}'.");
    }

    private static bool TryGetTupleItemIndex(string memberName, out int index)
    {
        if (memberName.Length > 4
            && memberName.StartsWith("Item", StringComparison.Ordinal)
            && int.TryParse(memberName[4..], out var oneBasedIndex)
            && oneBasedIndex > 0)
        {
            index = oneBasedIndex - 1;
            return true;
        }

        index = -1;
        return false;
    }

    private static object GetRequiredMemberValue(object instance, string name)
    {
        if (TryGetMemberValue(instance, name, out var value))
        {
            return value!;
        }

        throw new NotSupportedException(
            $"Member '{name}' was not found on '{instance.GetType().FullName}'."
        );
    }

    private static bool TryGetMemberValue(object instance, string name, out object? value)
    {
        const BindingFlags FLAGS = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        var type = instance.GetType();
        var property = type.GetProperty(name, FLAGS);
        if (property is not null)
        {
            value = property.GetValue(instance);
            return true;
        }

        var field = type.GetField(name, FLAGS);
        if (field is not null)
        {
            value = field.GetValue(instance);
            return true;
        }

        value = null;
        return false;
    }

    private static bool IsTorchReference(Expression expression)
    {
        return string.Equals(expression.ToString(), "torch", StringComparison.Ordinal)
            || expression.ToString().EndsWith(".torch", StringComparison.Ordinal);
    }

    private sealed class ForwardExportContext(
        TorchModule rootModule,
        OnnxGraph graph
    )
    {
        public TorchModule RootModule { get; } = rootModule;

        public OnnxGraph Graph { get; } = graph;

        public Dictionary<string, ExportValue> Values { get; } = new(StringComparer.Ordinal);
    }

    private readonly record struct ExportValue(object? Value)
    {
        public IOnnxGraphEdge GetRequiredEdge(AstNode source)
        {
            return Value as IOnnxGraphEdge
                ?? throw new NotSupportedException(
                    $"Expression '{source}' did not produce an ONNX graph edge."
                );
        }
    }

    private readonly record struct ShapeDimensionReference(int Index);

    private sealed class InlineArrayBuilder(int length)
    {
        public ExportValue?[] Values { get; } = new ExportValue?[length];
    }
}
