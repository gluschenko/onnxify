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

        ExportValue? result = null;
        foreach (var statement in forward.Body.Statements)
        {
            result = ExportStatement(context, statement) ?? result;
        }

        return result?.GetRequiredEdge(forward)
            ?? throw new NotSupportedException(
                $"Method '{forward.Name}' did not return a supported ONNX graph edge."
            );
    }

    private static ExportValue? ExportStatement(
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
                return ExportExpression(context, returnStatement.Expression);

            default:
                throw new NotSupportedException(
                    $"Unsupported forward statement '{statement.GetType().Name}': {statement}"
                );
        }
    }

    private static ExportValue? ExportUsingStatement(
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

        ExportValue? result = null;
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

            context.Values[variable.Name] = ExportExpression(context, variable.Initializer);
        }
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

            case UnaryOperatorExpression unaryOperator
                when unaryOperator.Operator == UnaryOperatorType.NullConditionalRewrap:
                return ExportExpression(context, unaryOperator.Expression);

            case IndexerExpression indexer:
                return ExportIndexer(context, indexer);

            case ParenthesizedExpression parenthesized:
                return ExportExpression(context, parenthesized.Expression);

            case CastExpression cast:
                return ExportExpression(context, cast.Expression);

            case PrimitiveExpression primitive:
                return new ExportValue(primitive.Value);

            case NullReferenceExpression:
                return new ExportValue(null);

            case ThrowExpression:
                throw new NotSupportedException($"Expression '{expression}' threw while being evaluated for export.");

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
        if (invocation.Target is IdentifierExpression identifier
            && TryExportLocalMethodInvocation(context, identifier.Identifier, invocation, out var localResult))
        {
            return localResult;
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

        if (string.Equals(memberReference.MemberName, "arange", StringComparison.Ordinal)
            && IsTorchReference(memberReference.Target))
        {
            return ExportTorchArange(context, invocation);
        }

        if (string.Equals(memberReference.MemberName, "full", StringComparison.Ordinal)
            && IsTorchReference(memberReference.Target))
        {
            return ExportTorchFull(context, invocation);
        }

        if (string.Equals(memberReference.MemberName, "triu", StringComparison.Ordinal)
            && IsTorchReference(memberReference.Target))
        {
            return ExportTorchTriu(context, invocation);
        }

        throw new NotSupportedException($"Unsupported forward invocation: {invocation}");
    }

    private static bool IsTensorMethod(string methodName)
    {
        return methodName is "view" or "reshape" or "permute" or "transpose" or "contiguous" or "unsqueeze" or "slice" or "expand";
    }

    private static ExportValue ExportTensorMethodInvocation(
        ForwardExportContext context,
        MemberReferenceExpression memberReference,
        InvocationExpression invocation
    )
    {
        var target = ExportExpression(context, memberReference.Target);
        if (target.Value is global::TorchSharp.torch.Tensor tensor)
        {
            return ExportRuntimeTensorMethod(context, tensor, memberReference.MemberName, invocation);
        }

        var input = target.GetRequiredEdge(memberReference.Target);

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
                ExportUnsqueeze(context, input, ResolveLongArguments(context, invocation.Arguments).Single())
            ),
            "expand" => new ExportValue(
                context.Graph.ExportExpand(input, ResolveLongArguments(context, invocation.Arguments).ToArray())
            ),
            "slice" => new ExportValue(
                ExportSlice(
                    context,
                    input,
                    dim: ResolveLongArgument(context, invocation.Arguments.ElementAtOrDefault(0), defaultValue: 0),
                    start: ResolveLongArgument(context, invocation.Arguments.ElementAtOrDefault(1), defaultValue: 0),
                    end: ResolveSliceEnd(context, input, invocation),
                    step: ResolveLongArgument(context, invocation.Arguments.ElementAtOrDefault(3), defaultValue: 1)
                )
            ),
            _ => throw new NotSupportedException($"Unsupported tensor method invocation: {invocation}"),
        };
    }

    private static ExportValue ExportRuntimeTensorMethod(
        ForwardExportContext context,
        global::TorchSharp.torch.Tensor tensor,
        string methodName,
        InvocationExpression invocation
    )
    {
        using var result = methodName switch
        {
            "transpose" => tensor.transpose(
                ResolveLongArgument(context, invocation.Arguments.ElementAtOrDefault(0)),
                ResolveLongArgument(context, invocation.Arguments.ElementAtOrDefault(1))
            ),
            "contiguous" => tensor.contiguous(),
            _ => throw new NotSupportedException($"Runtime tensor method '{methodName}' is not supported: {invocation}"),
        };

        return new ExportValue(AddTensorInitializer(context.Graph, context.Graph.NextName(methodName), result));
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

        // The exporter does not run full shape inference yet. Multi-head attention commonly
        // transposes rank-4 tensors, so rank 4 is the conservative floor when only axis ids exist.
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

    private static IOnnxGraphEdge ExportUnsqueeze(
        ForwardExportContext context,
        IOnnxGraphEdge input,
        long axis
    )
    {
        if (axis < 0 && input is OnnxTensor tensor)
        {
            axis += tensor.Shape.Length + 1;
        }

        var name = context.Graph.NextName("unsqueeze");
        var axes = context.Graph.AddTensor<long>(
            name: $"{name}_axes",
            shape: [1],
            value: [axis]
        );

        return context.Graph.Unsqueeze(
            name: name,
            options: new UnsqueezeInputOptions
            {
                Data = input,
                Axes = axes,
            }
        );
    }

    private static IOnnxGraphEdge ExportSlice(
        ForwardExportContext context,
        IOnnxGraphEdge input,
        long dim,
        long start,
        long end,
        long step
    )
    {
        if (step == 0)
        {
            throw new NotSupportedException("slice export does not support step = 0.");
        }

        var axis = input is OnnxTensor tensor
            ? NormalizeAxis(dim, tensor.Shape.Length)
            : dim;
        var name = context.Graph.NextName("slice");

        return context.Graph.Slice(
            name: name,
            options: new SliceInputOptions
            {
                Data = input,
                Starts = context.Graph.AddTensor<long>($"{name}_starts", [1], [start]),
                Ends = context.Graph.AddTensor<long>($"{name}_ends", [1], [end]),
                Axes = context.Graph.AddTensor<long>($"{name}_axes", [1], [axis]),
                Steps = context.Graph.AddTensor<long>($"{name}_steps", [1], [step]),
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
        if (binaryOperator.Operator == BinaryOperatorType.NullCoalescing)
        {
            var value = ExportExpression(context, binaryOperator.Left);
            return value.Value is not null ? value : ExportExpression(context, binaryOperator.Right);
        }

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

    private static ExportValue ExportTorchArange(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        var arguments = GetLongArguments(context, invocation).ToArray();
        if (arguments.Length is < 1 or > 3)
        {
            throw new NotSupportedException($"Unsupported torch.arange argument count: {invocation}");
        }

        var start = arguments.Length == 1 ? 0 : arguments[0];
        var end = arguments.Length == 1
            ? arguments[0]
            : arguments[1];
        var step = arguments.Length >= 3 ? arguments[2] : 1;
        if (step == 0)
        {
            throw new NotSupportedException($"torch.arange step must not be 0: {invocation}");
        }

        var values = new List<long>();
        for (var value = start; step > 0 ? value < end : value > end; value += step)
        {
            values.Add(value);
        }

        return new ExportValue(
            context.Graph.AddTensor(
                name: context.Graph.NextName("arange"),
                shape: [values.Count],
                value: values.ToArray()
            )
        );
    }

    private static ExportValue ExportTorchFull(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        var arguments = GetPositionalArguments(invocation).ToArray();
        if (arguments.Length < 2)
        {
            throw new NotSupportedException($"Unsupported torch.full argument count: {invocation}");
        }

        var shape = ResolveLongArray(context, arguments[0]).ToArray();
        var elementCount = shape.Aggregate(1L, checked((total, dimension) => total * dimension));
        var fillValue = Convert.ToSingle(ExportExpression(context, arguments[1]).Value);
        var values = Enumerable.Repeat(fillValue, checked((int)elementCount)).ToArray();

        return new ExportValue(
            context.Graph.AddTensor(
                name: context.Graph.NextName("full"),
                shape: shape,
                value: values
            )
        );
    }

    private static ExportValue ExportTorchTriu(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        var arguments = GetPositionalArguments(invocation).ToArray();
        if (arguments.Length is < 1 or > 2)
        {
            throw new NotSupportedException($"Unsupported torch.triu argument count: {invocation}");
        }

        var input = ExportExpression(context, arguments[0]).Value;
        if (input is not OnnxTensor tensor || tensor.DataType != typeof(float) || tensor.Shape.Length != 2)
        {
            var inputDescription = input is OnnxTensor actualTensor
                ? $"{input.GetType().FullName}, dtype={actualTensor.DataType.FullName}, rank={actualTensor.Shape.Length}"
                : input?.GetType().FullName ?? "<null>";
            throw new NotSupportedException(
                $"torch.triu currently requires a rank-2 float initializer, got {inputDescription}: {invocation}"
            );
        }

        var diagonal = GetNamedArgument(invocation, "diagonal") is { } namedDiagonal
            ? ResolveLongArgument(context, namedDiagonal)
            : arguments.Length == 2
                ? ResolveLongArgument(context, arguments[1])
                : 0;

        var rows = checked((int)tensor.Shape[0]);
        var columns = checked((int)tensor.Shape[1]);
        var values = GetFloatTensorValues(tensor);
        var output = new float[values.Length];
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                output[(row * columns) + column] = column - row >= diagonal
                    ? values[(row * columns) + column]
                    : 0f;
            }
        }

        return new ExportValue(
            context.Graph.AddTensor(
                name: context.Graph.NextName("triu"),
                shape: tensor.Shape,
                value: output
            )
        );
    }

    private static bool TryExportLocalMethodInvocation(
        ForwardExportContext context,
        string methodName,
        InvocationExpression invocation,
        out ExportValue result
    )
    {
        const BindingFlags FLAGS = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        var arguments = GetPositionalArguments(invocation).ToArray();
        var method = context.RootModule.GetType()
            .GetMethods(FLAGS)
            .Where(x => string.Equals(x.Name, methodName, StringComparison.Ordinal))
            .FirstOrDefault(x => x.GetParameters().Length == arguments.Length);

        if (method is null)
        {
            result = default;
            return false;
        }

        if (method.ReturnType == typeof(void) && IsValidationMethodName(method.Name))
        {
            result = new ExportValue(null);
            return true;
        }

        var declaration = DecompileMethod(context.RootModule, method);
        var nestedContext = new ForwardExportContext(context.RootModule, context.Graph);
        foreach (var (name, value) in context.Values)
        {
            nestedContext.Values[name] = value;
        }

        var parameters = declaration.Parameters.ToArray();
        for (var index = 0; index < parameters.Length; index++)
        {
            nestedContext.Values[parameters[index].Name] = ExportExpression(context, arguments[index]);
        }

        result = ExportMethodBody(nestedContext, declaration);
        return true;
    }

    private static ExportValue ExportMethodBody(
        ForwardExportContext context,
        MethodDeclaration method
    )
    {
        ExportValue? result = null;
        foreach (var statement in method.Body.Statements)
        {
            result = ExportStatement(context, statement) ?? result;
        }

        return result
            ?? throw new NotSupportedException(
                $"Method '{method.Name}' did not return a supported export value."
            );
    }

    private static ExportValue ExportMemberReference(
        ForwardExportContext context,
        MemberReferenceExpression memberReference
    )
    {
        var target = ExportExpression(context, memberReference.Target);
        if (target.Value is IOnnxGraphEdge)
        {
            if (memberReference.MemberName is "dtype" or "device")
            {
                return new ExportValue(new SymbolicTensorMember(memberReference.MemberName));
            }

            throw new NotSupportedException($"Cannot access member '{memberReference.MemberName}' on an ONNX edge.");
        }

        var value = target.Value
            ?? throw new NotSupportedException($"Cannot access member '{memberReference.MemberName}' on a null value.");

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
        var method = module.GetType().GetMethod("forward")
            ?? throw new InvalidOperationException(
                $"Could not find forward method on '{module.GetType().FullName}'."
            );

        return DecompileMethod(module, method);
    }

    private static MethodDeclaration DecompileMethod(
        TorchModule module,
        MethodInfo method
    )
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

    private static OnnxTensor AddTensorInitializer(
        OnnxGraph graph,
        string name,
        global::TorchSharp.torch.Tensor tensor
    )
    {
        var shape = tensor.GetShape();
        var detached = tensor.detach().cpu();
        return tensor.dtype switch
        {
            global::TorchSharp.torch.ScalarType.Int64 => graph.AddTensor(
                name: name,
                shape: shape,
                value: detached.data<long>().ToArray()
            ),
            global::TorchSharp.torch.ScalarType.Float32 => graph.AddTensor(
                name: name,
                shape: shape,
                value: detached.data<float>().ToArray()
            ),
            _ => throw new NotSupportedException(
                $"Runtime tensor initializer export does not support dtype '{tensor.dtype}'."
            ),
        };
    }

    private static float[] GetFloatTensorValues(OnnxTensor tensor)
    {
        return tensor is OnnxTensor<float> typedTensor
            ? typedTensor.Value.ToArray()
            : throw new NotSupportedException($"Expected float initializer '{tensor.Name}'.");
    }

    private static long ResolveSliceEnd(
        ForwardExportContext context,
        IOnnxGraphEdge input,
        InvocationExpression invocation
    )
    {
        var endExpression = invocation.Arguments.ElementAtOrDefault(2);
        if (endExpression is null)
        {
            return long.MaxValue;
        }

        var end = ExportExpression(context, UnwrapNamedArgument(endExpression));
        if (end.Value is ShapeDimensionReference
            && input is OnnxTensor tensor
            && invocation.Arguments.ElementAtOrDefault(0) is { } dimExpression)
        {
            var dim = ResolveLongArgument(context, dimExpression);
            return tensor.Shape[NormalizeAxis(dim, tensor.Shape.Length)];
        }

        return ConvertExportValueToLong(end, endExpression, long.MaxValue);
    }

    private static IEnumerable<Expression> GetPositionalArguments(InvocationExpression invocation)
    {
        return invocation.Arguments
            .OfType<Expression>()
            .Where(static x => x is not NamedArgumentExpression)
            .Select(UnwrapNamedArgument);
    }

    private static IEnumerable<long> GetLongArguments(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        var result = new List<long>();
        foreach (var argument in invocation.Arguments.OfType<Expression>())
        {
            try
            {
                result.Add(ResolveLongArgument(context, argument));
            }
            catch (NotSupportedException)
            {
                // Named dtype/device arguments decompile differently across compiler versions.
                // Constant-only torch factories can ignore those metadata hints during export.
            }
        }

        return result;
    }

    private static Expression? GetNamedArgument(
        InvocationExpression invocation,
        string name
    )
    {
        return invocation.Arguments
            .OfType<NamedArgumentExpression>()
            .FirstOrDefault(x => string.Equals(x.Name, name, StringComparison.Ordinal))
            ?.Expression;
    }

    private static Expression UnwrapNamedArgument(Expression expression)
    {
        return expression is NamedArgumentExpression namedArgument
            ? namedArgument.Expression
            : expression;
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
        expression = UnwrapNamedArgument(expression);

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

            case InvocationExpression invocation
                when TryResolveInlineArraySpan(context, invocation, out var values):
                return values;

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

    private static bool TryResolveInlineArraySpan(
        ForwardExportContext context,
        InvocationExpression invocation,
        out IReadOnlyList<long> values
    )
    {
        var text = invocation.ToString();
        if (!text.Contains("InlineArrayAsReadOnlySpan", StringComparison.Ordinal))
        {
            values = [];
            return false;
        }

        var match = Regex.Match(
            text,
            @"in\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*(?<length>\d+)",
            RegexOptions.CultureInvariant
        );
        if (!match.Success)
        {
            values = [];
            return false;
        }

        var name = match.Groups["name"].Value;
        if (!context.Values.TryGetValue(name, out var builderValue)
            || builderValue.Value is not InlineArrayBuilder builder)
        {
            values = [];
            return false;
        }

        values = builder.Values
            .Select(item => item is null
                ? throw new NotSupportedException($"Inline array '{name}' has unassigned elements.")
                : ConvertExportValueToLong(item.Value, invocation))
            .ToArray();
        return true;
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

        expression = UnwrapNamedArgument(expression);
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

    private readonly record struct SymbolicTensorMember(string Name);

    private sealed class InlineArrayBuilder(int length)
    {
        public ExportValue?[] Values { get; } = new ExportValue?[length];
    }
}
