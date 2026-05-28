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
    /// <summary>
    /// Exports a single-input TorchSharp module to an ONNX model by analyzing the module's
    /// decompiled <c>forward</c> method and synthesizing an equivalent inference graph.
    /// </summary>
    /// <param name="module">
    /// The TorchSharp <see cref="TorchModule"/> instance whose <c>forward(Tensor)</c> method
    /// should be exported.
    /// </param>
    /// <param name="input">
    /// The ONNX type metadata for the exported model input. The exporter uses this metadata
    /// as the graph input contract; it does not infer input shape or dtype by running the module.
    /// </param>
    /// <param name="output">
    /// The ONNX type metadata for the exported model output.
    /// </param>
    /// <param name="options">
    /// Model creation options, including the target opset and producer metadata.
    /// </param>
    /// <returns>
    /// A new <see cref="OnnxModel"/> whose graph contains the declared input, declared output,
    /// and the ONNX nodes produced from the supported tensor operations in <c>forward</c>.
    /// </returns>
    /// <remarks>
    /// The exporter decompiles <c>forward</c> with ICSharpCode.Decompiler, walks the resulting
    /// C# syntax tree as a small data-flow program, and maps local variables, assignments,
    /// returns, tensor method calls, static <c>torch</c> calls, scalar arithmetic, and supported
    /// module <c>forward</c> calls into ONNX graph edges and initializers.
    /// </remarks>
    /// <remarks>
    /// Calls to known TorchSharp modules are delegated to the ordinary
    /// <c>TorchModuleExtensions.Export</c> overloads. User-defined child modules are recursively
    /// deep-exported when no concrete overload exists. Unsupported control flow or expressions
    /// fail with <see cref="NotSupportedException"/> rather than emitting a lossy graph.
    /// </remarks>
    public static OnnxModel ExportOnnxModel(
        this TorchModule module,
        OnnxTensorType input,
        OnnxTensorType output,
        OnnxModelCreationOptions options
    )
    {
        return ExportOnnxModel(
            module: module,
            inputName: "input",
            outputName: "output",
            input: input,
            output: output,
            options: options
        );
    }

    /// <summary>
    /// Exports a single-input TorchSharp module to an ONNX model by analyzing the module's
    /// decompiled <c>forward</c> method and synthesizing an equivalent inference graph.
    /// </summary>
    /// <param name="module">
    /// The TorchSharp <see cref="TorchModule"/> instance whose <c>forward(Tensor)</c> method
    /// should be exported.
    /// </param>
    /// <param name="input">
    /// The ONNX type metadata for the exported model input. The exporter uses this metadata
    /// as the graph input contract; it does not infer input shape or dtype by running the module.
    /// </param>
    /// <param name="output">
    /// The ONNX type metadata for the exported model output.
    /// </param>
    /// <param name="options">
    /// Model creation options, including the target opset and producer metadata.
    /// </param>
    /// <returns>
    /// A new <see cref="OnnxModel"/> whose graph contains the declared input, declared output,
    /// and the ONNX nodes produced from the supported tensor operations in <c>forward</c>.
    /// </returns>
    /// <remarks>
    /// The exporter decompiles <c>forward</c> with ICSharpCode.Decompiler, walks the resulting
    /// C# syntax tree as a small data-flow program, and maps local variables, assignments,
    /// returns, tensor method calls, static <c>torch</c> calls, scalar arithmetic, and supported
    /// module <c>forward</c> calls into ONNX graph edges and initializers.
    /// </remarks>
    /// <remarks>
    /// Calls to known TorchSharp modules are delegated to the ordinary
    /// <c>TorchModuleExtensions.Export</c> overloads. User-defined child modules are recursively
    /// deep-exported when no concrete overload exists. Unsupported control flow or expressions
    /// fail with <see cref="NotSupportedException"/> rather than emitting a lossy graph.
    /// </remarks>
    public static OnnxModel ExportOnnxModel(
        this TorchModule module,
        string inputName,
        string outputName,
        OnnxTensorType input,
        OnnxTensorType output,
        OnnxModelCreationOptions options
    )
    {
        ArgumentNullException.ThrowIfNullOrWhiteSpace(inputName);
        ArgumentNullException.ThrowIfNullOrWhiteSpace(outputName);
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(options);

        var onnxModel = OnnxModel.Create(options);
        var graph = onnxModel.Graph;

        var graphInputName = graph.NextName(inputName);
        var graphOutputName = graph.NextName(outputName);

        var inputEdge = graph.AddInput(graphInputName, input);
        graph.AddOutput(graphOutputName, output);

        var forward = DecompileForward(module);

        // Interpret the decompiled forward body as a small data-flow program.
        // Only tensor-producing inference patterns are lowered; unsupported C# fails loudly.
        var result = ExportForwardBody(module, graph, inputEdge, forward);
        var outputEdge = graph.AddEdge(graphOutputName);

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
        // Examples:
        //   forward(Tensor input)
        //   forward(Tensor tokens)
        context.Values["input"] = new ExportValue(input);

        foreach (var parameter in forward.Parameters)
        {
            context.Values.TryAdd(parameter.Name, new ExportValue(input));
        }

        ExportValue? result = null;
        foreach (var statement in forward.Body.Statements)
        {
            result = ExportStatement(context, statement) ?? result;
            if (context.HasReturned)
            {
                break;
            }
        }

        return (context.ReturnValue ?? result)?.GetRequiredEdge(forward)
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
            // Handles local bindings from plain declarations and using-statement resources:
            //   var x = _embedding.forward(input);
            //   using var ids = torch.arange(...).unsqueeze(0);
            case VariableDeclarationStatement variableDeclaration:
                ExportVariableDeclaration(context, variableDeclaration);
                return null;

            // Handles rebinding a graph value, e.g.:
            //   x = _block.forward(x);
            //   attentionScores = attentionScores + causalMask;
            case ExpressionStatement { Expression: AssignmentExpression assignment }:
                AssignValue(context, assignment.Left, ExportExpression(context, assignment.Right));
                return null;

            // Handles side-effect-shaped calls, usually validation guards:
            //   ValidateInputShape(tokens);
            case ExpressionStatement { Expression: InvocationExpression invocation }:
                ExportStatementInvocation(context, invocation);
                return null;

            // Decompiler emits "using var tensor = ..." as a UsingStatement that contains
            // the remaining forward body, e.g. "using var mask = ...; return input + mask;".
            // Disposal is irrelevant for the synthesized ONNX graph.
            case UsingStatement usingStatement:
                return ExportUsingStatement(context, usingStatement);

            // Handles final graph outputs and helper returns:
            //   return _linear.forward(x);
            //   return matmul(hiddenStates, tiedWeight);
            case ReturnStatement returnStatement:
                return context.Return(ExportExpression(context, returnStatement.Expression));

            // Release and platform-specific decompilation can rewrite a conditional return:
            //   return _useAdd ? input + 1f : input * 2f;
            // into:
            //   if (!_useAdd) { return input * 2f; }
            //   return input + 1f;
            // The condition still has to be resolvable from module state; dynamic branches
            // remain unsupported.
            case IfElseStatement ifElseStatement:
                return ExportIfElseStatement(context, ifElseStatement);

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
        // "using var" resources are usually temporary tensors:
        //   using var positions = CreatePositionIds(...);
        //   using var causalMask = triu(...).slice(...);
        if (usingStatement.ResourceAcquisition is VariableDeclarationStatement declaration)
        {
            ExportVariableDeclaration(context, declaration);
        }
        else if (usingStatement.ResourceAcquisition is Expression resourceExpression)
        {
            // Handles less common decompiled forms where the resource is an expression:
            //   using (CreateMask(...)) { ... }
            ExportExpression(context, resourceExpression);
        }

        ExportValue? result = null;
        foreach (var nestedStatement in GetNestedStatements(usingStatement.EmbeddedStatement))
        {
            result = ExportStatement(context, nestedStatement) ?? result;
            if (context.HasReturned)
            {
                break;
            }
        }

        return result;
    }

    private static ExportValue? ExportIfElseStatement(
        ForwardExportContext context,
        IfElseStatement ifElseStatement
    )
    {
        if (!TryEvaluateBooleanExpression(context, ifElseStatement.Condition, out var condition))
        {
            throw new NotSupportedException(
                $"If statement condition must be statically resolvable during export: {ifElseStatement.Condition}"
            );
        }

        var selectedStatement = condition
            ? ifElseStatement.TrueStatement
            : ifElseStatement.FalseStatement;

        if (selectedStatement.IsNull)
        {
            return null;
        }

        ExportValue? result = null;
        foreach (var nestedStatement in GetNestedStatements(selectedStatement))
        {
            result = ExportStatement(context, nestedStatement) ?? result;
            if (context.HasReturned)
            {
                break;
            }
        }

        return result;
    }

    private static void ExportStatementInvocation(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        // Runtime validation guards such as "ValidateInputShape(tokens);" affect eager
        // execution only. The exported model already gets its input contract from the
        // caller-provided OnnxTensorType.
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
        // Each declared variable becomes a named data-flow slot:
        //   var query = qkv[0];
        //   var batchSize = x.shape[0];
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
        // Tuple deconstruction may survive as source syntax:
        //   var (lstm, _, _) = _lstm.forward(x);
        // Recurrent exporters return wrapper objects such as LSTMOutput, so tuple slots are
        // mapped to object members like Y, YH, and YC.
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
            // Local values and decompiled fields both arrive as identifiers:
            //   x
            //   _scale
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

            // Handles statically decidable ternaries produced by optional module branches:
            //   _pixelUnshuffle == null ? input : _pixelUnshuffle.forward(input)
            //   _optional != null ? _optional.forward(input) : input
            // Only the selected branch is exported, so nullable inactive branches may stay unsupported.
            case ConditionalExpression conditional:
                return ExportConditionalExpression(context, conditional);

            // ILSpy sometimes wraps null-flow expressions in NullConditionalRewrap even when
            // the source was just a nullable-suppressed member access such as "weight!".
            case UnaryOperatorExpression unaryOperator
                when unaryOperator.Operator == UnaryOperatorType.NullConditionalRewrap:
                return ExportExpression(context, unaryOperator.Expression);

            case UnaryOperatorExpression unaryOperator:
                return ExportUnaryOperator(context, unaryOperator);

            // Handles tensor indexing and shape indexing:
            //   qkv[0]
            //   input.shape[1]
            case IndexerExpression indexer:
                return ExportIndexer(context, indexer);

            // Parentheses and casts should not change the export meaning:
            //   (batchSize)
            //   (long)sequenceLength
            case ParenthesizedExpression parenthesized:
                return ExportExpression(context, parenthesized.Expression);

            case CastExpression cast:
                return ExportExpression(context, cast.Expression);

            // Handles scalar literals used by shape math and tensor arithmetic:
            //   1
            //   -10_000f
            case PrimitiveExpression primitive:
                return new ExportValue(primitive.Value);

            case NullReferenceExpression:
                return new ExportValue(null);

            case ThrowExpression:
                throw new NotSupportedException($"Expression '{expression}' threw while being evaluated for export.");

            // C# collection expressions like ".view([b, s, h])" can decompile into
            // compiler-generated InlineArray temporaries. Represent the temporary as a
            // mutable shape/permutation builder until subsequent assignments fill it.
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
        // Handles private helpers inside the model:
        //   ComputeLogits(x)
        //   CreatePositionIds(tokens.shape[0], tokens.device)
        if (invocation.Target is IdentifierExpression identifier
            && TryExportLocalMethodInvocation(context, identifier.Identifier, invocation, out var localResult))
        {
            return localResult;
        }

        if (invocation.Target is IdentifierExpression torchIdentifier
            && IsTorchConcatName(torchIdentifier.Identifier))
        {
            return ExportTorchConcat(context, invocation);
        }

        if (invocation.Target is not MemberReferenceExpression memberReference)
        {
            throw new NotSupportedException($"Unsupported invocation target: {invocation}");
        }

        if (string.Equals(memberReference.MemberName, "forward", StringComparison.Ordinal))
        {
            return ExportModuleForwardCall(context, memberReference, invocation);
        }

        // Tensor instance methods such as "x.view(...)" and "mask.unsqueeze(0)" are
        // lowered here because they are not nn.Module calls and therefore do not go
        // through TorchModuleExtensions.Export dispatch.
        if (IsTensorMethod(memberReference.MemberName))
        {
            return ExportTensorMethodInvocation(context, memberReference, invocation);
        }

        // Static torch functions decompile as member calls on "torch":
        //   torch.matmul(query, key)
        //   torch.arange(maxLength, dtype: ..., device: ...)
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

        if (string.Equals(memberReference.MemberName, "cat", StringComparison.Ordinal)
            && IsTorchReference(memberReference.Target))
        {
            return ExportTorchConcat(context, invocation);
        }

        if (string.Equals(memberReference.MemberName, "interpolate", StringComparison.Ordinal)
            && IsTorchFunctionalReference(memberReference.Target))
        {
            return ExportTorchInterpolate(context, invocation);
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
            // Constant tensor methods are executed once during export:
            //   _tokenEmbedding.weight!.transpose(0, 1)
            return ExportRuntimeTensorMethod(context, tensor, memberReference.MemberName, invocation);
        }

        var input = target.GetRequiredEdge(memberReference.Target);

        return memberReference.MemberName switch
        {
            // Dynamic tensor methods become ONNX graph operators:
            //   x.view([batchSize, sequenceLength, hidden])
            //   qkv.permute(2, 0, 3, 1, 4)
            //   mask.slice(0, 0, sequenceLength, 1)
            "view" or "reshape" => new ExportValue(
                context.Graph.ExportReshape(input, ResolveLongArguments(context, invocation.Arguments).ToArray())
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
            var shapeTarget = ExportExpression(context, shapeReference.Target);
            if (shapeTarget.Value is global::TorchSharp.torch.Tensor runtimeTensor)
            {
                return new ExportValue(runtimeTensor.shape[checked((int)dimensionIndex)]);
            }

            // Torch shape reads such as "x.shape[0]" feed static reshape templates in
            // the supported patterns.
            // ONNX Reshape uses 0 to copy the corresponding input dimension.
            return new ExportValue(new ShapeDimensionReference(shapeTarget.Value as IOnnxGraphEdge, checked((int)dimensionIndex)));
        }

        // Tensor indexers such as "qkv[0]", "qkv[1]", "qkv[2]" in attention are
        // exported as Gather along the first axis.
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
            // Handles null fallbacks preserved by decompilation:
            //   weight ?? throw new InvalidOperationException(...)
            var value = ExportExpression(context, binaryOperator.Left);
            return value.Value is not null ? value : ExportExpression(context, binaryOperator.Right);
        }

        if (TryEvaluateBooleanExpression(context, binaryOperator, out var booleanResult))
        {
            return new ExportValue(booleanResult);
        }

        var left = ExportExpression(context, binaryOperator.Left);
        var right = ExportExpression(context, binaryOperator.Right);
        if (TryFoldNumericBinary(binaryOperator.Operator, left, right, out var folded))
        {
            // Fold scalar-only shape/math expressions before they reach ONNX:
            //   ((_base + 2L) * 2L) - 3L
            //   (((_floatBase + 2f) * 4f) - 6f) / 3f
            return folded;
        }

        return binaryOperator.Operator switch
        {
            // Tensor arithmetic accepts tensor-tensor and tensor-scalar forms:
            //   input + mask
            //   attentionScores * _scale
            //   (input + offset) / divisor
            BinaryOperatorType.Add => new ExportValue(
                TryGetGraphEdge(left, out var leftEdge) && TryGetScalar(right, out var rightScalar)
                    ? context.Graph.ExportAdd(leftEdge, rightScalar)
                    : TryGetGraphEdge(right, out var rightEdge) && TryGetScalar(left, out var leftScalar)
                        ? context.Graph.ExportAdd(rightEdge, leftScalar)
                        : context.Graph.ExportAdd(
                            left.GetRequiredEdge(binaryOperator.Left),
                            right.GetRequiredEdge(binaryOperator.Right)
                        )
            ),
            BinaryOperatorType.Subtract => new ExportValue(
                TryGetGraphEdge(left, out var leftEdge) && TryGetScalar(right, out var rightScalar)
                    ? context.Graph.ExportSub(leftEdge, rightScalar)
                    : TryGetGraphEdge(right, out var rightEdge) && TryGetScalar(left, out var leftScalar)
                        ? context.Graph.ExportSub(AddScalar(context.Graph, "sub", leftScalar), rightEdge)
                        : context.Graph.ExportSub(
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
            BinaryOperatorType.Divide => new ExportValue(
                TryGetGraphEdge(left, out var leftEdge) && TryGetScalar(right, out var rightScalar)
                    ? context.Graph.ExportDiv(leftEdge, rightScalar)
                    : TryGetGraphEdge(right, out var rightEdge) && TryGetScalar(left, out var leftScalar)
                        ? context.Graph.ExportDiv(AddScalar(context.Graph, "div", leftScalar), rightEdge)
                        : context.Graph.ExportDiv(
                            left.GetRequiredEdge(binaryOperator.Left),
                            right.GetRequiredEdge(binaryOperator.Right)
                        )
            ),
            _ => throw new NotSupportedException($"Unsupported binary operator: {binaryOperator}"),
        };
    }

    private static ExportValue ExportConditionalExpression(
        ForwardExportContext context,
        ConditionalExpression conditional
    )
    {
        if (!TryEvaluateBooleanExpression(context, conditional.Condition, out var condition))
        {
            throw new NotSupportedException(
                $"Conditional expression condition must be statically resolvable during export: {conditional.Condition}"
            );
        }

        return ExportExpression(
            context,
            condition ? conditional.TrueExpression : conditional.FalseExpression
        );
    }

    private static ExportValue ExportUnaryOperator(
        ForwardExportContext context,
        UnaryOperatorExpression unaryOperator
    )
    {
        var operand = ExportExpression(context, unaryOperator.Expression);

        if (unaryOperator.Operator == UnaryOperatorType.Not
            && TryGetBoolean(operand, out var boolean))
        {
            return new ExportValue(!boolean);
        }

        if (TryGetIntegral(operand, out var integral))
        {
            // Unary signs on integral constants are common inside shapes and arange bounds:
            //   -1
            //   +sequenceLength
            return unaryOperator.Operator switch
            {
                UnaryOperatorType.Plus => new ExportValue(integral),
                UnaryOperatorType.Minus => new ExportValue(checked(-integral)),
                _ => throw new NotSupportedException($"Unsupported unary operator: {unaryOperator}"),
            };
        }

        if (TryGetFloating(operand, out var floating))
        {
            // Unary signs on float constants are common in masks:
            //   -10_000f
            return unaryOperator.Operator switch
            {
                UnaryOperatorType.Plus => new ExportValue(floating),
                UnaryOperatorType.Minus => new ExportValue(-floating),
                _ => throw new NotSupportedException($"Unsupported unary operator: {unaryOperator}"),
            };
        }

        throw new NotSupportedException($"Unsupported unary operator: {unaryOperator}");
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

        // First prefer concrete module exporters for calls like "_linear.forward(x)".
        // If no exporter exists, recursively decompile user-defined Module<Tensor, Tensor>
        // children such as "_block.forward(x)".
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

        // Exports reductions with an explicit dimension:
        //   sum(linear, 1)
        //   torch.sum(x, dim)
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

        // Handles batched and plain matrix multiplies with the same ONNX MatMul:
        //   matmul(query, key.transpose(2, 3))
        //   matmul(hiddenStates, tiedWeight)
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

        // Handles both positional and omitted dim:
        //   softmax(scores, dim: -1)
        //   torch.softmax(scores)
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

        // Supports torch.arange forms used for ids/positions:
        //   arange(end)
        //   arange(start, end)
        //   arange(start, end, step, dtype: ..., device: ...)
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

        // Constant factories such as "full([context, context], -10_000f, ...)" become
        // ONNX initializers instead of runtime graph operators.
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

        // Constant masks are evaluated during export:
        //   triu(full([n, n], -10_000f, ...), diagonal: 1)
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

    private static ExportValue ExportTorchConcat(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        var arguments = GetPositionalArguments(invocation).ToArray();
        if (arguments.Length < 1 || arguments.Length > 2)
        {
            throw new NotSupportedException($"Unsupported torch.cat argument count: {invocation}");
        }

        var inputs = ResolveGraphEdgeArray(context, arguments[0]);
        var dim = GetNamedArgument(invocation, "dim") is { } namedDim
            ? ResolveLongArgument(context, namedDim)
            : arguments.Length == 2
                ? ResolveLongArgument(context, arguments[1])
                : 0;

        // Dense blocks commonly use static tensor lists:
        //   cat(new[] { input, x1, x2 }, 1)
        //   torch.cat([left, right], dim: 1)
        return new ExportValue(context.Graph.ExportConcat(inputs, dim));
    }

    private static ExportValue ExportTorchInterpolate(
        ForwardExportContext context,
        InvocationExpression invocation
    )
    {
        var arguments = GetPositionalArguments(invocation).ToArray();
        if (arguments.Length < 1)
        {
            throw new NotSupportedException($"Unsupported interpolate argument count: {invocation}");
        }

        var input = ExportExpression(context, arguments[0]).GetRequiredEdge(invocation);
        var spatialSizes = arguments.Length >= 2
            ? ResolveLongArray(context, arguments[1]).ToArray()
            : [];
        var spatialScales = arguments.Length >= 3
            ? ResolveDoubleArray(context, arguments[2]).ToArray()
            : [];

        if (spatialSizes.Length != 0)
        {
            throw new NotSupportedException(
                $"functional.interpolate export currently supports scale_factor, not explicit size: {invocation}"
            );
        }

        if (spatialScales.Length == 0)
        {
            throw new NotSupportedException($"functional.interpolate export requires scale_factor: {invocation}");
        }

        var mode = arguments.Length >= 4
            ? ResolveInterpolationMode(arguments[3])
            : "nearest";
        var name = context.Graph.NextName("interpolate");
        var scales = context.Graph.AddTensor(
            name: $"{name}_scales",
            shape: [spatialScales.Length],
            value: spatialScales.Select(static x => checked((float)x)).ToArray()
        );

        return new ExportValue(
            context.Graph.Resize(
                name: name,
                options: new ResizeInputOptions
                {
                    X = input,
                    Roi = new OnnxEdge(string.Empty),
                    Scales = scales,
                    Axes = Enumerable.Range(2, spatialScales.Length).Select(static x => (long)x).ToArray(),
                    Antialias = 0,
                    CoordinateTransformationMode = mode == "nearest" ? "asymmetric" : "pytorch_half_pixel",
                    CubicCoeffA = -0.75f,
                    Mode = mode,
                    NearestMode = "floor",
                }
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
        const BindingFlags FLAGS = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

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

        // Inline helper methods by decompiling their own bodies:
        //   ComputeLogits(x) -> return matmul(hiddenStates, tiedWeight);
        //   CreatePositionIds(batch, device) -> return arange(...).unsqueeze(...);
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
                // Metadata reads are accepted when passed through to constant factories:
                //   full([n, n], value, dtype: x.dtype, device: x.device)
                return new ExportValue(new SymbolicTensorMember(memberReference.MemberName));
            }

            throw new NotSupportedException($"Cannot access member '{memberReference.MemberName}' on an ONNX edge.");
        }

        var value = target.Value
            ?? throw new NotSupportedException($"Cannot access member '{memberReference.MemberName}' on a null value.");

        if (TryGetTupleItemIndex(memberReference.MemberName, out var itemIndex))
        {
            // Deconstructed tuple values can later be read as ItemN by the decompiler:
            //   result.Item1
            return GetTupleElement(new ExportValue(value), itemIndex, memberReference);
        }

        // Regular member access resolves against live module/runtime objects:
        //   _tokenEmbedding.weight
        //   tokens.device
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
            // Plain assignment/reassignment:
            //   x = _block.forward(x);
            //   attentionScores = attentionScores + causalMask;
            context.Values[identifier.Identifier] = value;
            return;
        }

        if (target is IndexerExpression indexer
            && indexer.Target is IdentifierExpression targetIdentifier
            && context.Values.TryGetValue(targetIdentifier.Identifier, out var inlineArrayValue)
            && inlineArrayValue.Value is InlineArrayBuilder builder)
        {
            // Source-level collection expressions may appear as normal indexer assignments
            // when the decompiler can keep the shape literal close to C# syntax:
            //   buffer[0] = batchSize;
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

        // Newer C# collection expressions can decompile to calls into PrivateImplementationDetails:
        //   InlineArrayElementRef(ref buffer, 0) = batchSize;
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
        // Resolve module paths that appear on the left side of ".forward":
        //   _linear
        //   this._block
        //   _container._projection
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

        if (module is TorchModules.Sequential sequential)
        {
            return ExportSequentialWithDeepFallback(sequential, graph, input);
        }

        // Look for extension overloads that are more specific than TorchModule:
        //   Export(this Linear module, graph, input)
        //   Export(this LayerNorm module, graph, input)
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
            // TryDeepExportModule returns the produced edge directly when no concrete
            // Export overload exists for a user-defined child module.
            return deepExportOutput;
        }

        return ((MethodInfo)exportMethod).Invoke(null, [module, graph, input])
            ?? throw new InvalidOperationException(
                $"Export overload '{exportMethod}' returned null for '{module.GetType().FullName}'."
            );
    }

    private static IOnnxGraphEdge ExportSequentialWithDeepFallback(
        TorchModules.Sequential sequential,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        var children = sequential.children().OfType<TorchModule>().ToArray();
        if (children.Length == 0)
        {
            throw new NotSupportedException($"Unsupported TorchSharp module leaf: {sequential.GetType().FullName}.");
        }

        var current = input;
        foreach (var child in children)
        {
            current = (IOnnxGraphEdge)InvokeModuleExport(child, graph, current);
        }

        return current;
    }

    private static object? TryDeepExportModule(
        object module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        // Concrete TorchSharp module types (Linear, LayerNorm, LSTM, ...) should use explicit
        // exporters. This fallback is only for user modules whose forward can be lowered, e.g.
        // a custom transformer block composed from supported modules and tensor ops.
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
        // Decompile only the requested method token, then find the matching MethodDeclaration:
        //   forward(...)
        //   ComputeLogits(...)
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
        // UsingStatement.EmbeddedStatement can be either a block or a single statement:
        //   using var x = ...; { stmt1; stmt2; }
        //   using var x = ...; return x;
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
        // Runtime tensor constants are materialized as initializers:
        //   tiedWeight = _embedding.weight.transpose(0, 1)
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
            && invocation.Arguments.ElementAtOrDefault(0) is { } dimExpression)
        {
            // "mask.slice(0, 0, sequenceLength, 1)" may use "sequenceLength = x.shape[1]".
            // Static initializers can use the concrete shape; dynamic edges fall back to
            // long.MaxValue so Slice means "through the end" instead of accidentally ending at 0.
            var dim = ResolveLongArgument(context, dimExpression);
            if (input is OnnxTensor tensor)
            {
                return tensor.Shape[NormalizeAxis(dim, tensor.Shape.Length)];
            }

            return long.MaxValue;
        }

        return ConvertExportValueToLong(end, endExpression, long.MaxValue);
    }

    private static IEnumerable<Expression> GetPositionalArguments(InvocationExpression invocation)
    {
        // Keep ordinary arguments and unwrap named ones when they carry real values:
        //   arange(start, end, step, dtype: ..., device: ...)
        // becomes start/end/step for constant factory evaluation.
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
                // Named dtype/device arguments decompile differently across compiler versions:
                //   dtype: ScalarType.Int64
                //   device: input.device
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
        // Tensor-scalar arithmetic accepts primitive numeric values:
        //   x * _scale
        //   input + 8f
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

    private static bool TryEvaluateBooleanExpression(
        ForwardExportContext context,
        Expression expression,
        out bool result
    )
    {
        expression = UnwrapNamedArgument(expression);

        switch (expression)
        {
            case ParenthesizedExpression parenthesized:
                return TryEvaluateBooleanExpression(context, parenthesized.Expression, out result);

            case CastExpression cast:
                return TryEvaluateBooleanExpression(context, cast.Expression, out result);

            case UnaryOperatorExpression { Operator: UnaryOperatorType.Not } unary:
                if (TryEvaluateBooleanExpression(context, unary.Expression, out var negated))
                {
                    result = !negated;
                    return true;
                }

                break;

            case BinaryOperatorExpression binary:
                return TryEvaluateBooleanBinary(context, binary, out result);
        }

        var value = ExportExpression(context, expression);
        return TryGetBoolean(value, out result);
    }

    private static bool TryEvaluateBooleanBinary(
        ForwardExportContext context,
        BinaryOperatorExpression binaryOperator,
        out bool result
    )
    {
        switch (binaryOperator.Operator)
        {
            case BinaryOperatorType.Equality:
            case BinaryOperatorType.InEquality:
                var left = ExportExpression(context, binaryOperator.Left).Value;
                var right = ExportExpression(context, binaryOperator.Right).Value;
                result = Equals(left, right);
                if (binaryOperator.Operator == BinaryOperatorType.InEquality)
                {
                    result = !result;
                }

                return true;

            case BinaryOperatorType.ConditionalAnd:
                if (TryEvaluateBooleanExpression(context, binaryOperator.Left, out var andLeft)
                    && TryEvaluateBooleanExpression(context, binaryOperator.Right, out var andRight))
                {
                    result = andLeft && andRight;
                    return true;
                }

                break;

            case BinaryOperatorType.ConditionalOr:
                if (TryEvaluateBooleanExpression(context, binaryOperator.Left, out var orLeft)
                    && TryEvaluateBooleanExpression(context, binaryOperator.Right, out var orRight))
                {
                    result = orLeft || orRight;
                    return true;
                }

                break;
        }

        result = default;
        return false;
    }

    private static bool TryGetBoolean(ExportValue value, out bool result)
    {
        if (value.Value is bool boolean)
        {
            result = boolean;
            return true;
        }

        result = default;
        return false;
    }

    private static bool TryFoldNumericBinary(
        BinaryOperatorType operatorType,
        ExportValue left,
        ExportValue right,
        out ExportValue result
    )
    {
        // Integral math feeds shape and index arguments:
        //   var step = ((_longBase + 7L) / 4L) + 1L;
        // Division is kept in floating space to avoid surprising integer truncation here.
        if (TryGetIntegral(left, out var leftIntegral)
            && TryGetIntegral(right, out var rightIntegral)
            && operatorType != BinaryOperatorType.Divide)
        {
            result = operatorType switch
            {
                BinaryOperatorType.Add => new ExportValue(checked(leftIntegral + rightIntegral)),
                BinaryOperatorType.Subtract => new ExportValue(checked(leftIntegral - rightIntegral)),
                BinaryOperatorType.Multiply => new ExportValue(checked(leftIntegral * rightIntegral)),
                BinaryOperatorType.Modulus => new ExportValue(leftIntegral % rightIntegral),
                _ => default,
            };

            return result.Value is not null;
        }

        if (TryGetFloating(left, out var leftFloating)
            && TryGetFloating(right, out var rightFloating))
        {
            // Floating math feeds scalar tensor arithmetic:
            //   var scale = (((_floatBase + 2f) * 4f) - 6f) / 3f;
            result = operatorType switch
            {
                BinaryOperatorType.Add => new ExportValue(leftFloating + rightFloating),
                BinaryOperatorType.Subtract => new ExportValue(leftFloating - rightFloating),
                BinaryOperatorType.Multiply => new ExportValue(leftFloating * rightFloating),
                BinaryOperatorType.Divide => new ExportValue(leftFloating / rightFloating),
                BinaryOperatorType.Modulus => new ExportValue(leftFloating % rightFloating),
                _ => default,
            };

            return result.Value is not null;
        }

        result = default;
        return false;
    }

    private static bool TryGetIntegral(ExportValue value, out long result)
    {
        switch (value.Value)
        {
            case byte x:
                result = x;
                return true;
            case sbyte x:
                result = x;
                return true;
            case short x:
                result = x;
                return true;
            case ushort x:
                result = x;
                return true;
            case int x:
                result = x;
                return true;
            case uint x:
                result = x;
                return true;
            case long x:
                result = x;
                return true;
            case ulong x when x <= long.MaxValue:
                result = checked((long)x);
                return true;
            default:
                result = default;
                return false;
        }
    }

    private static bool TryGetFloating(ExportValue value, out float result)
    {
        if (TryGetIntegral(value, out var integral))
        {
            result = integral;
            return true;
        }

        switch (value.Value)
        {
            case float x:
                result = x;
                return true;
            case double x:
                result = checked((float)x);
                return true;
            case decimal x:
                result = checked((float)x);
                return true;
            default:
                result = default;
                return false;
        }
    }

    private static IReadOnlyList<long> ResolveLongArguments(
        ForwardExportContext context,
        IEnumerable<Expression> arguments
    )
    {
        var argumentList = arguments.ToArray();
        if (argumentList.Length == 1)
        {
            // Single argument may itself be a shape/permutation list:
            //   view([batchSize, sequenceLength, hidden])
            return ResolveLongArray(context, argumentList[0]);
        }

        // Multiple arguments are already the desired long list:
        //   permute(2, 0, 3, 1, 4)
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
            case ParenthesizedExpression parenthesized:
                // Shape arrays may be wrapped by syntax noise:
                //   ([batchSize, sequenceLength, hidden])
                return ResolveLongArray(context, parenthesized.Expression);

            case CastExpression cast:
                // Casts can appear around generated collection-expression temporaries.
                return ResolveLongArray(context, cast.Expression);

            case ArrayCreateExpression arrayCreate:
                // Handles explicit arrays:
                //   new long[] { batchSize, sequenceLength, hidden }
                return arrayCreate.Initializer.Elements
                    .OfType<Expression>()
                    .Select(element => ResolveLongArgument(context, element))
                    .ToArray();

            case ArrayInitializerExpression arrayInitializer:
                // Handles collection expressions and array initializers:
                //   [batchSize, sequenceLength, hidden]
                //   { batchSize, sequenceLength, hidden }
                return arrayInitializer.Elements
                    .OfType<Expression>()
                    .Select(element => ResolveLongArgument(context, element))
                    .ToArray();

            case InvocationExpression invocation
                when IsArrayEmptyInvocation(invocation):
                return [];

            case InvocationExpression invocation
                when TryResolveInlineArraySpan(context, invocation, out var values):
                // Handles compiler-generated reads:
                //   InlineArrayAsReadOnlySpan(in buffer, 3)
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

    private static IReadOnlyList<double> ResolveDoubleArray(
        ForwardExportContext context,
        Expression expression
    )
    {
        expression = UnwrapNamedArgument(expression);

        switch (expression)
        {
            case ParenthesizedExpression parenthesized:
                return ResolveDoubleArray(context, parenthesized.Expression);

            case CastExpression cast:
                return ResolveDoubleArray(context, cast.Expression);

            case NullReferenceExpression:
                return [];

            case ArrayCreateExpression arrayCreate:
                return arrayCreate.Initializer.Elements
                    .OfType<Expression>()
                    .Select(element => Convert.ToDouble(ExportExpression(context, element).Value))
                    .ToArray();

            case ArrayInitializerExpression arrayInitializer:
                return arrayInitializer.Elements
                    .OfType<Expression>()
                    .Select(element => Convert.ToDouble(ExportExpression(context, element).Value))
                    .ToArray();

            case InvocationExpression invocation
                when IsArrayEmptyInvocation(invocation):
                return [];

            default:
                return [Convert.ToDouble(ExportExpression(context, expression).Value)];
        }
    }

    private static IReadOnlyList<IOnnxGraphEdge> ResolveGraphEdgeArray(
        ForwardExportContext context,
        Expression expression
    )
    {
        expression = UnwrapNamedArgument(expression);

        var elements = expression switch
        {
            ParenthesizedExpression parenthesized => ResolveGraphEdgeArray(context, parenthesized.Expression),
            CastExpression cast => ResolveGraphEdgeArray(context, cast.Expression),
            ArrayCreateExpression arrayCreate => arrayCreate.Initializer.Elements
                .OfType<Expression>()
                .Select(element => ExportExpression(context, element).GetRequiredEdge(element))
                .ToArray(),
            ArrayInitializerExpression arrayInitializer => arrayInitializer.Elements
                .OfType<Expression>()
                .Select(element => ExportExpression(context, element).GetRequiredEdge(element))
                .ToArray(),
            InvocationExpression invocation
                when TryResolveInlineArrayGraphEdges(context, invocation, out var values) => values,
            _ => throw new NotSupportedException($"Expression '{expression}' did not produce a tensor list."),
        };

        if (elements.Count == 0)
        {
            throw new NotSupportedException($"Tensor list '{expression}' must not be empty.");
        }

        return elements;
    }

    private static bool IsArrayEmptyInvocation(InvocationExpression invocation)
    {
        var text = invocation.ToString();
        // Decompiler output for empty array arguments commonly looks like:
        //   Array.Empty<long>()
        //   global::System.Array.Empty<double>()
        return text.Contains("Array.Empty", StringComparison.Ordinal);
    }

    private static bool TryResolveInlineArraySpan(
        ForwardExportContext context,
        InvocationExpression invocation,
        out IReadOnlyList<long> values
    )
    {
        if (!TryResolveInlineArrayBuilder(context, invocation, out var name, out var builder))
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

    private static bool TryResolveInlineArrayGraphEdges(
        ForwardExportContext context,
        InvocationExpression invocation,
        out IReadOnlyList<IOnnxGraphEdge> values
    )
    {
        if (!TryResolveInlineArrayBuilder(context, invocation, out var name, out var builder))
        {
            values = [];
            return false;
        }

        values = builder.Values
            .Select(item => item is null
                ? throw new NotSupportedException($"Inline array '{name}' has unassigned elements.")
                : item.Value.GetRequiredEdge(invocation))
            .ToArray();
        return true;
    }

    private static bool TryResolveInlineArrayBuilder(
        ForwardExportContext context,
        InvocationExpression invocation,
        out string name,
        out InlineArrayBuilder builder
    )
    {
        var text = invocation.ToString();
        if (!text.Contains("InlineArrayAsReadOnlySpan", StringComparison.Ordinal)
            && !text.Contains("InlineArrayAsSpan", StringComparison.Ordinal))
        {
            name = string.Empty;
            builder = null!;
            return false;
        }

        var match = Regex.Match(
            text,
            @"(?:in|ref)\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*(?<length>\d+)",
            RegexOptions.CultureInvariant
        );
        // The decompiler text contains the generated buffer name and length:
        //   InlineArrayAsReadOnlySpan(in buffer, 5)
        //   InlineArrayAsSpan(ref buffer, 5)
        if (!match.Success)
        {
            name = string.Empty;
            builder = null!;
            return false;
        }

        name = match.Groups["name"].Value;
        if (context.Values.TryGetValue(name, out var builderValue)
            && builderValue.Value is InlineArrayBuilder inlineArrayBuilder)
        {
            builder = inlineArrayBuilder;
            return true;
        }

        builder = null!;
        return false;
    }

    private static long ResolveLongArgument(
        ForwardExportContext context,
        Expression? expression,
        long defaultValue = 0
    )
    {
        if (expression is null)
        {
            // Optional tensor method parameters have TorchSharp defaults:
            //   slice(dim, start) -> end defaults through the helper call site.
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
            { Value: ShapeDimensionReference reference } => TryResolveShapeDimension(reference, out var dimension) ? dimension : 0,
            { Value: null } => defaultValue,
            { Value: IConvertible convertible } => Convert.ToInt64(convertible),
            _ => throw new NotSupportedException($"Expression '{source}' did not produce an integer value."),
        };
    }

    private static bool TryResolveShapeDimension(
        ShapeDimensionReference reference,
        out long dimension
    )
    {
        if (reference.Source is OnnxTensor tensor
            && reference.Index >= 0
            && reference.Index < tensor.Shape.Length)
        {
            dimension = tensor.Shape[reference.Index];
            return true;
        }

        if (reference.Source is OnnxValue value
            && value.Type is OnnxTensorType { Shape: not null } tensorType
            && reference.Index >= 0
            && reference.Index < tensorType.Shape.Dimensions.Length
            && tensorType.Shape.Dimensions[reference.Index].GetValue() is long fixedDimension)
        {
            dimension = fixedDimension;
            return true;
        }

        dimension = default;
        return false;
    }

    private static int NormalizeAxis(long axis, long rank)
    {
        var normalized = axis < 0 ? axis + rank : axis;
        // Normalize PyTorch-style negative axes:
        //   dim: -1 on rank 4 -> axis 3
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
        // Scalar literals become rank-0 initializers when paired with graph edges:
        //   attentionScores * _scale
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
        // Default values of compiler-generated inline arrays look like:
        //   default(<>y__InlineArray3<long>)
        var match = Regex.Match(typeText, @"InlineArray(?<length>\d+)<", RegexOptions.CultureInvariant);
        if (match.Success && int.TryParse(match.Groups["length"].Value, out var length))
        {
            builder = new InlineArrayBuilder(length);
            return true;
        }

        builder = null!;
        return false;
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
            // TorchSharp recurrent exports expose named outputs, while true tuples expose
            // Item1/Item2/etc. Support both shapes for:
            //   var (y, h, c) = _lstm.forward(x);
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

        // Reflection bridges decompiled member syntax back to the live module instance:
        //   _scale
        //   _embedding.weight
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
        // Static imports and fully qualified calls both tend to print with "torch":
        //   torch.arange(...)
        //   global::TorchSharp.torch.full(...)
        return string.Equals(expression.ToString(), "torch", StringComparison.Ordinal)
            || expression.ToString().EndsWith(".torch", StringComparison.Ordinal);
    }

    private static bool IsTorchFunctionalReference(Expression expression)
    {
        var text = expression.ToString();
        // Functional calls may decompile as:
        //   torch.nn.functional.interpolate(...)
        //   global::TorchSharp.torch.nn.functional.interpolate(...)
        return text.EndsWith(".functional", StringComparison.Ordinal)
            || text.Contains(".torch.nn.functional", StringComparison.Ordinal);
    }

    private static bool IsTorchConcatName(string name)
    {
        return string.Equals(name, "cat", StringComparison.Ordinal)
            || string.Equals(name, "concat", StringComparison.Ordinal)
            || string.Equals(name, "concatenate", StringComparison.Ordinal);
    }

    private static string ResolveInterpolationMode(Expression expression)
    {
        var text = UnwrapNamedArgument(expression).ToString();
        if (text.EndsWith("Nearest", StringComparison.Ordinal))
        {
            return "nearest";
        }

        if (text.EndsWith("Linear", StringComparison.Ordinal)
            || text.EndsWith("Bilinear", StringComparison.Ordinal)
            || text.EndsWith("Trilinear", StringComparison.Ordinal))
        {
            return "linear";
        }

        if (text.EndsWith("Bicubic", StringComparison.Ordinal))
        {
            return "cubic";
        }

        throw new NotSupportedException($"Unsupported interpolation mode expression: {expression}");
    }

    // Per-method mutable export state. Values tracks graph edges, constants, and small
    // symbolic placeholders by decompiled variable name:
    //   x -> ONNX edge
    //   batchSize -> ShapeDimensionReference
    //   scale -> 0.3535f
    private sealed class ForwardExportContext(
        TorchModule rootModule,
        OnnxGraph graph
    )
    {
        public TorchModule RootModule { get; } = rootModule;

        public OnnxGraph Graph { get; } = graph;

        public Dictionary<string, ExportValue> Values { get; } = new(StringComparer.Ordinal);

        public bool HasReturned { get; private set; }

        public ExportValue? ReturnValue { get; private set; }

        public ExportValue Return(ExportValue value)
        {
            HasReturned = true;
            ReturnValue = value;
            return value;
        }
    }

    private readonly record struct ExportValue(object? Value)
    {
        public IOnnxGraphEdge GetRequiredEdge(AstNode source)
        {
            // Call sites use this when the syntax must resolve to a tensor-producing graph edge:
            //   matmul(query, key)
            //   _linear.forward(x)
            return Value as IOnnxGraphEdge
                ?? throw new NotSupportedException(
                    $"Expression '{source}' did not produce an ONNX graph edge."
                );
        }
    }

    // Placeholder for shape reads such as "x.shape[1]" before we know whether the consumer
    // wants ONNX Reshape's "copy input dimension" sentinel or a concrete initializer bound.
    private readonly record struct ShapeDimensionReference(IOnnxGraphEdge? Source, int Index);

    // Placeholder for metadata reads such as "x.dtype" and "x.device"; constant factories
    // accept them syntactically but current initializer emission decides dtype from values.
    private readonly record struct SymbolicTensorMember(string Name);

    // Builder for C# collection expressions after decompilation:
    //   [batchSize, sequenceLength, hidden]
    // may become "default(InlineArray3<long>)" plus element assignments.
    private sealed class InlineArrayBuilder(int length)
    {
        public ExportValue?[] Values { get; } = new ExportValue?[length];
    }
}
