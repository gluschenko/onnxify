using Onnxify;

namespace Onnxify.Compiler;

/// <summary>Lowers structured C# control flow retained by the syntax frontend into ONNX subgraphs.</summary>
internal sealed class CSharpOnnxControlFlowLowerer
{
    private readonly CompilerComputationTree _tree;
    private readonly OnnxGraph _graph;
    private readonly List<CompilerDiagnostic> _diagnostics;
    private readonly string? _caller;
    private readonly Dictionary<string, CompilerType> _types;
    private int _sequence;

    public CSharpOnnxControlFlowLowerer(
        CompilerComputationTree tree,
        OnnxGraph graph,
        List<CompilerDiagnostic> diagnostics,
        string? caller)
    {
        _tree = tree;
        _graph = graph;
        _diagnostics = diagnostics;
        _caller = caller;
        _types = tree.Inputs.Concat(tree.Outputs).Concat(tree.IntermediateValues)
            .GroupBy(value => value.Name, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.First().Type, StringComparer.Ordinal);
    }

    public void Lower()
    {
        var environment = _tree.Inputs.ToDictionary(value => value.Name,
            value => new Binding(value.Name, value.Type), StringComparer.Ordinal);
        var flow = LowerStatement(_tree.SyntaxBody!, _graph, environment, loopContext: null);
        if (flow == Flow.Return)
            return;

        foreach (var output in _tree.Outputs)
        {
            var source = environment.TryGetValue(output.Name, out var binding)
                ? binding.Name
                : output.Name;
            if (source != output.Name)
            {
                EmitIdentity(_graph, source, output.Name, output.Type, "return");
            }
        }
    }

    private Flow LowerStatement(
        CompilerStatement statement,
        OnnxGraph graph,
        Dictionary<string, Binding> environment,
        LoopContext? loopContext)
    {
        switch (statement)
        {
            case CompilerBlockStatement block:
                foreach (var child in block.Statements)
                {
                    var flow = loopContext is not null && child is not (CompilerBreakStatement or CompilerContinueStatement)
                        ? LowerIf(new CompilerRuntimeIfStatement(
                            new CompilerReferenceExpression(loopContext.ActiveName),
                            child), graph, environment, loopContext)
                        : LowerStatement(child, graph, environment, loopContext);
                    if (flow != Flow.Normal)
                    {
                        return flow;
                    }
                }
                return Flow.Normal;
            case CompilerDeclarationStatement declaration:
                if (declaration.Initializer is not null)
                {
                    var value = LowerExpression(declaration.Initializer, graph, environment);
                    environment[declaration.Name] = new Binding(value, GetType(declaration.Name, value));
                }
                else if (environment.TryGetValue(declaration.Name, out var prior))
                {
                    environment[declaration.Name] = prior;
                }
                else if (_types.TryGetValue(declaration.Name, out var declaredType))
                {
                    environment[declaration.Name] = new Binding(declaration.Name, declaredType);
                }
                else
                {
                    throw Unsupported(declaration, $"Local '{declaration.Name}' has no initializer and no incoming value.");
                }
                return Flow.Normal;
            case CompilerAssignmentStatement assignment:
                if (assignment.Target is not CompilerReferenceExpression target)
                {
                    throw Unsupported(assignment, "Only local-variable assignment can be lowered to ONNX control flow.");
                }
                var assigned = assignment.Operator == "="
                    ? LowerExpression(assignment.Value, graph, environment)
                    : LowerExpression(new CompilerBinaryExpression(
                        new CompilerReferenceExpression(target.Name), assignment.Operator.TrimEnd('='), assignment.Value),
                        graph, environment);
                var assignedType = environment.TryGetValue(target.Name, out var old)
                    ? old.Type : GetType(target.Name, assigned);
                environment[target.Name] = new Binding(assigned, assignedType);
                return Flow.Normal;
            case CompilerExpressionStatement expression:
                _ = LowerExpression(expression.Expression, graph, environment);
                return Flow.Normal;
            case CompilerReturnStatement { Expression: not null } ret:
                var returns = ret.Expression is CompilerTupleExpression tuple ? tuple.Items : [ret.Expression];
                if (graph == _graph)
                {
                    if (returns.Count != _tree.Outputs.Count)
                        throw Unsupported(ret, "Return value count does not match the declared ONNX outputs.");
                    for (var i = 0; i < returns.Count; i++)
                    {
                        var value = LowerExpression(returns[i], graph, environment);
                        var output = _tree.Outputs[i];
                        if (value != output.Name) EmitIdentity(graph, value, output.Name, output.Type, "return");
                    }
                    return Flow.Return;
                }
                throw Unsupported(ret, "Early returns from an ONNX control-flow subgraph are not supported.");
            case CompilerRuntimeIfStatement conditional:
                return LowerIf(conditional, graph, environment, loopContext);
            case CompilerRuntimeLoopStatement loop:
                LowerLoop(loop, graph, environment);
                return Flow.Normal;
            case CompilerBreakStatement when loopContext is not null:
                environment[loopContext.ActiveName] = new Binding(
                    AddScalar(graph, false, new CompilerScalarType(CompilerElementType.Boolean)),
                    new CompilerScalarType(CompilerElementType.Boolean));
                environment[loopContext.BreakingName] = new Binding(
                    AddScalar(graph, true, new CompilerScalarType(CompilerElementType.Boolean)),
                    new CompilerScalarType(CompilerElementType.Boolean));
                return Flow.Normal;
            case CompilerContinueStatement when loopContext is not null:
                environment[loopContext.ActiveName] = new Binding(
                    AddScalar(graph, false, new CompilerScalarType(CompilerElementType.Boolean)),
                    new CompilerScalarType(CompilerElementType.Boolean));
                return Flow.Normal;
            case CompilerStaticIfStatement conditional:
                if (TryStaticBoolean(conditional.Condition, out var staticCondition))
                    return LowerStatement(staticCondition
                        ? conditional.WhenTrue
                        : conditional.WhenFalse ?? new CompilerBlockStatement([]), graph, environment, loopContext);
                throw Unsupported(conditional, "Static if condition is not a compile-time boolean literal.");
            case CompilerStaticForeachStatement foreachStatement when foreachStatement.Collection is CompilerArrayExpression array:
                foreach (var item in array.Items)
                {
                    var itemName = LowerExpression(item, graph, environment);
                    var itemType = item is CompilerLiteralExpression { Literal: CompilerScalarLiteral scalar }
                        ? new CompilerScalarType(scalar.ElementType)
                        : environment.TryGetValue(itemName, out var itemBinding) ? itemBinding.Type
                        : throw Unsupported(foreachStatement, "Static foreach item type cannot be inferred.");
                    environment[foreachStatement.VariableName] = new Binding(itemName, itemType);
                    var flow = LowerStatement(foreachStatement.Body, graph, environment, loopContext);
                    if (flow != Flow.Normal) return flow;
                }
                return Flow.Normal;
            case CompilerStaticForeachStatement foreachStatement:
                throw Unsupported(foreachStatement, "Static foreach collection must be represented as a compiler-time array.");
            default:
                throw Unsupported(statement, $"Statement '{statement.GetType().Name}' cannot be lowered to ONNX.");
        }
    }

    private Flow LowerIf(
        CompilerRuntimeIfStatement conditional,
        OnnxGraph graph,
        Dictionary<string, Binding> environment,
        LoopContext? loopContext)
    {
        var condition = LowerExpression(conditional.Condition, graph, environment);
        var assigned = AssignedNames(conditional.WhenTrue)
            .Concat(conditional.WhenFalse is null ? [] : AssignedNames(conditional.WhenFalse))
            .Concat(loopContext is null ? [] : [loopContext.ActiveName, loopContext.BreakingName])
            .Where(environment.ContainsKey)
            .Distinct(StringComparer.Ordinal).ToArray();
        var thenGraph = OnnxModel.Create().Graph;
        thenGraph.Name = Next("if_then");
        var elseGraph = OnnxModel.Create().Graph;
        elseGraph.Name = Next("if_else");
        var thenEnvironment = new Dictionary<string, Binding>(environment, StringComparer.Ordinal);
        var elseEnvironment = new Dictionary<string, Binding>(environment, StringComparer.Ordinal);
        var thenFlow = LowerStatement(conditional.WhenTrue, thenGraph, thenEnvironment, loopContext);
        var elseFlow = Flow.Normal;
        if (conditional.WhenFalse is not null)
            elseFlow = LowerStatement(conditional.WhenFalse, elseGraph, elseEnvironment, loopContext);
        if (thenFlow != Flow.Normal || elseFlow != Flow.Normal)
            throw Unsupported(conditional, "break, continue, and early return inside a runtime if require conditional control-state lowering.");

        var resultNames = new List<string>();
        var resultTypes = new List<CompilerType>();
        foreach (var name in assigned)
        {
            if (!thenEnvironment.TryGetValue(name, out var thenValue)
                || !elseEnvironment.TryGetValue(name, out var elseValue))
                throw Unsupported(conditional, $"Variable '{name}' must be assigned on both runtime-if branches.");
            var type = environment.TryGetValue(name, out var current) ? current.Type : thenValue.Type;
            if (!type.Equals(thenValue.Type) || !type.Equals(elseValue.Type))
                throw Unsupported(conditional, $"Variable '{name}' changes compiler type across runtime-if branches.");
            var output = Next("if_value");
            AddBranchOutput(thenGraph, thenValue.Name, output, type);
            AddBranchOutput(elseGraph, elseValue.Name, output, type);
            resultNames.Add(output);
            resultTypes.Add(type);
        }
        var outputs = resultNames.Select(name => (IOnnxGraphEdge)new OnnxEdge(name)).ToArray();
        graph.AddNode(Next("if"), "If", string.Empty, string.Empty,
            [new OnnxEdge(condition)], outputs,
            [new OnnxAttribute<OnnxGraph>("then_branch", thenGraph), new OnnxAttribute<OnnxGraph>("else_branch", elseGraph)]);
        for (var index = 0; index < assigned.Length; index++)
            environment[assigned[index]] = new Binding(resultNames[index], resultTypes[index]);
        return Flow.Normal;
    }

    private void LowerLoop(
        CompilerRuntimeLoopStatement loop,
        OnnxGraph graph,
        Dictionary<string, Binding> environment)
    {
        foreach (var initializer in loop.Initializers)
            LowerStatement(initializer, graph, environment, loopContext: null);

        var carriedNames = AssignedNames(loop.Body)
            .Concat(loop.Increments.SelectMany(ReferencedNames))
            .Concat(loop.Condition is null ? [] : ReferencedNames(loop.Condition))
            .Distinct(StringComparer.Ordinal)
            .Where(environment.ContainsKey)
            .ToArray();
        var carried = carriedNames.Select(name => environment[name]).ToArray();
        var initialCondition = loop.Condition is null
            ? AddScalar(graph, true, new CompilerScalarType(CompilerElementType.Boolean))
            : LowerExpression(loop.Condition, graph, environment);

        var body = OnnxModel.Create().Graph;
        body.Name = Next("loop_body");
        var iterationName = body.NextName("__oxy042_iteration");
        body.AddInput(iterationName, OnnxTensorType.Create<long>(Array.Empty<OnnxDimension>()));
        var conditionName = body.NextName("__oxy042_condition");
        var bodyCondition = body.AddInput(conditionName, OnnxTensorType.Create<bool>(Array.Empty<OnnxDimension>()));
        var bodyEnvironment = new Dictionary<string, Binding>(environment, StringComparer.Ordinal);
        for (var index = 0; index < carriedNames.Length; index++)
        {
            var inputName = body.NextName($"__oxy042_carried_in_{index}");
            body.AddInput(inputName, ToOnnxType(carried[index].Type));
            bodyEnvironment[carriedNames[index]] = new Binding(inputName, carried[index].Type);
        }

        var activeName = Next("loop_active");
        var breakingName = Next("loop_breaking");
        var booleanType = new CompilerScalarType(CompilerElementType.Boolean);
        bodyEnvironment[activeName] = new Binding(AddScalar(body, true, booleanType), booleanType);
        bodyEnvironment[breakingName] = new Binding(AddScalar(body, false, booleanType), booleanType);
        var context = new LoopContext(activeName, breakingName);
        var bodyFlow = LowerStatement(loop.Body, body, bodyEnvironment, context);
        if (bodyFlow == Flow.Return)
            throw Unsupported(loop, "A return from inside a runtime loop cannot be represented by the current ONNX loop-carried contract.");
        if (loop.Increments.Count > 0)
        {
            var incrementStatements = new CompilerBlockStatement(
                loop.Increments.Select(increment => (CompilerStatement)new CompilerExpressionStatement(increment)));
            var runIncrement = new CompilerUnaryExpression("!", new CompilerReferenceExpression(breakingName));
            _ = LowerIf(new CompilerRuntimeIfStatement(runIncrement, incrementStatements), body, bodyEnvironment, loopContext: null);
        }
        var nextCondition = loop.Condition is null
            ? bodyCondition.Name
            : LowerExpression(loop.Condition, body, bodyEnvironment);
        var mergedCondition = Next("loop_condition_value");
        bodyEnvironment[mergedCondition] = new Binding(mergedCondition, booleanType);
        var breakCondition = new CompilerReferenceExpression(breakingName);
        var selectCondition = new CompilerRuntimeIfStatement(
            breakCondition,
            new CompilerAssignmentStatement(new CompilerReferenceExpression(mergedCondition),
                new CompilerLiteralExpression(new CompilerBooleanLiteral(false))),
            new CompilerAssignmentStatement(new CompilerReferenceExpression(mergedCondition),
                new CompilerReferenceExpression(nextCondition)));
        _ = LowerIf(selectCondition, body, bodyEnvironment, context);
        nextCondition = bodyEnvironment[mergedCondition].Name;
        if (body.GetValue(nextCondition) is not null)
        {
            var conditionOutput = body.NextName("__oxy042_condition_out");
            EmitIdentity(body, nextCondition, conditionOutput, new CompilerScalarType(CompilerElementType.Boolean), "loop_condition");
            nextCondition = conditionOutput;
        }
        body.AddOutput(nextCondition, OnnxTensorType.Create<bool>(Array.Empty<OnnxDimension>()));

        var loopOutputs = new List<IOnnxGraphEdge>();
        var outputNames = new List<string>();
        for (var index = 0; index < carriedNames.Length; index++)
        {
            var outputName = Next("loop_value");
            var state = bodyEnvironment[carriedNames[index]];
            var carriedOutput = state.Name;
            if (body.GetValue(carriedOutput) is not null)
            {
                carriedOutput = body.NextName($"__oxy042_carried_out_{index}");
                EmitIdentity(body, state.Name, carriedOutput, state.Type, "loop_carried");
            }
            body.AddOutput(carriedOutput, ToOnnxType(state.Type));
            loopOutputs.Add(new OnnxEdge(outputName));
            outputNames.Add(outputName);
        }
        graph.AddNode(Next("loop"), "Loop", string.Empty, string.Empty,
            [new OnnxEdge(string.Empty), new OnnxEdge(initialCondition), .. carried.Select(value => (IOnnxGraphEdge)new OnnxEdge(value.Name))],
            loopOutputs,
            [new OnnxAttribute<OnnxGraph>("body", body)]);
        for (var index = 0; index < carriedNames.Length; index++)
            environment[carriedNames[index]] = new Binding(outputNames[index], carried[index].Type);
        _ = iterationName;
    }

    private string LowerExpression(CompilerExpression expression, OnnxGraph graph, Dictionary<string, Binding> environment)
    {
        switch (expression)
        {
            case CompilerReferenceExpression reference:
                return environment.TryGetValue(reference.Name, out var binding)
                    ? binding.Name
                    : reference.Name;
            case CompilerLiteralExpression { Literal: CompilerBooleanLiteral boolean }:
                return AddScalar(graph, boolean.Value, new CompilerScalarType(boolean.ElementType));
            case CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral integer }:
                return AddScalar(graph, integer.Value, new CompilerScalarType(integer.ElementType));
            case CompilerLiteralExpression { Literal: CompilerUnsignedIntegerLiteral integer }:
                return AddScalar(graph, integer.Value, new CompilerScalarType(integer.ElementType));
            case CompilerLiteralExpression { Literal: CompilerFloatingPointLiteral floating }:
                return AddScalar(graph, floating.Value, new CompilerScalarType(floating.ElementType));
            case CompilerBinaryExpression binary:
            {
                if (!CompilerOperatorRegistry.TryGetTorchSharpBinaryOperator(binary.Operator, out var mapping) || mapping is null)
                    throw Unsupported(binary, $"Binary operator '{binary.Operator}' has no ONNX mapping.");
                var left = LowerExpression(binary.Left, graph, environment);
                var right = LowerExpression(binary.Right, graph, environment);
                var output = Next("value");
                graph.AddNode(mapping.CreateNode(Next(mapping.OnnxName), [new(left), new(right)], [new(output)]).Node);
                return output;
            }
            case CompilerUnaryExpression unary:
            {
                if (unary.Operator is "++" or "--" && unary.Expression is CompilerReferenceExpression reference)
                {
                    if (!environment.TryGetValue(reference.Name, out var target))
                        throw Unsupported(unary, $"Increment target '{reference.Name}' has no current value.");
                    var one = AddScalar(graph, ScalarOne(target.Type), target.Type);
                    var value = EmitBinary(graph, unary.Operator == "++" ? "+" : "-", target.Name, one, target.Type);
                    environment[reference.Name] = new Binding(value, target.Type);
                    return value;
                }
                if (!CompilerOperatorRegistry.TryGetTorchSharpUnaryOperator(unary.Operator, out var mapping) || mapping is null)
                    throw Unsupported(unary, $"Unary operator '{unary.Operator}' has no ONNX mapping.");
                var input = LowerExpression(unary.Expression, graph, environment);
                var output = Next("value");
                graph.AddNode(mapping.CreateNode(Next(mapping.OnnxName), [new(input)], [new(output)]).Node);
                return output;
            }
            case CompilerInvocationExpression invocation:
            {
                if (invocation.Target is CompilerMemberAccessExpression member
                    && member.MemberName == "item"
                    && member.Target is CompilerReferenceExpression itemTarget)
                    return LowerExpression(itemTarget, graph, environment);
                if (!CompilerOperatorRegistry.TryGetTorchSharpCall(invocation.Target, out var mapping, out var receiver)
                    || mapping is null)
                    throw Unsupported(invocation, "Invocation has no registered ONNX operator mapping.");
                var args = new List<string>();
                if (receiver is not null) args.Add(LowerExpression(receiver, graph, environment));
                var argumentInputs = mapping.InputCount - args.Count;
                if (argumentInputs < 0 || invocation.Arguments.Count < argumentInputs)
                    throw Unsupported(invocation, $"Call to '{mapping.TorchSharpNames.FirstOrDefault()}' has too few tensor inputs.");
                args.AddRange(invocation.Arguments.Take(argumentInputs).Select(argument => LowerExpression(argument, graph, environment)));
                var output = Next("value");
                var attrs = new List<CompilerAttribute>();
                foreach (var (attributeName, index) in mapping.AttributeNames.Select((name, index) => (name, index)))
                {
                    var arg = invocation.Arguments.ElementAtOrDefault(argumentInputs + index);
                    if (arg is CompilerLiteralExpression literal)
                        attrs.Add(new CompilerAttribute(attributeName, literal.Literal));
                }
                var operation = mapping.CreateNode(Next(mapping.OnnxName), args.Take(mapping.InputCount).Select(name => new CompilerValueReference(name)),
                    [new CompilerValueReference(output)], attributes: attrs, span: invocation.Span);
                graph.AddNode(operation.Node);
                return output;
            }
            default:
                throw Unsupported(expression, $"Expression '{expression.GetType().Name}' cannot be lowered to ONNX.");
        }
    }

    private static IEnumerable<string> AssignedNames(CompilerStatement statement)
    {
        switch (statement)
        {
            case CompilerDeclarationStatement declaration: yield return declaration.Name; break;
            case CompilerAssignmentStatement { Target: CompilerReferenceExpression reference }: yield return reference.Name; break;
            case CompilerBlockStatement block:
                foreach (var name in block.Statements.SelectMany(AssignedNames)) yield return name;
                break;
            case CompilerRuntimeIfStatement conditional:
                foreach (var name in AssignedNames(conditional.WhenTrue)) yield return name;
                if (conditional.WhenFalse is not null)
                    foreach (var name in AssignedNames(conditional.WhenFalse)) yield return name;
                break;
            case CompilerRuntimeLoopStatement loop:
                foreach (var name in AssignedNames(loop.Body)) yield return name;
                break;
            case CompilerExpressionStatement { Expression: CompilerUnaryExpression { Operator: "++" or "--", Expression: CompilerReferenceExpression reference } }:
                yield return reference.Name;
                break;
        }
    }

    private static IEnumerable<string> ReferencedNames(CompilerExpression expression)
    {
        if (expression is CompilerReferenceExpression reference) yield return reference.Name;
        else if (expression is CompilerBinaryExpression binary)
        {
            foreach (var name in ReferencedNames(binary.Left)) yield return name;
            foreach (var name in ReferencedNames(binary.Right)) yield return name;
        }
        else if (expression is CompilerUnaryExpression unary)
            foreach (var name in ReferencedNames(unary.Expression)) yield return name;
    }

    private static bool TryStaticBoolean(CompilerExpression expression, out bool value)
    {
        switch (expression)
        {
            case CompilerLiteralExpression { Literal: CompilerBooleanLiteral literal }:
                value = literal.Value;
                return true;
            case CompilerUnaryExpression { Operator: "!", Expression: var operand }
                when TryStaticBoolean(operand, out var inner):
                value = !inner;
                return true;
            default:
                value = false;
                return false;
        }
    }

    private void AddBranchOutput(OnnxGraph graph, string input, string output, CompilerType type)
    {
        if (input != output)
            EmitIdentity(graph, input, output, type, "branch_value");
        graph.AddOutput(output, ToOnnxType(type));
    }

    private void EmitIdentity(OnnxGraph graph, string input, string output, CompilerType type, string prefix)
    {
        graph.AddNode(Next("Identity"), "Identity", string.Empty, string.Empty,
            [new OnnxEdge(input)], [new OnnxEdge(output)], []);
    }

    private string AddScalar(OnnxGraph graph, object value, CompilerType type)
    {
        var name = Next("constant");
        switch (type)
        {
            case CompilerScalarType { ElementType: CompilerElementType.Boolean }:
                graph.AddTensor(name, [], [Convert.ToBoolean(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.Int8 }:
                graph.AddTensor(name, [], [Convert.ToSByte(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.UInt8 }:
                graph.AddTensor(name, [], [Convert.ToByte(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.Int16 }:
                graph.AddTensor(name, [], [Convert.ToInt16(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.UInt16 }:
                graph.AddTensor(name, [], [Convert.ToUInt16(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.Int32 }:
                graph.AddTensor(name, [], [Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.UInt32 }:
                graph.AddTensor(name, [], [Convert.ToUInt32(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.Int64 }:
                graph.AddTensor(name, [], [Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.UInt64 }:
                graph.AddTensor(name, [], [Convert.ToUInt64(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.Float32 }:
                graph.AddTensor(name, [], [Convert.ToSingle(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            case CompilerScalarType { ElementType: CompilerElementType.Float64 }:
                graph.AddTensor(name, [], [Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture)]); break;
            default: throw new CompilerConversionException(CompilerDiagnosticCodes.Unsupported, $"Scalar initializer type '{type.GetType().Name}' is unsupported.");
        }
        return name;
    }

    private string EmitBinary(OnnxGraph graph, string token, string left, string right, CompilerType type)
    {
        if (!CompilerOperatorRegistry.TryGetTorchSharpBinaryOperator(token, out var mapping) || mapping is null)
            throw new CompilerConversionException(CompilerDiagnosticCodes.Unsupported, $"Binary operator '{token}' has no ONNX mapping.");
        var output = Next("value");
        graph.AddNode(mapping.CreateNode(Next(mapping.OnnxName), [new(left), new(right)], [new(output)]).Node);
        return output;
    }

    private static object ScalarOne(CompilerType type) => type is CompilerScalarType scalar
        ? scalar.ElementType switch
        {
            CompilerElementType.Int64 => 1L,
            CompilerElementType.UInt64 => 1UL,
            CompilerElementType.Float32 => 1f,
            CompilerElementType.Float64 => 1d,
            _ => 1,
        }
        : 1f;

    private CompilerType GetType(string variable, string value) => _types.TryGetValue(variable, out var type)
        ? type
        : throw new CompilerConversionException(CompilerDiagnosticCodes.MissingReference, $"No type metadata is available for local '{variable}' (value '{value}').");

    private static OnnxTensorType ToOnnxType(CompilerType type) => type switch
    {
        CompilerTensorType tensor => (OnnxTensorType)OnnxCompilerTypeMap.ToOnnxType(tensor),
        CompilerScalarType scalar => new OnnxTensorType(
            CompilerElementTypeMap.ToSystemType(scalar.ElementType),
            OnnxTensorShape.Create(Array.Empty<long>()),
            string.Empty),
        _ => throw new CompilerConversionException(CompilerDiagnosticCodes.Unsupported, $"Control-flow values must be tensors or scalars; received '{type.GetType().Name}'."),
    };

    private string Next(string prefix) => _graph.NextName($"__oxy042_{prefix}_{_sequence++}");
    private static CompilerConversionException Unsupported(CompilerStatement node, string message) =>
        new(CompilerDiagnosticCodes.Unsupported, message);
    private static CompilerConversionException Unsupported(CompilerExpression node, string message) =>
        new(CompilerDiagnosticCodes.Unsupported, message);
    private static CompilerConversionException Unsupported(CompilerRuntimeLoopStatement node, string message) =>
        new(CompilerDiagnosticCodes.Unsupported, message);

    private sealed record Binding(string Name, CompilerType Type);
    private enum Flow { Normal, Return }

    private sealed class LoopContext(string activeName, string breakingName)
    {
        public string ActiveName { get; } = activeName;
        public string BreakingName { get; } = breakingName;
    }
}
