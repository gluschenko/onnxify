using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::logit")]
internal sealed class LogitOperator() : TorchSharpCompositeOperator("Logit", "com.onnxify.compiler", "torch.logit")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override int MinimumInputCount => 1;
    public override int MaximumInputCount => 2;
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return ValidateFloatingPointTensorInputs(node, knownTypes,
            CompilerElementType.Float16, CompilerElementType.Float32, CompilerElementType.Float64);
    }
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.logit"),
        CompilerTorchSharpForm.Call("Tensor.logit"),
    ];

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        if (operation.Node.Inputs.Count != 1)
        {
            throw Unsupported(operation.Span, "Logit with a compile-time epsilon requires a TorchSharp print context.");
        }

        return $"torch.logit({CompilerCSharpNaming.Identifier(operation.Node.Inputs[0].Name)})";
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation, CompilerTorchSharpPrintContext context)
    {
        var input = TorchSharpInput(operation, 0, context);
        return operation.Node.Inputs.Count == 1
            ? $"torch.logit({input})"
            : $"torch.logit({input}, eps: {TorchSharpInput(operation, 1, context)})";
    }

    protected override bool AllowsScalarOperand(int index) => index == 1;

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        var invocation = context.Invocation;
        var input = context.Receiver ?? invocation?.Arguments.FirstOrDefault();
        var epsilonExpression = context.Receiver is null
            ? invocation?.Arguments.Skip(1).SingleOrDefault()
            : invocation?.Arguments.SingleOrDefault();
        if (input is null
            || invocation is null
            || invocation.Arguments.Count > 2
            || (context.Receiver is null && invocation.Arguments.Count is < 1 or > 2)
            || (context.Receiver is not null && invocation.Arguments.Count > 1))
        {
            throw context.Unsupported(expression, "TorchSharp logit requires one tensor input and an optional compile-time epsilon.");
        }

        var inputReference = context.RequireTensorReference(input, OnnxName);
        var inputs = new List<CompilerValueReference> { inputReference };
        if (epsilonExpression is not null
            && epsilonExpression is not CompilerLiteralExpression { Literal: CompilerNullLiteral })
        {
            if (context.TryAddScalarInitializer(this, 1, epsilonExpression, out var epsilon))
            {
                inputs.Add(epsilon);
            }
            else
            {
                throw context.Unsupported(epsilonExpression, "Logit epsilon must be null or a compile-time numeric literal.");
            }
        }

        context.AddOperation(this, inputs, span: expression.Span);
        return true;
    }

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (TryScanDefaultCastLikePattern(context, out operation, out consumedNodeCount))
        {
            return true;
        }

        if (TryScanDefaultPattern(context, out operation, out consumedNodeCount))
        {
            return true;
        }

        if (TryScanCastLikeClampPattern(context, out operation, out consumedNodeCount))
        {
            return true;
        }

        if (!context.TryGetNode<Onnxify.LessOrEqual>(0, out var upperClamp)
            || !context.TryGetNode<Onnxify.Where>(1, out var upperWhere)
            || !context.TryGetNode<Onnxify.Less>(2, out var lowerClamp)
            || !context.TryGetNode<Onnxify.Where>(3, out var lowerWhere)
            || !context.TryGetNode<Onnxify.Sub>(4, out var subtract)
            || !context.TryGetNode<Onnxify.Div>(5, out var divide)
            || !context.TryGetNode<Onnxify.Log>(6, out var logarithm)
            || upperClamp is null || upperWhere is null || lowerClamp is null || lowerWhere is null
            || subtract is null || divide is null || logarithm is null
            || !context.IsScalarInitializer(upperClamp.B.Name)
            || !context.IsScalarInitializer(lowerClamp.B.Name)
            || !AreComplementaryScalars(context.Graph, upperClamp.B.Name, lowerClamp.B.Name)
            || !string.Equals(upperClamp.A.Name, upperWhere.X.Name, StringComparison.Ordinal)
            || !string.Equals(upperClamp.C.Name, upperWhere.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(upperWhere.Output.Name, lowerClamp.A.Name, StringComparison.Ordinal)
            || !string.Equals(lowerClamp.C.Name, lowerWhere.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(lowerClamp.B.Name, lowerWhere.X.Name, StringComparison.Ordinal)
            || !string.Equals(upperWhere.Output.Name, lowerWhere.Y.Name, StringComparison.Ordinal)
            || !context.IsScalarConstant(subtract.A.Name, 1d)
            || !string.Equals(lowerWhere.Output.Name, subtract.B.Name, StringComparison.Ordinal)
            || !string.Equals(lowerWhere.Output.Name, divide.A.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.C.Name, divide.B.Name, StringComparison.Ordinal)
            || !string.Equals(divide.C.Name, logarithm.Input.Name, StringComparison.Ordinal)
            || !context.CanCollapse(7))
        {
            operation = null;
            consumedNodeCount = 0;
            return false;
        }

        operation = context.CreateCompositeStep(this, [upperClamp.A.Name, lowerClamp.B.Name], logarithm.Output.Name, 7);
        consumedNodeCount = 7;
        return true;
    }

    private static bool AreComplementaryScalars(OnnxGraph graph, string firstName, string secondName)
    {
        if (graph.GetValue(firstName) is not OnnxTensor { Values.Count: 1 } first
            || graph.GetValue(secondName) is not OnnxTensor { Values.Count: 1 } second
            || first.Values[0] is not IConvertible firstValue
            || second.Values[0] is not IConvertible secondValue)
        {
            return false;
        }

        try
        {
            return Math.Abs(firstValue.ToDouble(System.Globalization.CultureInfo.InvariantCulture)
                + secondValue.ToDouble(System.Globalization.CultureInfo.InvariantCulture) - 1d) <= 1e-6;
        }
        catch (Exception)
        {
            return false;
        }
    }

    private bool TryScanDefaultPattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Sub>(0, out var subtract)
            || !context.TryGetNode<Onnxify.Div>(1, out var divide)
            || !context.TryGetNode<Onnxify.Log>(2, out var logarithm)
            || subtract is null || divide is null || logarithm is null
            || !context.IsScalarConstant(subtract.A.Name, 1d)
            || !string.Equals(divide.A.Name, subtract.B.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.C.Name, divide.B.Name, StringComparison.Ordinal)
            || !string.Equals(divide.C.Name, logarithm.Input.Name, StringComparison.Ordinal)
            || !context.CanCollapse(3))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [subtract.B.Name], logarithm.Output.Name, 3);
        consumedNodeCount = 3;
        return true;
    }

    private bool TryScanDefaultCastLikePattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Sub>(1, out var subtract)
            || !context.TryGetNode<Onnxify.Div>(2, out var divide)
            || !context.TryGetNode<Onnxify.Log>(3, out var logarithm)
            || subtract is null || divide is null || logarithm is null)
        {
            return false;
        }

        var castLike = context.Graph.Nodes[context.StartIndex];
        if (!IsDefaultCastLike(castLike)
            || !context.IsScalarConstant(castLike.Inputs[0].Name, 1d)
            || !string.Equals(castLike.Inputs[1].Name, subtract.B.Name, StringComparison.Ordinal)
            || !string.Equals(castLike.Outputs[0].Name, subtract.A.Name, StringComparison.Ordinal)
            || !string.Equals(divide.A.Name, subtract.B.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.C.Name, divide.B.Name, StringComparison.Ordinal)
            || !string.Equals(divide.C.Name, logarithm.Input.Name, StringComparison.Ordinal)
            || !context.CanCollapse(4))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [subtract.B.Name], logarithm.Output.Name, 4);
        consumedNodeCount = 4;
        return true;
    }

    private bool TryScanCastLikeClampPattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.LessOrEqual>(2, out var upperClamp)
            || !context.TryGetNode<Onnxify.Where>(3, out var upperWhere)
            || !context.TryGetNode<Onnxify.Less>(4, out var lowerClamp)
            || !context.TryGetNode<Onnxify.Where>(5, out var lowerWhere)
            || !context.TryGetNode<Onnxify.Sub>(7, out var subtract)
            || !context.TryGetNode<Onnxify.Div>(8, out var divide)
            || !context.TryGetNode<Onnxify.Log>(9, out var logarithm)
            || upperClamp is null || upperWhere is null || lowerClamp is null || lowerWhere is null
            || subtract is null || divide is null || logarithm is null)
        {
            return false;
        }

        var nodes = context.Graph.Nodes;
        var upperBoundCast = nodes[context.StartIndex];
        var epsilonCast = nodes[context.StartIndex + 1];
        var oneCast = nodes[context.StartIndex + 6];
        var epsilonName = epsilonCast.Inputs.Count == 2 ? epsilonCast.Inputs[0].Name : string.Empty;
        if (!IsDefaultCastLike(upperBoundCast)
            || !IsDefaultCastLike(epsilonCast)
            || !IsDefaultCastLike(oneCast)
            || !context.IsScalarInitializer(epsilonName)
            || !context.IsScalarConstant(oneCast.Inputs[0].Name, 1d)
            || !AreComplementaryScalars(context.Graph, upperBoundCast.Inputs[0].Name, epsilonName)
            || !string.Equals(upperBoundCast.Inputs[1].Name, upperClamp.A.Name, StringComparison.Ordinal)
            || !string.Equals(epsilonCast.Inputs[1].Name, upperClamp.A.Name, StringComparison.Ordinal)
            || !string.Equals(oneCast.Inputs[1].Name, upperClamp.A.Name, StringComparison.Ordinal)
            || !string.Equals(upperBoundCast.Outputs[0].Name, upperClamp.B.Name, StringComparison.Ordinal)
            || !string.Equals(epsilonCast.Outputs[0].Name, lowerClamp.B.Name, StringComparison.Ordinal)
            || !string.Equals(upperClamp.A.Name, upperWhere.X.Name, StringComparison.Ordinal)
            || !string.Equals(upperClamp.C.Name, upperWhere.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(upperWhere.Output.Name, lowerClamp.A.Name, StringComparison.Ordinal)
            || !string.Equals(lowerClamp.C.Name, lowerWhere.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(lowerClamp.B.Name, lowerWhere.X.Name, StringComparison.Ordinal)
            || !string.Equals(upperWhere.Output.Name, lowerWhere.Y.Name, StringComparison.Ordinal)
            || !string.Equals(oneCast.Outputs[0].Name, subtract.A.Name, StringComparison.Ordinal)
            || !string.Equals(lowerWhere.Output.Name, subtract.B.Name, StringComparison.Ordinal)
            || !string.Equals(lowerWhere.Output.Name, divide.A.Name, StringComparison.Ordinal)
            || !string.Equals(subtract.C.Name, divide.B.Name, StringComparison.Ordinal)
            || !string.Equals(divide.C.Name, logarithm.Input.Name, StringComparison.Ordinal)
            || !context.CanCollapse(10))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [upperClamp.A.Name, epsilonName], logarithm.Output.Name, 10);
        consumedNodeCount = 10;
        return true;
    }

    private static bool IsDefaultCastLike(OnnxNode node) =>
        string.Equals(node.Domain, string.Empty, StringComparison.Ordinal)
        && string.Equals(node.OpType, "CastLike", StringComparison.Ordinal)
        && node.Inputs.Count == 2
        && node.Outputs.Count == 1
        && node.Attributes.All(static attribute =>
            string.Equals(attribute.Name, "saturate", StringComparison.Ordinal)
            && attribute.GetValue() is long value
            && value == 1);

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        if (operation.Node.Inputs.Count == 2)
        {
            PrintClampedOnnx(graph, operation);
            return;
        }

        var oneSource = graph.AddTensor(GetUniqueValueName(graph, $"{operation.Name}__one_source"), [], [1d]);
        var one = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__one"));
        graph.AddNode(new OnnxNode(
            GetUniqueValueName(graph, $"{operation.Name}__cast_one"),
            "CastLike",
            string.Empty,
            string.Empty,
            [oneSource, Input(operation)],
            [one],
            []));
        var denominator = new OnnxEdge($"{operation.Name}__one_minus_input");
        var ratio = new OnnxEdge($"{operation.Name}__ratio");
        graph.AddNode(new Onnxify.Sub($"{operation.Name}__sub", new Onnxify.SubInputOutputOptions
        {
            A = one, B = Input(operation), C = denominator,
        }));
        graph.AddNode(new Onnxify.Div($"{operation.Name}__div", new Onnxify.DivInputOutputOptions
        {
            A = Input(operation), B = denominator, C = ratio,
        }));
        graph.AddNode(new Onnxify.Log(operation.Name, new Onnxify.LogInputOutputOptions
        {
            Input = ratio, Output = Output(operation),
        }));
    }

    private static void PrintClampedOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var epsilonName = operation.Node.Inputs[1].Name;
        if (graph.GetValue(epsilonName) is not OnnxTensor { Values.Count: 1 } epsilonTensor
            || epsilonTensor.Values[0] is not IConvertible epsilonValue)
        {
            throw new CompilerConversionException(CompilerDiagnosticCodes.Unsupported, "Logit epsilon must be a scalar initializer.");
        }

        var epsilon = epsilonValue.ToDouble(System.Globalization.CultureInfo.InvariantCulture);
        var oneMinusEpsilonSource = graph.AddTensor(
            GetUniqueValueName(graph, $"{operation.Name}__one_minus_eps_source"), [], [1d - epsilon]);
        var oneMinusEpsilon = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__one_minus_eps"));
        var epsilonEdge = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__eps"));
        var epsilonLikeNode = GetUniqueValueName(graph, $"{operation.Name}__cast_eps");
        var oneMinusEpsilonNode = GetUniqueValueName(graph, $"{operation.Name}__cast_one_minus_eps");
        graph.AddNode(new OnnxNode(oneMinusEpsilonNode, "CastLike", string.Empty, string.Empty,
            [oneMinusEpsilonSource, Input(operation)], [oneMinusEpsilon], []));
        graph.AddNode(new OnnxNode(epsilonLikeNode, "CastLike", string.Empty, string.Empty,
            [Input(operation, 1), Input(operation)], [epsilonEdge], []));
        var upperCondition = new OnnxEdge($"{operation.Name}__upper_condition");
        var upperValue = new OnnxEdge($"{operation.Name}__upper_value");
        var lowerCondition = new OnnxEdge($"{operation.Name}__lower_condition");
        var clamped = new OnnxEdge($"{operation.Name}__clamped");
        var denominator = new OnnxEdge($"{operation.Name}__denominator");
        var ratio = new OnnxEdge($"{operation.Name}__ratio");

        graph.AddNode(new Onnxify.LessOrEqual($"{operation.Name}__upper_check", new Onnxify.LessOrEqualInputOutputOptions
        {
            A = Input(operation), B = oneMinusEpsilon, C = upperCondition,
        }));
        graph.AddNode(new Onnxify.Where($"{operation.Name}__upper_where", new WhereInputOutputOptions
        {
            Condition = upperCondition, X = Input(operation), Y = oneMinusEpsilon, Output = upperValue,
        }));
        graph.AddNode(new Onnxify.Less($"{operation.Name}__lower_check", new Onnxify.LessInputOutputOptions
        {
            A = upperValue, B = epsilonEdge, C = lowerCondition,
        }));
        graph.AddNode(new Onnxify.Where($"{operation.Name}__lower_where", new WhereInputOutputOptions
        {
            Condition = lowerCondition, X = epsilonEdge, Y = upperValue, Output = clamped,
        }));
        var oneSource = graph.AddTensor(GetUniqueValueName(graph, $"{operation.Name}__one_source"), [], [1d]);
        var one = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__one"));
        graph.AddNode(new OnnxNode(GetUniqueValueName(graph, $"{operation.Name}__cast_one"),
            "CastLike", string.Empty, string.Empty, [oneSource, Input(operation)], [one], []));
        graph.AddNode(new Onnxify.Sub($"{operation.Name}__sub", new Onnxify.SubInputOutputOptions
        {
            A = one, B = clamped, C = denominator,
        }));
        graph.AddNode(new Onnxify.Div($"{operation.Name}__div", new Onnxify.DivInputOutputOptions
        {
            A = clamped, B = denominator, C = ratio,
        }));
        graph.AddNode(new Onnxify.Log(operation.Name, new Onnxify.LogInputOutputOptions
        {
            Input = ratio, Output = Output(operation),
        }));
    }

    private static string GetUniqueValueName(OnnxGraph graph, string baseName)
    {
        var candidate = baseName;
        for (var suffix = 1; graph.GetValue(candidate) is not null; suffix++)
        {
            candidate = $"{baseName}__{suffix}";
        }

        return candidate;
    }
}
