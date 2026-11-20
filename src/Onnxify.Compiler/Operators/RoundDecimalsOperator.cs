using System.Globalization;
using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class RoundDecimalsOperator : TorchSharpCompositeOperator
{
    public static RoundDecimalsOperator Instance { get; } = new();

    private RoundDecimalsOperator() : base("RoundDecimals", "com.onnxify.compiler") { }

    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override IReadOnlyList<string> AttributeNames { get; } = ["decimals"];
    protected override IReadOnlyCollection<CompilerElementType> SupportedTensorInputTypes =>
    [CompilerElementType.Float16, CompilerElementType.Float32, CompilerElementType.Float64];

    public override CompilerOnnxStep CreateNode(
        string name,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        IEnumerable<CompilerAttribute>? attributes = null,
        CompilerSourceSpan? span = null)
    {
        var node = new OnnxNode(
            name,
            OnnxName,
            OnnxDomain,
            string.Empty,
            inputs.Select(static input => (IOnnxGraphEdge)new OnnxEdge(input.IsEmptyOptional ? string.Empty : input.Name)),
            outputs.Select(static output => (IOnnxGraphEdge)new OnnxEdge(output.IsEmptyOptional ? string.Empty : output.Name)),
            (attributes ?? []).Select(attribute => OnnxCompilerBackend.CreateOnnxAttribute(attribute, [], null)));
        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        return $"torch.round({CompilerCSharpNaming.Identifier(operation.Node.Inputs[0].Name)}, decimals: {GetDecimals(operation)})";
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation, CompilerTorchSharpPrintContext context)
    {
        return PrintTorchSharp(operation);
    }

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (TryScanCastLikeScalePattern(context, out operation))
        {
            consumedNodeCount = 4;
            return true;
        }

        if (TryScanTorchLibCastLikePattern(context, out operation))
        {
            consumedNodeCount = 6;
            return true;
        }

        if (!context.TryGetNode<Onnxify.Mul>(0, out var multiply)
            || !context.TryGetNode<Onnxify.Round>(1, out var round)
            || !context.TryGetNode<Onnxify.Div>(2, out var divide)
            || multiply is null
            || round is null
            || divide is null
            || !context.CanCollapse(3)
            || !string.Equals(multiply.C.Name, round.X.Name, StringComparison.Ordinal)
            || !string.Equals(round.Y.Name, divide.A.Name, StringComparison.Ordinal))
        {
            return false;
        }

        var inputName = context.IsScalarInitializer(multiply.A.Name) ? multiply.B.Name
            : context.IsScalarInitializer(multiply.B.Name) ? multiply.A.Name
            : null;
        var scaleName = inputName == multiply.A.Name ? multiply.B.Name
            : inputName == multiply.B.Name ? multiply.A.Name
            : null;
        if (inputName is null
            || scaleName is null
            || !string.Equals(scaleName, divide.B.Name, StringComparison.Ordinal)
            || !TryReadPowerOfTen(context.Graph, scaleName, out var decimals))
        {
            return false;
        }

        var finalNode = context.Graph.Nodes[context.StartIndex + 2];
        var name = string.IsNullOrWhiteSpace(finalNode.Name) ? $"round_decimals_{context.StartIndex}" : finalNode.Name;
        operation = CreateNode(
            name,
            [new CompilerValueReference(inputName)],
            [new CompilerValueReference(divide.C.Name)],
            [new CompilerAttribute("decimals", new CompilerSignedIntegerLiteral(CompilerElementType.Int64, decimals))],
            context.Span);
        consumedNodeCount = 3;
        return true;
    }

    private bool TryScanCastLikeScalePattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation)
    {
        operation = null;
        var start = context.StartIndex;
        var nodes = context.Graph.Nodes;
        if (start < 0
            || start + 4 > nodes.Count
            || !IsCastLike(nodes[start])
            || !context.TryGetNode<Onnxify.Mul>(1, out var multiply)
            || !context.TryGetNode<Onnxify.Round>(2, out var round)
            || !context.TryGetNode<Onnxify.Div>(3, out var divide)
            || multiply is null || round is null || divide is null)
        {
            return false;
        }

        var castLike = nodes[start];
        var inputName = castLike.Inputs[1].Name;
        var scaleName = castLike.Outputs[0].Name;
        if (!context.IsScalarInitializer(castLike.Inputs[0].Name)
            || !TryReadPowerOfTen(context.Graph, castLike.Inputs[0].Name, out var decimals)
            || !(multiply.A.Name == inputName && multiply.B.Name == scaleName
                || multiply.B.Name == inputName && multiply.A.Name == scaleName)
            || !string.Equals(multiply.C.Name, round.X.Name, StringComparison.Ordinal)
            || !string.Equals(round.Y.Name, divide.A.Name, StringComparison.Ordinal)
            || !string.Equals(scaleName, divide.B.Name, StringComparison.Ordinal)
            || !context.CanCollapse(4))
        {
            return false;
        }

        var finalNode = nodes[start + 3];
        var name = string.IsNullOrWhiteSpace(finalNode.Name) ? $"round_decimals_{start}" : finalNode.Name;
        operation = CreateNode(
            name,
            [new CompilerValueReference(inputName)],
            [new CompilerValueReference(divide.C.Name)],
            [new CompilerAttribute("decimals", new CompilerSignedIntegerLiteral(CompilerElementType.Int64, decimals))],
            context.Span);
        return true;
    }

    private bool TryScanTorchLibCastLikePattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation)
    {
        operation = null;
        var start = context.StartIndex;
        var nodes = context.Graph.Nodes;
        if (start < 0 || start + 6 > nodes.Count)
        {
            return false;
        }

        var tenCast = nodes[start];
        var decimalsCast = nodes[start + 1];
        if (!IsCastLike(tenCast)
            || !IsCastLike(decimalsCast)
            || !context.TryGetNode<Onnxify.Pow>(2, out var power)
            || !context.TryGetNode<Onnxify.Mul>(3, out var multiply)
            || !context.TryGetNode<Onnxify.Round>(4, out var round)
            || !context.TryGetNode<Onnxify.Div>(5, out var divide)
            || power is null
            || multiply is null
            || round is null
            || divide is null
            || !string.Equals(tenCast.Inputs[1].Name, decimalsCast.Inputs[1].Name, StringComparison.Ordinal)
            || !context.IsScalarConstant(tenCast.Inputs[0].Name, 10d)
            || !TryReadScalarInteger(context.Graph, decimalsCast.Inputs[0].Name, out var decimals)
            || !string.Equals(power.X.Name, tenCast.Outputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(power.Y.Name, decimalsCast.Outputs[0].Name, StringComparison.Ordinal)
            || !string.Equals(multiply.C.Name, round.X.Name, StringComparison.Ordinal)
            || !string.Equals(round.Y.Name, divide.A.Name, StringComparison.Ordinal)
            || !string.Equals(power.Z.Name, divide.B.Name, StringComparison.Ordinal)
            || !context.CanCollapse(6))
        {
            return false;
        }

        var inputName = string.Equals(multiply.A.Name, power.Z.Name, StringComparison.Ordinal) ? multiply.B.Name
            : string.Equals(multiply.B.Name, power.Z.Name, StringComparison.Ordinal) ? multiply.A.Name
            : null;
        if (inputName is null
            || !string.Equals(inputName, tenCast.Inputs[1].Name, StringComparison.Ordinal))
        {
            return false;
        }

        var finalNode = nodes[start + 5];
        var name = string.IsNullOrWhiteSpace(finalNode.Name) ? $"round_decimals_{start}" : finalNode.Name;
        operation = CreateNode(
            name,
            [new CompilerValueReference(inputName)],
            [new CompilerValueReference(divide.C.Name)],
            [new CompilerAttribute("decimals", new CompilerSignedIntegerLiteral(CompilerElementType.Int64, decimals))],
            context.Span);
        return true;
    }

    private static bool IsCastLike(OnnxNode node)
    {
        return string.Equals(node.Domain, string.Empty, StringComparison.Ordinal)
            && string.Equals(node.OpType, "CastLike", StringComparison.Ordinal)
            && node.Inputs.Count == 2
            && node.Outputs.Count == 1
            && node.Inputs.All(static input => !string.IsNullOrWhiteSpace(input.Name))
            && node.Outputs.All(static output => !string.IsNullOrWhiteSpace(output.Name))
            && node.Attributes.All(static attribute =>
                string.Equals(attribute.Name, "saturate", StringComparison.Ordinal)
                && attribute.GetValue() is long value
                && value == 1);
    }

    private static bool TryReadScalarInteger(OnnxGraph graph, string name, out long value)
    {
        value = 0;
        if (graph.GetValue(name) is not OnnxTensor { Shape.Length: 0, Values.Count: 1 } tensor
            || tensor.Values[0] is not IConvertible scalar)
        {
            return false;
        }

        try
        {
            var numericValue = scalar.ToDouble(CultureInfo.InvariantCulture);
            if (double.IsNaN(numericValue)
                || double.IsInfinity(numericValue)
                || Math.Truncate(numericValue) != numericValue
                || numericValue < long.MinValue
                || numericValue > long.MaxValue)
            {
                return false;
            }

            value = Convert.ToInt64(numericValue, CultureInfo.InvariantCulture);
            return true;
        }
        catch (Exception exception) when (exception is FormatException or InvalidCastException or OverflowException)
        {
            return false;
        }
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var decimals = GetDecimals(operation);
        var scaleValue = Math.Pow(10d, decimals);
        if (!IsScaleSupportedForInput(graph, operation, scaleValue))
        {
            throw Unsupported(operation.Span, $"RoundDecimals decimals '{decimals}' produces a scale outside the supported range for the input tensor dtype.");
        }

        var scaleSourceName = GetUniqueValueName(graph, $"{operation.Name}__scale_source");
        var scaleSource = graph.AddTensor(scaleSourceName, [], [scaleValue]);
        var scale = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__scale"));
        var scaled = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__scaled"));
        var rounded = new OnnxEdge(GetUniqueValueName(graph, $"{operation.Name}__rounded"));
        graph.AddNode(new OnnxNode(
            $"{operation.Name}__cast_scale",
            "CastLike",
            string.Empty,
            string.Empty,
            [scaleSource, Input(operation)],
            [scale],
            []));
        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__multiply",
            new Onnxify.MulInputOutputOptions { A = Input(operation), B = scale, C = scaled }));
        graph.AddNode(new Onnxify.Round(
            $"{operation.Name}__round",
            new Onnxify.RoundInputOutputOptions { X = scaled, Y = rounded }));
        graph.AddNode(new Onnxify.Div(
            operation.Name,
            new Onnxify.DivInputOutputOptions { A = rounded, B = scale, C = Output(operation) }));
    }

    private static bool IsScaleSupportedForInput(OnnxGraph graph, CompilerOnnxStep operation, double scale)
    {
        if (double.IsNaN(scale) || double.IsInfinity(scale) || scale == 0d)
        {
            return false;
        }

        if (graph.GetValue(operation.Node.Inputs[0].Name) is not OnnxValue { Type: OnnxTensorType tensorType })
        {
            return true;
        }

        return OnnxCompilerTypeMap.FromOnnxType(tensorType) is CompilerTensorType { ElementType: var elementType }
            && elementType switch
            {
                CompilerElementType.Float16 => scale <= 65504d && scale >= 2.98023223876953125e-8d,
                CompilerElementType.Float32 => !float.IsInfinity((float)scale) && (float)scale != 0f,
                CompilerElementType.Float64 => true,
                _ => false,
            };
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

    public static bool TryReadInteger(CompilerExpression expression, out long value)
    {
        switch (expression)
        {
            case CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral integer }:
                value = integer.Value;
                return true;
            case CompilerUnaryExpression
            {
                Operator: "-",
                Expression: CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral integer },
            }:
                value = -integer.Value;
                return true;
            default:
                value = 0;
                return false;
        }
    }

    private static long GetDecimals(CompilerOnnxStep operation)
    {
        if (operation.Node.Attributes.SingleOrDefault(static attribute => attribute.Name == "decimals")?.GetValue() is long decimals)
        {
            return decimals;
        }

        throw Unsupported(operation.Span, "RoundDecimals requires an integer 'decimals' attribute.");
    }

    private static bool TryReadPowerOfTen(OnnxGraph graph, string name, out long decimals)
    {
        decimals = 0;
        if (graph.GetValue(name) is not OnnxTensor { Shape.Length: 0, Values.Count: 1 } tensor
            || tensor.Values[0] is not IConvertible scalar)
        {
            return false;
        }

        try
        {
            var scale = scalar.ToDouble(CultureInfo.InvariantCulture);
            if (double.IsNaN(scale) || double.IsInfinity(scale) || scale <= 0d)
            {
                return false;
            }

            var nearestPower = Math.Round(Math.Log10(scale));
            if (nearestPower < int.MinValue || nearestPower > int.MaxValue)
            {
                return false;
            }

            var candidate = (int)nearestPower;
            var exactScale = Math.Pow(10d, candidate);
            var floatScale = (float)exactScale;
            if (!exactScale.Equals(scale)
                && !( !float.IsNaN(floatScale) && !float.IsInfinity(floatScale) && (double)floatScale == scale))
            {
                return false;
            }

            decimals = candidate;
            return true;
        }
        catch (Exception exception) when (exception is FormatException or InvalidCastException or OverflowException)
        {
            return false;
        }
    }
}
