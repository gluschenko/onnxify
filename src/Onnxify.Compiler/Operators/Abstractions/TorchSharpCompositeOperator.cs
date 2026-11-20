using Onnxify;
using System.Globalization;

namespace Onnxify.Compiler.Operators;

internal abstract class TorchSharpCompositeOperator : CompilerOperator
{
    private static readonly IReadOnlyCollection<CompilerElementType> DEFAULT_FLOATING_POINT_TYPES =
    [CompilerElementType.Float32, CompilerElementType.Float64];

    protected TorchSharpCompositeOperator(string name, string domain, params string[] forms)
        : base(CompilerOperatorIdentity.Onnx(name, domain))
    {
        TorchSharpForms = forms.Select(CompilerTorchSharpForm.Call).ToArray();
    }

    public override Type NodeType => typeof(OnnxNode);
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; }
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ExportOnly;
    public override int MinimumInputCount => 1;
    public override int MaximumInputCount => 1;
    public override IReadOnlyList<string> AttributeNames => [];
    protected virtual IReadOnlyCollection<CompilerElementType> SupportedTensorInputTypes => DEFAULT_FLOATING_POINT_TYPES;

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return ValidateFloatingPointTensorInputs(node, knownTypes, SupportedTensorInputTypes.ToArray());
    }

    protected virtual bool AllowsScalarOperand(int index)
    {
        return false;
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        if (TorchSharpForms.Count != 1)
        {
            throw Unsupported(operation.Span, $"'{OnnxName}' does not have one unambiguous TorchSharp call form.");
        }

        var arguments = operation.Node.Inputs
            .Select(static input => CompilerCSharpNaming.Identifier(input.Name));
        return $"{TorchSharpForms.Single().Name}({string.Join(", ", arguments)})";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (context.Invocation is null)
        {
            throw context.Unsupported(expression, $"TorchSharp '{OnnxName}' must be invoked as a method.");
        }

        var operands = (context.Receiver is null
                ? Array.Empty<CompilerExpression>()
                : [context.Receiver])
            .Concat(context.Invocation.Arguments)
            .ToArray();
        if (operands.Length < MinimumInputCount || operands.Length > MaximumInputCount)
        {
            throw context.Unsupported(
                expression,
                $"TorchSharp '{OnnxName}' requires between {MinimumInputCount} and {MaximumInputCount} tensor operands.");
        }

        var inputs = new List<CompilerValueReference>(operands.Length);
        for (var index = 0; index < operands.Length; index++)
        {
            var operand = operands[index];
            if (AllowsScalarOperand(index) && context.TryAddScalarInitializer(this, index, operand, out var scalar))
            {
                inputs.Add(scalar);
                continue;
            }

            if (operand is CompilerReferenceExpression)
            {
                inputs.Add(context.RequireTensorReference(operand, OnnxName));
                continue;
            }

            throw context.Unsupported(operand, $"TorchSharp '{OnnxName}' requires tensor operands in this position.");
        }

        context.AddOperation(this, inputs, span: expression.Span);
        return true;
    }

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
            Array.Empty<OnnxAttribute>());
        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    protected static OnnxEdge Input(CompilerOnnxStep operation, int index = 0)
    {
        return new OnnxEdge(operation.Node.Inputs[index].Name);
    }

    protected static OnnxEdge Output(CompilerOnnxStep operation)
    {
        return new OnnxEdge(operation.Node.Outputs[0].Name);
    }

    protected static string TorchSharpInput(
        CompilerOnnxStep operation,
        int index,
        CompilerTorchSharpPrintContext context
    )
    {
        var inputName = operation.Node.Inputs[index].Name;
        if (context.TryGetTensorInitializer(inputName, out var tensor)
            && tensor is not null
            && tensor.Dimensions.Count == 0
            && tensor.Values.Count == 1)
        {
            return tensor.Values[0] switch
            {
                CompilerFloatingPointLiteral value when value.ElementType == CompilerElementType.Float32 =>
                    value.Value.ToString("R", CultureInfo.InvariantCulture) + "f",
                CompilerFloatingPointLiteral value when value.ElementType == CompilerElementType.Float64 =>
                    value.Value.ToString("R", CultureInfo.InvariantCulture) + "d",
                CompilerFloatingPointLiteral value when value.ElementType is CompilerElementType.Float16 or CompilerElementType.BFloat16 =>
                    ((float)value.Value).ToString("R", CultureInfo.InvariantCulture) + "f",
                CompilerSignedIntegerLiteral value => value.Value.ToString(CultureInfo.InvariantCulture) + "L",
                CompilerUnsignedIntegerLiteral value => value.Value.ToString(CultureInfo.InvariantCulture) + "UL",
                _ => CompilerCSharpNaming.Identifier(inputName),
            };
        }

        return CompilerCSharpNaming.Identifier(inputName);
    }

    protected static IOnnxGraphEdge AddFloatInitializer(OnnxGraph graph, CompilerOnnxStep operation, string suffix, double value)
    {
        var inputName = operation.Node.Inputs[0].Name;
        var elementType = graph.GetValue(inputName) switch
        {
            OnnxValue { Type: OnnxTensorType tensorType } => tensorType.Type,
            OnnxTensor tensor => tensor.DataType,
            _ => null,
        };
        var name = $"{operation.Name}__{suffix}";
        var isFloat16 = string.Equals(elementType?.Name, "Half", StringComparison.Ordinal);
        var initializerType = elementType == typeof(double) ? typeof(double) : typeof(float);
        var typedValue = initializerType == typeof(double) ? value : (double)(float)value;

        if (graph.GetValue(name) is { } existingValue)
        {
            if (existingValue is OnnxTensor existingTensor
                && existingTensor.DataType == initializerType
                && existingTensor.Shape.Length == 0
                && existingTensor.Values.Count == 1
                && existingTensor.Values[0] is IConvertible existingScalar
                && existingScalar.ToDouble(CultureInfo.InvariantCulture).Equals(typedValue))
            {
                return existingTensor;
            }

            var candidate = $"{name}__compiler_initializer";
            for (var collision = 1; graph.GetValue(candidate) is not null; collision++)
            {
                candidate = $"{name}__compiler_initializer_{collision}";
            }

            name = candidate;
        }

        if (elementType is null || elementType == typeof(float))
        {
            return graph.AddTensor(name, [], [(float)value]);
        }

        if (elementType == typeof(double))
        {
            return graph.AddTensor(name, [], [value]);
        }

        if (isFloat16)
        {
            var castLikeName = $"{name}__cast_like";
            for (var collision = 1; graph.GetValue(castLikeName) is not null; collision++)
            {
                castLikeName = $"{name}__cast_like_{collision}";
            }

            var castLike = new OnnxEdge(castLikeName);
            graph.AddNode(new OnnxNode(
                $"{name}__cast_like_node",
                "CastLike",
                string.Empty,
                string.Empty,
                [graph.GetValue(name) as OnnxTensor ?? graph.AddTensor(name, [], [(float)value]), Input(operation)],
                [castLike],
                []));
            return castLike;
        }

        throw new CompilerConversionException(
            CompilerDiagnosticCodes.Unsupported,
            $"TorchSharp composite '{operation.Descriptor.Name}' supports Float16/Float32/Float64 inputs, not '{elementType.Name}'.");
    }

    protected string? ValidateFloatingPointTensorInputs(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes,
        params CompilerElementType[] supportedTypes)
    {
        CompilerElementType? firstInputType = null;
        foreach (var input in node.Inputs)
        {
            if (string.IsNullOrWhiteSpace(input.Name)
                || !knownTypes.TryGetValue(input.Name, out var knownType)
                || knownType is not CompilerTensorType tensorType)
            {
                continue;
            }

            // Rank-zero initializers are compile-time scalar operands. Their source dtype is
            // cast to the tensor dtype by composite lowerings such as Logit and RoundDecimals.
            if (tensorType.Dimensions is { Count: 0 })
            {
                continue;
            }

            if (!supportedTypes.Contains(tensorType.ElementType))
            {
                var supported = string.Join("/", supportedTypes.Select(static type => type.ToString()));
                return $"Compiler '{OnnxName}' supports {supported} tensor inputs, not '{tensorType.ElementType}'.";
            }

            if (firstInputType is { } firstType && firstType != tensorType.ElementType)
            {
                return $"Compiler '{OnnxName}' requires matching tensor input element types; found '{firstType}' and '{tensorType.ElementType}'.";
            }

            firstInputType ??= tensorType.ElementType;
        }

        return null;
    }
}
