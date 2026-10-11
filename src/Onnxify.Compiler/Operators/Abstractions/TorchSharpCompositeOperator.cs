using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class TorchSharpCompositeOperator : CompilerOperator
{
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

    protected virtual bool AllowsScalarOperand(int index)
    {
        return false;
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        throw Unsupported(operation.Span, $"'{OnnxName}' is an export-only TorchSharp composite operation.");
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

    protected static IOnnxGraphEdge AddFloatInitializer(OnnxGraph graph, CompilerOnnxStep operation, string suffix, float value)
    {
        var inputName = operation.Node.Inputs[0].Name;
        var elementType = graph.GetValue(inputName) switch
        {
            OnnxValue { Type: OnnxTensorType tensorType } => tensorType.Type,
            OnnxTensor tensor => tensor.DataType,
            _ => null,
        };
        var name = $"{operation.Name}__{suffix}";

        if (elementType is null || elementType == typeof(float))
        {
            return graph.AddTensor(name, [], [value]);
        }

        if (elementType == typeof(double))
        {
            return graph.AddTensor(name, [], [(double)value]);
        }

        throw new CompilerConversionException(
            CompilerDiagnosticCodes.Unsupported,
            $"TorchSharp composite '{operation.Descriptor.Name}' supports Float32 and Float64 inputs, not '{elementType.Name}'.");
    }
}
