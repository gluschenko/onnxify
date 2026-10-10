using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::trunc")]
internal sealed class TruncOperator : CompilerOperator
{
    public TruncOperator()
        : base(CompilerOperatorIdentity.Onnx("Trunc"))
    {
    }

    public override CompilerOperationCapability Capability => CompilerOperationCapability.ExportOnly;

    public override IReadOnlyList<string> AttributeNames { get; } = [];

    public override Type NodeType => typeof(OnnxNode);

    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.trunc"),
        CompilerTorchSharpForm.Call("Tensor.trunc"),
    ];

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        return $"{operation.Node.Inputs[0].Name}.trunc()";
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

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        var operand = expression switch
        {
            CompilerUnaryExpression unary => unary.Expression,
            CompilerInvocationExpression invocation when invocation.Arguments.Count == 1 => invocation.Arguments[0],
            CompilerInvocationExpression invocation when context.Receiver is not null && invocation.Arguments.Count == 0 => context.Receiver,
            _ => throw context.Unsupported(expression, "TorchSharp operator 'Trunc' requires one tensor input."),
        };
        context.AddOperation(this, [context.RequireTensorReference(operand, OnnxName)], span: expression.Span);
        return true;
    }

    public override CompilerOnnxStep ScanOnnx(OnnxNode node, string name, CompilerSourceSpan? span)
    {
        if (node.OpType != OnnxName || node.Domain != OnnxDomain || node.GetType() != typeof(OnnxNode))
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                $"ONNX operator '{OnnxDomain}::{OnnxName}' cannot scan node '{node.Domain}::{node.OpType}'.");
        }

        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        if (operation.Node.Inputs.Count != 1 || operation.Node.Outputs.Count != 1)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                "TorchSharp truncation requires exactly one input and one output.");
        }

        var inputName = operation.Node.Inputs[0].Name;
        var outputName = operation.Node.Outputs[0].Name;
        var inputEdge = graph.GetValue(inputName) ?? new OnnxEdge(inputName);
        var inputType = inputEdge switch
        {
            OnnxValue { Type: OnnxTensorType tensorType } => tensorType.Type,
            OnnxTensor tensor => tensor.DataType,
            _ => typeof(float),
        };
        var one = inputType == typeof(double)
            ? (IOnnxGraphEdge)graph.AddTensor($"{operation.Name}__one", [], [1d])
            : inputType == typeof(float)
                ? graph.AddTensor($"{operation.Name}__one", [], [1f])
                : throw new CompilerConversionException(
                    CompilerDiagnosticCodes.Unsupported,
                    $"TorchSharp truncation supports Float32 and Float64 inputs, not '{inputType.Name}'.");

        var remainderName = $"{operation.Name}__remainder";
        graph.AddNode(
            new Onnxify.Mod(
                $"{operation.Name}__mod",
                new Onnxify.ModInputOutputOptions
                {
                    A = inputEdge,
                    B = one,
                    Fmod = 1,
                    C = new OnnxEdge(remainderName),
                }));
        graph.AddNode(
            new Onnxify.Sub(
                operation.Name,
                new Onnxify.SubInputOutputOptions
                {
                    A = inputEdge,
                    B = new OnnxEdge(remainderName),
                    C = new OnnxEdge(outputName),
                }));
    }
}
