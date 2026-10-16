using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

internal sealed class LstmOperator : CompilerOperator
{
    private static readonly string[] _attributes =
    ["activation_alpha", "activation_beta", "activations", "clip", "direction", "hidden_size", "input_forget", "layout"];

    public LstmOperator() : base(CompilerOperatorIdentity.Onnx("LSTM")) { }

    public override Type NodeType => typeof(OnnxNode);
    // This is the compiler's canonical functional form. It deliberately describes ONNX LSTM
    // tensors directly, rather than routing the operator through Deep Export's module exporter.
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
        [CompilerTorchSharpForm.Call("torch.nn.functional.lstm")];
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override int MinimumInputCount => 3;
    public override int MaximumInputCount => 8;
    public override int MinimumOutputCount => 1;
    public override int MaximumOutputCount => 3;
    public override IReadOnlyList<string> AttributeNames => _attributes;
    protected override bool AllowsEmptyOptionalInputs => true;
    protected override bool AllowsEmptyOptionalOutputs => true;

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (context.CallName != "torch.nn.functional.lstm"
            || context.Receiver is not null
            || context.Invocation is not { Arguments.Count: >= 4 and <= 7 } invocation)
        {
            return false;
        }

        var arguments = invocation.Arguments;
        var inputs = new List<CompilerValueReference>
        {
            context.RequireTensorReference(arguments[0], OnnxName),
            context.RequireTensorReference(arguments[1], OnnxName),
            context.RequireTensorReference(arguments[2], OnnxName),
        };
        inputs.Add(arguments[3] is CompilerLiteralExpression { Literal: CompilerNullLiteral }
            ? new CompilerValueReference(string.Empty, isEmptyOptional: true)
            : context.RequireTensorReference(arguments[3], OnnxName));

        var hiddenSize = arguments.Count > 4
            ? StaticInt64Vector.ReadScalar(arguments[4], context, "LSTM hidden_size")
            : GetHiddenSizeFromWeight(arguments[2], context);
        if (hiddenSize <= 0)
        {
            throw context.Unsupported(expression, "LSTM hidden_size must be a positive compile-time integer.");
        }

        var direction = arguments.Count > 5 ? ReadString(arguments[5], context, "LSTM direction") : "forward";
        if (direction is not ("forward" or "reverse" or "bidirectional"))
        {
            throw context.Unsupported(arguments.Count > 5 ? arguments[5] : expression,
                "LSTM direction must be 'forward', 'reverse', or 'bidirectional'.");
        }

        var inputForget = arguments.Count > 6 && StaticInt64Vector.ReadScalar(arguments[6], context, "LSTM input_forget") != 0;
        context.AddOperation(this, inputs,
        [
            new CompilerAttribute("hidden_size", new CompilerSignedIntegerLiteral(CompilerElementType.Int64, hiddenSize)),
            new CompilerAttribute("direction", new CompilerStringLiteral(direction)),
            new CompilerAttribute("input_forget", new CompilerSignedIntegerLiteral(CompilerElementType.Int64, inputForget ? 1 : 0)),
        ], expression.Span);
        return true;
    }

    private static string ReadString(CompilerExpression expression, TorchSharpOperatorScanContext context, string name)
    {
        return expression is CompilerLiteralExpression { Literal: CompilerStringLiteral value }
            ? value.Value
            : throw context.Unsupported(expression, $"{name} must be a compile-time string literal.");
    }

    private static long GetHiddenSizeFromWeight(CompilerExpression expression, TorchSharpOperatorScanContext context)
    {
        if (expression is CompilerReferenceExpression reference)
        {
            var type = context.Inputs.FirstOrDefault(input => input.Name == reference.Name)?.Type as CompilerTensorType;
            if (type?.Dimensions?.LastOrDefault() is CompilerFixedDimension dimension)
            {
                return dimension.Value;
            }
        }

        throw context.Unsupported(expression, "LSTM hidden_size must be supplied explicitly when it cannot be inferred from recurrent weight metadata.");
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
            inputs.Select(static value => (IOnnxGraphEdge)new OnnxEdge(value.IsEmptyOptional ? string.Empty : value.Name)),
            outputs.Select(static value => (IOnnxGraphEdge)new OnnxEdge(value.IsEmptyOptional ? string.Empty : value.Name)),
            (attributes ?? []).Select(attribute => OnnxCompilerBackend.CreateOnnxAttribute(attribute, [], null)));
        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    public override CompilerOnnxStep ScanOnnx(OnnxNode node, string name, CompilerSourceSpan? span)
    {
        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        var node = operation.Node;
        var inputs = node.Inputs.Select(static edge => string.IsNullOrEmpty(edge.Name)
            ? "null"
            : CompilerCSharpNaming.Identifier(edge.Name)).ToArray();
        var hiddenSize = AttributeValue<long>(node, "hidden_size", 1L);
        var direction = AttributeValue<string>(node, "direction", "forward");
        var inputForget = AttributeValue<long>(node, "input_forget", 0L);
        var layout = AttributeValue<long>(node, "layout", 0L);
        var clip = AttributeValue<float?>(node, "clip", null);
        if (hiddenSize <= 0 || direction is not ("forward" or "reverse" or "bidirectional")
            || layout != 0 || clip is not null
            || AttributeValue<string[]>(node, "activations", []).Length != 0
            || AttributeValue<float[]>(node, "activation_alpha", []).Length != 0
            || AttributeValue<float[]>(node, "activation_beta", []).Length != 0
            || inputs.Length > 4 && inputs.Skip(4).Any(static input => input != "null"))
        {
            throw Unsupported(operation.Span, "TorchSharp LSTM lowering currently supports layout=0, default activations, no clipping, and omitted sequence/state/peephole inputs.");
        }

        var bias = inputs.Length > 3 ? inputs[3] : "null";
        var expression = $"RunOnnxLstm({inputs[0]}, {inputs[1]}, {inputs[2]}, {bias}, {hiddenSize}L, \"{direction}\", {inputForget}L)";
        return operation.Outputs.Count == 1 ? expression + "[0]" : expression;
    }

    public override string PrintTorchSharp(
        CompilerOnnxStep operation,
        CompilerTorchSharpPrintContext context)
    {
        var expression = PrintTorchSharp(operation);
        return operation.Outputs.Count > 1
            && operation.Outputs.Skip(1).All(output => !context.IsValueUsed(output.Name))
                ? expression + "[0]"
                : expression;
    }

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var direction = AttributeValue<string>(node, "direction", "forward");
        if (direction is not ("forward" or "reverse" or "bidirectional"))
        {
            return $"ONNX LSTM direction '{direction}' is not supported.";
        }

        var layout = AttributeValue<long>(node, "layout", 0L);
        if (layout != 0)
        {
            return "ONNX LSTM layout=1 is not supported by the current TorchSharp lowering.";
        }

        if (AttributeValue<string[]>(node, "activations", []).Length != 0
            || AttributeValue<float[]>(node, "activation_alpha", []).Length != 0
            || AttributeValue<float[]>(node, "activation_beta", []).Length != 0
            || AttributeValue<float?>(node, "clip", null) is not null
            || node.Inputs.Skip(4).Any(static input => !string.IsNullOrEmpty(input.Name)))
        {
            return "ONNX LSTM lowering currently supports default activations without clipping and omitted sequence/state/peephole inputs.";
        }

        return null;
    }

    private static T AttributeValue<T>(OnnxNode node, string name, T defaultValue)
    {
        var attribute = node.Attributes.FirstOrDefault(attribute => string.Equals(attribute.Name, name, StringComparison.Ordinal));
        if (attribute is null) return defaultValue;
        var value = attribute.GetValue();
        return value is T typed ? typed : defaultValue;
    }
}
