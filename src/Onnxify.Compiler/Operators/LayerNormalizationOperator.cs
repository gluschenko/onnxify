using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::layer_norm")]
internal sealed class LayerNormalizationOperator() : CompilerOperator<Onnxify.LayerNormalization>(CompilerOperatorIdentity.Onnx("LayerNormalization"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [CompilerTorchSharpForm.Call("torch.nn.functional.layer_norm")];
    public override IReadOnlyList<string> AttributeNames { get; } = ["axis", "epsilon", "stash_type"];
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 3;
    public override int MinimumOutputCount => 1;
    public override int MaximumOutputCount => 3;
    protected override bool AllowsEmptyOptionalInputs => true;
    protected override bool AllowsEmptyOptionalOutputs => true;
    public override string PrintTorchSharp(CompilerOnnxStep operation, CompilerTorchSharpPrintContext context)
    {
        if (operation.Node is not Onnxify.LayerNormalization node) throw Unsupported(operation.Span, "Expected LayerNormalization node.");
        if (operation.Outputs.Skip(1).Any(output => context.IsValueUsed(output.Name)))
            throw Unsupported(operation.Span, "LayerNormalization Mean and InvStdDev outputs are not supported when consumed.");
        var x = CompilerCSharpNaming.Identifier(node.Inputs[0].Name);
        var scale = CompilerCSharpNaming.Identifier(node.Inputs[1].Name);
        var bias = node.Inputs.Count > 2 && !string.IsNullOrEmpty(node.Inputs[2].Name) ? CompilerCSharpNaming.Identifier(node.Inputs[2].Name) : "null";
        var axis = node.Attributes.FirstOrDefault(attribute => attribute.Name == "axis")?.GetValue() as long? ?? -1;
        var epsilon = node.Attributes.FirstOrDefault(attribute => attribute.Name == "epsilon")?.GetValue() as float? ?? 1e-5f;
        var stashType = node.Attributes.FirstOrDefault(attribute => attribute.Name == "stash_type")?.GetValue() as long? ?? 1;
        if (stashType != 1)
            throw Unsupported(operation.Span, "LayerNormalization TorchSharp lowering supports ONNX stash_type=1.");
        if (axis != -1)
            throw Unsupported(operation.Span, "LayerNormalization lowering currently supports normalization over the final tensor dimension.");
        return $"torch.nn.functional.layer_norm({x}, {scale}.shape, {scale}, {bias}, eps: {epsilon:R}f)";
    }
    protected override string PrintTorchSharp(Onnxify.LayerNormalization node, CompilerSourceSpan? span) => throw Unsupported(span, "LayerNormalization needs tree context to validate auxiliary outputs.");

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (context.CallName != "torch.nn.functional.layer_norm"
            || context.Receiver is not null
            || context.Invocation is not { Arguments.Count: >= 3 and <= 5 } invocation)
            return false;
        var input = context.RequireTensorReference(invocation.Arguments[0], OnnxName);
        var normalizedShape = StaticInt64Vector.ReadVector(invocation.Arguments[1], context, "LayerNormalization normalized_shape");
        if (normalizedShape.Length == 0 || normalizedShape.Any(static dimension => dimension <= 0))
            throw context.Unsupported(invocation.Arguments[1], "LayerNormalization normalized_shape must contain positive static dimensions.");
        var inputType = context.Inputs.FirstOrDefault(value => value.Name == input.Name)?.Type as CompilerTensorType;
        if (inputType?.Dimensions is not { } inputDimensions || inputDimensions.Count < normalizedShape.Length)
            throw context.Unsupported(invocation.Arguments[0], "LayerNormalization needs a known input rank to derive its ONNX axis.");
        var scale = context.RequireTensorReference(invocation.Arguments[2], OnnxName);
        var bias = invocation.Arguments.Count > 3 && invocation.Arguments[3] is not CompilerLiteralExpression { Literal: CompilerNullLiteral }
            ? new OnnxEdge(context.RequireTensorReference(invocation.Arguments[3], OnnxName).Name)
            : null;
        var epsilon = invocation.Arguments.Count > 4 ? context.RequireNumericLiteral(invocation.Arguments[4], "eps") : 1e-5f;
        var output = context.RequireSingleOutputEdge(this);
        context.AddOperation(this, new Onnxify.LayerNormalization("layer_normalization", new Onnxify.LayerNormalizationInputOutputOptions
        {
            X = new OnnxEdge(input.Name),
            Scale = new OnnxEdge(scale.Name),
            B = bias,
            Axis = inputDimensions.Count - normalizedShape.Length,
            Epsilon = epsilon,
            StashType = 1,
            Y = output,
        }), expression.Span);
        return true;
    }
}
