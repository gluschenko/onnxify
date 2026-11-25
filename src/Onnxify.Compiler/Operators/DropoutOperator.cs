namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::dropout")]
internal sealed class DropoutOperator() : CompilerOperator<Onnxify.Dropout>(CompilerOperatorIdentity.Onnx("Dropout"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [CompilerTorchSharpForm.Call("torch.nn.functional.dropout")];
    public override int MinimumInputCount => 1;
    public override int MaximumInputCount => 3;
    public override int MinimumOutputCount => 1;
    public override int MaximumOutputCount => 2;
    protected override bool AllowsEmptyOptionalInputs => true;
    protected override bool AllowsEmptyOptionalOutputs => true;
    protected override string PrintTorchSharp(Onnxify.Dropout node, CompilerSourceSpan? span)
    {
        if (node.Inputs.Count > 2 && !string.IsNullOrEmpty(node.Inputs[2].Name))
            throw Unsupported(span, "Training-mode Dropout requires runtime mask lowering; inference-mode Dropout is supported.");
        return CompilerCSharpNaming.Identifier(node.Inputs[0].Name);
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (context.CallName != "torch.nn.functional.dropout" || context.Receiver is not null || context.Invocation is not { Arguments.Count: >= 1 and <= 3 } invocation)
            return false;
        var data = context.RequireTensorReference(invocation.Arguments[0], OnnxName);
        var inputs = new List<CompilerValueReference> { data };
        if (invocation.Arguments.Count > 1)
        {
            if (!context.TryAddScalarInitializer(this, 0, invocation.Arguments[1], out var ratio))
                throw context.Unsupported(invocation.Arguments[1], "Dropout ratio must be a compile-time scalar.");
            inputs.Add(ratio);
        }
        if (invocation.Arguments.Count > 2)
        {
            if (invocation.Arguments[2] is not CompilerLiteralExpression { Literal: CompilerBooleanLiteral { Value: false } })
                throw context.Unsupported(invocation.Arguments[2], "Only inference-mode Dropout (training=false) is supported.");
            const string trainingName = "__dropout_training_false";
            context.Builder.AddStateMember(new CompilerStateMember(
                trainingName,
                CompilerStateMemberKind.Initializer,
                new CompilerTensorType(CompilerElementType.Boolean, Array.Empty<CompilerDimension>()),
                new CompilerTensorLiteral(CompilerElementType.Boolean, Array.Empty<CompilerDimension>(), [new CompilerBooleanLiteral(false)])));
            inputs.Add(new CompilerValueReference(trainingName));
        }
        context.AddOperation(this, inputs, span: expression.Span);
        return true;
    }
}
