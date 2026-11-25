namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::softmax.int")]
internal sealed class SoftmaxOperator() : CompilerOperator<Onnxify.Softmax>(CompilerOperatorIdentity.Onnx("Softmax"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [CompilerTorchSharpForm.Call("Tensor.softmax")];
    public override IReadOnlyList<string> AttributeNames { get; } = ["axis"];
    protected override string PrintTorchSharp(Onnxify.Softmax node, CompilerSourceSpan? span) =>
        $"{CompilerCSharpNaming.Identifier(node.Input.Name)}.softmax({node.Axis ?? -1}L)";
}
