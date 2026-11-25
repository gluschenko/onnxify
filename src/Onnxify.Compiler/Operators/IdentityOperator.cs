namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::alias")]
internal sealed class IdentityOperator() : CompilerOperator<Onnxify.Identity>(CompilerOperatorIdentity.Onnx("Identity"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [CompilerTorchSharpForm.Call("Tensor.clone")];
    protected override string PrintTorchSharp(Onnxify.Identity node, CompilerSourceSpan? span) => CompilerCSharpNaming.Identifier(node.Inputs[0].Name);
}
