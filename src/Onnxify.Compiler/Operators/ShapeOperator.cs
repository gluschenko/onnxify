namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::sym_size.int")]
internal sealed class ShapeOperator() : CompilerOperator<Onnxify.Shape>(CompilerOperatorIdentity.Onnx("Shape"))
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("Tensor.shape"),
        CompilerTorchSharpForm.Call("Tensor.size"),
    ];
    public override IReadOnlyList<string> AttributeNames { get; } = ["start", "end"];
    public override CompilerElementType? OutputElementType => CompilerElementType.Int64;

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        if (operation.Node is Onnxify.Shape { Start: not null } or Onnxify.Shape { End: not null })
        {
            graph.AddNode(operation.Node);
            return;
        }

        // Shape-15 added start/end attributes. Keep older models on their original schema.
        var node = operation.Node;
        graph.AddNode(new OnnxNode(node.Name, node.OpType, node.Domain, node.DocString, node.Inputs, node.Outputs, node.Attributes));
    }
    protected override string PrintTorchSharp(Onnxify.Shape node, CompilerSourceSpan? span)
    {
        var input = CompilerCSharpNaming.Identifier(node.Inputs[0].Name);
        var start = node.Start ?? 0;
        var end = node.End;
        return $"ShapeWithRange({input}, {start}L, {(end is null ? "null" : $"{end}L")})";
    }
}
