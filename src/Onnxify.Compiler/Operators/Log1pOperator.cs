using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::log1p")]
internal sealed class Log1pOperator() : TorchSharpCompositeOperator("Log1p", "com.onnxify.compiler", "torch.log1p")
{
    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var one = AddFloatInitializer(graph, operation, "one", 1f);
        var sum = new OnnxEdge($"{operation.Name}__sum");
        graph.AddNode(new Onnxify.Add(
            $"{operation.Name}__add",
            new Onnxify.AddInputOutputOptions
            {
                A = Input(operation),
                B = one,
                C = sum,
            }));
        graph.AddNode(new Onnxify.Log(
            operation.Name,
            new Onnxify.LogInputOutputOptions
            {
                Input = sum,
                Output = Output(operation),
            }));
    }
}
