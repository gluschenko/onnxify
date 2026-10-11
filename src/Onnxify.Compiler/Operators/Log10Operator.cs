using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::log10")]
internal sealed class Log10Operator() : TorchSharpCompositeOperator("Log10", "com.onnxify.compiler", "torch.log10")
{
    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var ten = AddFloatInitializer(graph, operation, "ten", 10f);
        var denominator = new OnnxEdge($"{operation.Name}__log_ten");
        var numerator = new OnnxEdge($"{operation.Name}__log_input");
        graph.AddNode(new Onnxify.Log(
            $"{operation.Name}__log_ten_node",
            new Onnxify.LogInputOutputOptions
            {
                Input = ten,
                Output = denominator,
            }));
        graph.AddNode(new Onnxify.Log(
            $"{operation.Name}__log_input_node",
            new Onnxify.LogInputOutputOptions
            {
                Input = Input(operation),
                Output = numerator,
            }));
        graph.AddNode(new Onnxify.Div(
            operation.Name,
            new Onnxify.DivInputOutputOptions
            {
                A = numerator,
                B = denominator,
                C = Output(operation),
            }));
    }
}
