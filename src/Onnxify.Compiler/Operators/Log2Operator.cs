using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::log2")]
internal sealed class Log2Operator() : TorchSharpCompositeOperator("Log2", "com.onnxify.compiler", "torch.log2")
{
    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var two = AddFloatInitializer(graph, operation, "two", 2f);
        var denominator = new OnnxEdge($"{operation.Name}__log_two");
        var numerator = new OnnxEdge($"{operation.Name}__log_input");
        graph.AddNode(new Onnxify.Log(
            $"{operation.Name}__log_two_node",
            new Onnxify.LogInputOutputOptions
            {
                Input = two,
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
