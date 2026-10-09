using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class SwishOperator() : ActivationOperator<Onnxify.Swish>("Swish", "torch.nn.functional.silu", ["alpha"])
{
    protected override string PrintTorchSharp(Onnxify.Swish node, CompilerSourceSpan? span)
    {
        var x = Input(node);
        var alpha = node.Alpha ?? 1f;
        return alpha == 1f ? $"({x} * {x}.sigmoid())" : $"({x} * ({x} * {Float(alpha)}).sigmoid())";
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        if (operation.Inputs.Count != 1 || operation.Outputs.Count != 1)
        {
            throw new CompilerConversionException(CompilerDiagnosticCodes.Unsupported, "Swish requires exactly one input and one output.");
        }

        if (operation.Node is not Onnxify.Swish node)
        {
            throw new CompilerConversionException(
                CompilerDiagnosticCodes.Unsupported,
                "Swish requires a generated ONNX Swish node.");
        }

        var input = node.X.Name;
        var sigmoidInput = input;
        var alpha = node.Alpha ?? 1f;
        if (alpha != 1f)
        {
            sigmoidInput = $"{operation.Name}__scaled";
            var scale = graph.AddTensor($"{operation.Name}__alpha", [], [alpha]);
            graph.Mul(
                name: $"{operation.Name}__scale",
                options: new MulInputOutputOptions
                {
                    A = new OnnxEdge(input),
                    B = scale,
                    C = new OnnxEdge(sigmoidInput),
                });
        }

        var sigmoidOutput = $"{operation.Name}__sigmoid";
        graph.Sigmoid(
            name: $"{operation.Name}__sigmoid_node",
            options: new SigmoidInputOutputOptions
            {
                X = new OnnxEdge(sigmoidInput),
                Y = new OnnxEdge(sigmoidOutput),
            });
        graph.Mul(
            name: operation.Name,
            options: new MulInputOutputOptions
            {
                A = new OnnxEdge(input),
                B = new OnnxEdge(sigmoidOutput),
                C = node.Y,
            });
    }
}
