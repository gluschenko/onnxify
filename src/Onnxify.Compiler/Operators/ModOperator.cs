using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class ModOperator() : BinaryMethodOperator<Onnxify.Mod>("Mod", "remainder")
{
    public override IReadOnlyList<string> AttributeNames { get; } = ["fmod"];

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.Mod mod)
        {
            return "Mod requires the generated ONNX node type 'Mod'.";
        }

        if (mod.Fmod == 1)
        {
            return "TorchSharp remainder semantics do not match ONNX Mod with fmod=1.";
        }

        return mod.Fmod is null or 0
            ? ValidateBroadcast(node, knownTypes)
            : "ONNX Mod attribute 'fmod' must be 0 or 1.";
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        if (operation.Node is not Onnxify.Mod mod || mod.Fmod == 1)
        {
            throw Unsupported(operation.Span, "TorchSharp remainder export requires ONNX Mod with fmod=0 semantics.");
        }

        if (mod.Inputs.Count != 2 || mod.Outputs.Count != 1)
        {
            throw Unsupported(operation.Span, "TorchSharp remainder requires exactly two inputs and one output.");
        }

        var dividend = graph.GetValue(mod.A.Name) ?? new OnnxEdge(mod.A.Name);
        var divisor = graph.GetValue(mod.B.Name) ?? new OnnxEdge(mod.B.Name);
        var inputType = GetElementType(dividend);
        if (inputType is not null && inputType != typeof(float) && inputType != typeof(double))
        {
            if (inputType == typeof(byte)
                || inputType == typeof(sbyte)
                || inputType == typeof(short)
                || inputType == typeof(ushort)
                || inputType == typeof(int)
                || inputType == typeof(uint)
                || inputType == typeof(long)
                || inputType == typeof(ulong))
            {
                graph.AddNode(mod);
                return;
            }

            throw Unsupported(operation.Span, $"TorchSharp remainder export does not support ONNX element type '{inputType.Name}'.");
        }

        var quotient = new OnnxEdge($"{operation.Name}__quotient");
        var flooredQuotient = new OnnxEdge($"{operation.Name}__floored_quotient");
        var product = new OnnxEdge($"{operation.Name}__product");

        graph.AddNode(new Onnxify.Div(
            $"{operation.Name}__div",
            new Onnxify.DivInputOutputOptions
            {
                A = dividend,
                B = divisor,
                C = quotient,
            }));
        graph.AddNode(new Onnxify.Floor(
            $"{operation.Name}__floor",
            new Onnxify.FloorInputOutputOptions
            {
                X = quotient,
                Y = flooredQuotient,
            }));
        graph.AddNode(new Onnxify.Mul(
            $"{operation.Name}__mul",
            new Onnxify.MulInputOutputOptions
            {
                A = flooredQuotient,
                B = divisor,
                C = product,
            }));
        graph.AddNode(new Onnxify.Sub(
            operation.Name,
            new Onnxify.SubInputOutputOptions
            {
                A = dividend,
                B = product,
                C = new OnnxEdge(mod.Outputs[0].Name),
            }));
    }

    private static Type? GetElementType(IOnnxGraphEdge edge)
    {
        return edge switch
        {
            OnnxValue { Type: OnnxTensorType tensorType } => tensorType.Type,
            OnnxTensor tensor => tensor.DataType,
            _ => null,
        };
    }
}
