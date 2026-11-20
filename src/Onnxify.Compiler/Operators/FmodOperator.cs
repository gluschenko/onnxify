using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::fmod")]
internal sealed class FmodOperator() : TorchSharpCompositeOperator("Fmod", "com.onnxify.compiler", "torch.fmod", "Tensor.fmod")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 2;
    public override bool SupportsMultidirectionalBroadcast => true;

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        return $"torch.fmod({CompilerCSharpNaming.Identifier(operation.Node.Inputs[0].Name)}, {CompilerCSharpNaming.Identifier(operation.Node.Inputs[1].Name)})";
    }

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (!context.TryGetNode<Onnxify.Mod>(0, out var mod)
            || mod is null
            || mod.Fmod != 1
            || !context.CanCollapse(1))
        {
            return false;
        }

        // Trunc lowers to Mod(fmod=1, 1) followed by Sub(input, remainder).
        // Preserve that two-node mapping before claiming its first node as a plain Fmod.
        if (context.TryGetNode<Onnxify.Sub>(1, out var subtract)
            && subtract is not null
            && context.IsScalarConstant(mod.B.Name, 1d)
            && string.Equals(mod.A.Name, subtract.A.Name, StringComparison.Ordinal)
            && string.Equals(mod.C.Name, subtract.B.Name, StringComparison.Ordinal))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [mod.A.Name, mod.B.Name], mod.C.Name, 1);
        consumedNodeCount = 1;
        return true;
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node.Inputs.Count != 2 || node.Outputs.Count != 1)
        {
            return "Fmod requires exactly two inputs and one output.";
        }

        return ValidateBroadcast(node, knownTypes);
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        if (operation.Node.Inputs.Count != 2 || operation.Node.Outputs.Count != 1)
        {
            throw Unsupported(operation.Span, "Fmod requires exactly two tensor inputs and one output.");
        }

        graph.AddNode(new Onnxify.Mod(operation.Name, new Onnxify.ModInputOutputOptions
        {
            A = Input(operation, 0),
            B = Input(operation, 1),
            C = Output(operation),
            Fmod = 1,
        }));
    }
}
