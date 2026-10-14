namespace Onnxify.Compiler;

/// <summary>Selects state tensors according to their storage contract.</summary>
internal static class CompilerWeightSelection
{
    public static IReadOnlyList<CompilerStateMember> Select(
        CompilerComputationTree tree,
        CompilerWeightStorageKind storageKind
    )
    {
        var members = tree.Parameters
            .Concat(tree.Buffers)
            .Concat(tree.Initializers)
            .Where(static member => member.Value is CompilerTensorLiteral);

        return storageKind switch
        {
            CompilerWeightStorageKind.Onnx => members
                .Where(static member => member.IsOnnxInitializer)
                .ToArray(),
            CompilerWeightStorageKind.Safetensors => members.ToArray(),
            _ => throw new ArgumentOutOfRangeException(nameof(storageKind)),
        };
    }
}
