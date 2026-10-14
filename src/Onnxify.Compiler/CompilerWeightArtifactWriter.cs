#if NET8_0_OR_GREATER
using Onnxify.Safetensors;

namespace Onnxify.Compiler;

internal static class CompilerWeightArtifactWriter
{
    public static byte[] WriteSafetensors(CompilerComputationTree tree)
    {
        var members = CompilerWeightSelection.Select(tree, CompilerWeightStorageKind.Safetensors);
        var archive = new SafeTensors();
        foreach (var member in members)
        {
            if (member.Value is not CompilerTensorLiteral tensor)
            {
                throw new InvalidOperationException($"State member '{member.Name}' is not a dense tensor and cannot be written to Safetensors.");
            }

            var values = CompilerElementTypeMap.ToArray(tensor);
            if (tensor.ElementType == CompilerElementType.BFloat16)
            {
                values = ((Onnxify.Data.Numerics.BFloat16[])values)
                    .Select(static value => BFloat16.FromEncoded(value.Value))
                    .ToArray();
            }

            var shape = tensor.Dimensions
                .Select(static dimension => checked((ulong)((CompilerFixedDimension)dimension).Value))
                .ToArray();
            archive.SetTensor(member.Name, values, shape);
        }

        return archive.Serialize();
    }
}
#endif
