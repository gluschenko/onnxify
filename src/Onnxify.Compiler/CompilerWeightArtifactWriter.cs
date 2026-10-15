using Onnxify.Safetensors;
using Onnxify.Data.Numerics;

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

            Array values;
            if (tensor.ElementType == CompilerElementType.Float16)
            {
                values = tensor.Values
                    .Cast<CompilerFloatingPointLiteral>()
                    .Select(static value => Float16.FromSingle((float)value.Value))
                    .ToArray();
            }
            else if (tensor.ElementType == CompilerElementType.BFloat16)
            {
                values = tensor.Values
                    .Cast<CompilerPackedScalarLiteral>()
                    .Select(static value => BFloat16.FromEncoded(checked((ushort)value.EncodedValue)))
                    .ToArray();
            }
            else
            {
                values = CompilerElementTypeMap.ToArray(tensor);
            }

            var shape = tensor.Dimensions
                .Select(static dimension => checked((ulong)((CompilerFixedDimension)dimension).Value))
                .ToArray();
            archive.SetTensor(member.Name, values, shape);
        }

        return archive.Serialize();
    }
}
