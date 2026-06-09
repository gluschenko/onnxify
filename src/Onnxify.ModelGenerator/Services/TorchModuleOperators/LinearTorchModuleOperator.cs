using System.Collections.Immutable;
using static Onnxify.ModelGenerator.Helpers.TextHelper;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

internal sealed class LinearTorchModuleOperator : TorchModuleOperator
{
    internal override string OnnxOpType => "Gemm";

    internal override bool TryCreateModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        module = null!;
        consumedInitializers = [];
        if (node.Inputs.Length < 2
            || !initializers.TryGetValue(node.Inputs[1], out var weight)
            || weight.Shape.Length != 2
            || weight.ClrTypeName != "float"
            || GetFloatAttribute(node, "alpha", 1f) != 1f
            || GetFloatAttribute(node, "beta", 1f) != 1f)
        {
            return false;
        }

        var transA = GetLongAttribute(node, "transA", 0L) != 0;
        var transB = GetLongAttribute(node, "transB", 0L) != 0;
        if (transA)
        {
            return false;
        }

        TorchInitializerSpecification? bias = null;
        if (node.Inputs.Length > 2
            && (!initializers.TryGetValue(node.Inputs[2], out bias) || bias.ClrTypeName != "float" || bias.Shape.Length != 1))
        {
            return false;
        }

        var inputFeatures = transB ? weight.Shape[1] : weight.Shape[0];
        var outputFeatures = transB ? weight.Shape[0] : weight.Shape[1];
        var fieldName = MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "linear"), usedFieldNames, "_module");
        module = new TorchModuleNodeSpecification(
            node.Name,
            TorchModuleNodeKind.Linear,
            fieldName,
            "TorchModules.Linear",
            $"Linear({inputFeatures}, {outputFeatures}, hasBias: {FormatBool(bias is not null)})",
            TransposeInput: false,
            [
                transB
                    ? $"LoadFloatTensor(tensors, \"{Escape(weight.OnnxName)}\", {fieldName}.weight);"
                    : $"LoadFloatTensorTransposed2D(tensors, \"{Escape(weight.OnnxName)}\", {fieldName}.weight);",
                .. bias is null
                    ? []
                    : new[] { $"LoadFloatTensor(tensors, \"{Escape(bias.OnnxName)}\", {fieldName}.bias!);" },
            ]
        );
        consumedInitializers = bias is null ? [weight.OnnxName] : [weight.OnnxName, bias.OnnxName];
        return true;
    }
}
