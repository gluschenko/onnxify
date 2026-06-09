using System.Collections.Immutable;
using static Onnxify.ModelGenerator.Helpers.TextHelper;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

internal sealed class BatchNorm2dTorchModuleOperator : TorchModuleOperator
{
    internal override string OnnxOpType => "BatchNormalization";

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
        if (node.Inputs.Length < 5
            || !initializers.TryGetValue(node.Inputs[1], out var scale)
            || !initializers.TryGetValue(node.Inputs[2], out var bias)
            || !initializers.TryGetValue(node.Inputs[3], out var mean)
            || !initializers.TryGetValue(node.Inputs[4], out var variance)
            || scale.Shape.Length != 1
            || scale.ClrTypeName != "float"
            || bias.ClrTypeName != "float"
            || mean.ClrTypeName != "float"
            || variance.ClrTypeName != "float")
        {
            return false;
        }

        var fieldName = MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "batchNorm"), usedFieldNames, "_module");
        module = new TorchModuleNodeSpecification(
            node.Name,
            TorchModuleNodeKind.BatchNorm2d,
            fieldName,
            "TorchModules.BatchNorm2d",
            $"BatchNorm2d({scale.Shape[0]})",
            TransposeInput: false,
            [
                $"LoadFloatTensor(tensors, \"{Escape(scale.OnnxName)}\", {fieldName}.weight!);",
                $"LoadFloatTensor(tensors, \"{Escape(bias.OnnxName)}\", {fieldName}.bias!);",
                $"LoadFloatTensor(tensors, \"{Escape(mean.OnnxName)}\", {fieldName}.running_mean);",
                $"LoadFloatTensor(tensors, \"{Escape(variance.OnnxName)}\", {fieldName}.running_var);",
            ]
        );
        consumedInitializers = [scale.OnnxName, bias.OnnxName, mean.OnnxName, variance.OnnxName];
        return true;
    }
}
