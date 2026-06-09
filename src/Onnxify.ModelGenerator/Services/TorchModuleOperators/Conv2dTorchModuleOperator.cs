using System.Collections.Immutable;
using static Onnxify.ModelGenerator.Helpers.TextHelper;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

[TorchOp("Conv2d")]
internal sealed class Conv2dTorchModuleOperator : TorchModuleOperator
{
    internal override string OnnxOpType => "Conv";

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
            || weight.ClrTypeName != "float"
            || weight.Shape.Length != 4)
        {
            return false;
        }

        TorchInitializerSpecification? bias = null;
        if (node.Inputs.Length > 2
            && (!initializers.TryGetValue(node.Inputs[2], out bias) || bias.ClrTypeName != "float"))
        {
            return false;
        }

        var pads = GetLongArrayAttribute(node, "pads", [0L, 0L, 0L, 0L]);
        if (pads.Length != 4 || pads[0] != pads[2] || pads[1] != pads[3])
        {
            return false;
        }

        var strides = GetLongArrayAttribute(node, "strides", [1L, 1L]);
        var dilations = GetLongArrayAttribute(node, "dilations", [1L, 1L]);
        var group = GetLongAttribute(node, "group", 1L);
        var fieldName = MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "conv"), usedFieldNames, "_module");
        var constructor = $"Conv2d({weight.Shape[1] * group}, {weight.Shape[0]}, kernel_size: {FormatModuleArgument(weight.Shape.Skip(2))}, stride: {FormatModuleArgument(strides)}, padding: {FormatModuleArgument(pads.Take(2))}, dilation: {FormatModuleArgument(dilations)}, groups: {group}L, bias: {FormatBool(bias is not null)})";

        module = new TorchModuleNodeSpecification(
            node.Name,
            TorchModuleNodeKind.Conv2d,
            fieldName,
            "TorchModules.Conv2d",
            constructor,
            TransposeInput: false,
            [
                $"LoadFloatTensor(tensors, \"{Escape(weight.OnnxName)}\", {fieldName}.weight);",
                .. bias is null
                    ? []
                    : new[] { $"LoadFloatTensor(tensors, \"{Escape(bias.OnnxName)}\", {fieldName}.bias!);" },
            ]
        );
        consumedInitializers = bias is null ? [weight.OnnxName] : [weight.OnnxName, bias.OnnxName];
        return true;
    }
}
