using System.Collections.Immutable;
using static Onnxify.ModelGenerator.Helpers.TextHelper;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

[TorchSharpOp("MaxPool2d")]
internal sealed class MaxPool2dTorchModuleOperator : TorchModuleOperator
{
    internal override string OnnxOpType => "MaxPool";

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

        var kernelShape = GetLongArrayAttribute(node, "kernel_shape", []);
        if (kernelShape.Length != 2)
        {
            return false;
        }

        var pads = GetLongArrayAttribute(node, "pads", [0L, 0L, 0L, 0L]);
        if (pads.Length != 4 || pads[0] != pads[2] || pads[1] != pads[3])
        {
            return false;
        }

        var strides = GetLongArrayAttribute(node, "strides", kernelShape);
        var dilations = GetLongArrayAttribute(node, "dilations", [1L, 1L]);
        var ceilMode = GetLongAttribute(node, "ceil_mode", 0L) != 0L;
        var fieldName = MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "maxPool"), usedFieldNames, "_module");
        module = new TorchModuleNodeSpecification(
            node.Name,
            TorchModuleNodeKind.MaxPool2d,
            fieldName,
            "TorchModules.MaxPool2d",
            $"MaxPool2d(kernel_size: {FormatModuleArgument(kernelShape)}, stride: {FormatModuleArgument(strides)}, padding: {FormatModuleArgument(pads.Take(2))}, dilation: {FormatModuleArgument(dilations)}, ceil_mode: {FormatBool(ceilMode)})",
            TransposeInput: false,
            []
        );
        return true;
    }
}
