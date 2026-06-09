using System.Collections.Immutable;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

internal sealed class Relu6TorchModuleOperator : TorchModuleOperator
{
    internal override string OnnxOpType => "Clip";

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
        if (!IsRelu6Pattern(node, initializers))
        {
            return false;
        }

        var consumed = ImmutableArray.CreateBuilder<string>();
        if (node.Inputs.Length > 1 && initializers.ContainsKey(node.Inputs[1]))
        {
            consumed.Add(node.Inputs[1]);
        }

        if (node.Inputs.Length > 2 && initializers.ContainsKey(node.Inputs[2]))
        {
            consumed.Add(node.Inputs[2]);
        }

        consumedInitializers = consumed.ToImmutable();
        return TryCreateStatelessModuleNode(
            node,
            TorchModuleNodeKind.ReLU6,
            "TorchModules.ReLU6",
            "ReLU6()",
            usedFieldNames,
            out module
        );
    }

    private static bool IsRelu6Pattern(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers
    )
    {
        if (node.OpType != "Clip")
        {
            return false;
        }

        var min = TryGetClipBound(node, initializers, 1, "min");
        var max = TryGetClipBound(node, initializers, 2, "max");
        return min == 0f && max == 6f;
    }

    private static float? TryGetClipBound(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        int inputIndex,
        string attributeName
    )
    {
        if (node.Inputs.Length > inputIndex
            && initializers.TryGetValue(node.Inputs[inputIndex], out var initializer)
            && initializer.ScalarFloatValue is not null)
        {
            return initializer.ScalarFloatValue.Value;
        }

        if (node.Attributes.TryGetValue(attributeName, out var attributeValue))
        {
            return attributeValue switch
            {
                float floatValue => floatValue,
                double doubleValue => (float)doubleValue,
                _ => null,
            };
        }

        return null;
    }
}
