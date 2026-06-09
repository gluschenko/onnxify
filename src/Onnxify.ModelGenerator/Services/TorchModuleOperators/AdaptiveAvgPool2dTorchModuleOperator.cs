using System.Collections.Immutable;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

internal sealed class AdaptiveAvgPool2dTorchModuleOperator : TorchModuleOperator
{
    internal override string OnnxOpType => "GlobalAveragePool";

    internal override bool TryCreateModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    )
    {
        consumedInitializers = [];
        return TryCreateStatelessModuleNode(
            node,
            TorchModuleNodeKind.AdaptiveAvgPool2d,
            "TorchModules.AdaptiveAvgPool2d",
            "AdaptiveAvgPool2d(1)",
            usedFieldNames,
            out module
        );
    }
}
