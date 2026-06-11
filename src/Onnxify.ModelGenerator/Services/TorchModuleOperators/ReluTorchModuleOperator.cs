using System.Collections.Immutable;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

[TorchSharpOp("ReLU")]
internal sealed class ReluTorchModuleOperator : TorchModuleOperator
{
    internal override string OnnxOpType => "Relu";

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
            TorchModuleNodeKind.ReLU,
            "TorchModules.ReLU",
            "ReLU()",
            usedFieldNames,
            out module
        );
    }
}
