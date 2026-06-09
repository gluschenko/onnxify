using System.Collections.Immutable;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleOperators;

internal abstract class TorchModuleOperator
{
    internal abstract string OnnxOpType { get; }

    internal abstract bool TryCreateModuleNode(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, TorchInitializerSpecification> initializers,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module,
        out ImmutableArray<string> consumedInitializers
    );

    protected static bool TryCreateStatelessModuleNode(
        TorchNodeSpecification node,
        TorchModuleNodeKind kind,
        string fieldTypeName,
        string constructorExpression,
        HashSet<string> usedFieldNames,
        out TorchModuleNodeSpecification module
    )
    {
        module = new TorchModuleNodeSpecification(
            node.Name,
            kind,
            MakeUniqueIdentifier("_" + ToCamelIdentifier(node.Name, "module"), usedFieldNames, "_module"),
            fieldTypeName,
            constructorExpression,
            TransposeInput: false,
            []
        );
        return true;
    }
}
