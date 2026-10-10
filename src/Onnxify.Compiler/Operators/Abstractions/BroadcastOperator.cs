using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class BroadcastOperator<TNode> : CompilerOperator<TNode>
    where TNode : OnnxNode
{
    protected BroadcastOperator(CompilerOperatorIdentity identity)
        : base(identity)
    {
    }

    public override bool SupportsMultidirectionalBroadcast => true;

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        return ValidateBroadcast(node, knownTypes);
    }
}

