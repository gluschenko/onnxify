using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class UnaryMethodOperator<TNode>(
    string name,
    string methodName,
    CompilerElementType? outputType = null,
    IReadOnlyList<CompilerElementType?>? inputTypes = null)
    : UnaryOperator<TNode>(name, methodName)
    where TNode : OnnxNode
{
    public override CompilerElementType? OutputElementType { get; } = outputType;
    public override IReadOnlyList<CompilerElementType?> InputElementTypes { get; } = inputTypes ?? Array.Empty<CompilerElementType?>();
}

