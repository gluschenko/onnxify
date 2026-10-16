using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class BinaryMethodOperator<TNode>(
    string name,
    string methodName,
    CompilerElementType? outputType = null,
    IReadOnlyList<CompilerElementType?>? inputTypes = null)
    : BinaryOperator<TNode>(name, methodName, supportsBroadcast: true, outputType, inputTypes)
    where TNode : OnnxNode
{
    protected override string PrintTorchSharp(TNode node, CompilerSourceSpan? span)
    {
        return $"{node.Inputs[0].Name}.{MethodName}({node.Inputs[1].Name})";
    }
}

