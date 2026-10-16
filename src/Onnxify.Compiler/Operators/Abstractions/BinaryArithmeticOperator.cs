using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class BinaryArithmeticOperator<TNode> : BinaryOperator<TNode>
    where TNode : OnnxNode
{
    protected BinaryArithmeticOperator(string name, string methodName, string token)
        : base(name, methodName, supportsBroadcast: true, binaryToken: token)
    {
        OperatorTokenValue = token;
    }

    private string OperatorTokenValue { get; }
    protected override string OperatorToken => OperatorTokenValue;
}

