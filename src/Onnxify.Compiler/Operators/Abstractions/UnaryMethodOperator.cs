using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class UnaryMethodOperator<TNode>(
    string name,
    string methodName,
    CompilerElementType? outputType = null,
    IReadOnlyList<CompilerElementType?>? inputTypes = null,
    IReadOnlyCollection<CompilerElementType>? supportedInputTypes = null)
    : UnaryOperator<TNode>(name, methodName)
    where TNode : OnnxNode
{
    public override CompilerElementType? OutputElementType { get; } = outputType;
    public override IReadOnlyList<CompilerElementType?> InputElementTypes { get; } = inputTypes ?? Array.Empty<CompilerElementType?>();

    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var baseError = base.ValidateOnnxNode(node, knownTypes);
        if (baseError is not null || supportedInputTypes is null)
        {
            return baseError;
        }

        return ValidateKnownInputElementTypes(
            node,
            knownTypes,
            supportedInputTypes,
            "the ONNX Runtime CPU kernel and TorchLib converter intersection does not include this type.");
    }
}

