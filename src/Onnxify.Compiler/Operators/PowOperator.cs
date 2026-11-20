using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::pow.Tensor_Tensor")]
[CompilerTorchOperator("aten::pow.Tensor_Scalar")]
[CompilerTorchOperator("aten::pow.Scalar")]
internal sealed class PowOperator() : BinaryOperator<Onnxify.Pow>("Pow", "pow", true)
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        if (broadcastError is not null)
        {
            return broadcastError;
        }

        var inputTypeError = ValidateKnownInputElementTypes(
            node,
            knownTypes,
            PointwiseSupportedElementTypes.Pow,
            "ONNX Runtime CPU Pow supports Float16/32/64 and Int32/64 inputs in this build.",
            requireMatchingTypes: false);
        if (inputTypeError is not null)
        {
            return inputTypeError;
        }

        if (node.Inputs.Count >= 2
            && knownTypes.TryGetValue(node.Inputs[0].Name, out var baseType)
            && baseType is CompilerTensorType baseTensorType
            && knownTypes.TryGetValue(node.Inputs[1].Name, out var exponentType)
            && exponentType is CompilerTensorType exponentTensorType
            && (baseTensorType.ElementType == CompilerElementType.Float16
                || exponentTensorType.ElementType == CompilerElementType.Float16)
            && baseTensorType.ElementType != exponentTensorType.ElementType)
        {
            return "ONNX Runtime CPU Pow supports Float16 only when both inputs are Float16; mixed Float16 input types are not registered on CPU.";
        }

        if (node.Inputs.Count >= 1
            && node.Outputs.Count >= 1
            && knownTypes.TryGetValue(node.Inputs[0].Name, out baseType)
            && baseType is CompilerTensorType knownBaseType
            && knownTypes.TryGetValue(node.Outputs[0].Name, out var outputType)
            && outputType is CompilerTensorType outputTensorType
            && knownBaseType.ElementType != outputTensorType.ElementType)
        {
            return "ONNX Pow output element type must match its base input element type.";
        }

        return null;
    }

    protected override string PrintTorchSharp(Onnxify.Pow node, CompilerSourceSpan? span)
    {
        return $"{node.Inputs[0].Name}.pow({node.Inputs[1].Name})";
    }
}
