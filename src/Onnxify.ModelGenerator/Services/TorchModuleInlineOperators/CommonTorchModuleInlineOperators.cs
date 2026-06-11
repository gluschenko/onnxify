using Onnx;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleInlineOperators;

[TorchSharpOp("Add")]
internal sealed class AddTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Add";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)} + {Input(node, values, 1)}";
}

[TorchSharpOp("Sub")]
internal sealed class SubTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Sub";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)} - {Input(node, values, 1)}";
}

[TorchSharpOp("Mul")]
internal sealed class MulTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Mul";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)} * {Input(node, values, 1)}";
}

[TorchSharpOp("Div")]
internal sealed class DivTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Div";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)} / {Input(node, values, 1)}";
}

[TorchSharpOp("Pow")]
internal sealed class PowTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Pow";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.pow({Input(node, values, 1)})";
}

internal sealed class AbsTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Abs";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.abs()";
}

internal sealed class NegTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Neg";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"-{Input(node, values, 0)}";
}

internal sealed class ExpTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Exp";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.exp()";
}

[TorchSharpOp("Log")]
internal sealed class LogTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Log";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.log()";
}

[TorchSharpOp("Sqrt")]
internal sealed class SqrtTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Sqrt";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.sqrt()";
}

internal sealed class FloorTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Floor";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.floor()";
}

internal sealed class CeilTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Ceil";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.ceil()";
}

[TorchSharpOp("Round")]
internal sealed class RoundTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Round";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.round()";
}

[TorchSharpOp("Sign")]
internal sealed class SignTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Sign";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.sign()";
}

internal sealed class SinTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Sin";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.sin()";
}

internal sealed class CosTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Cos";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.cos()";
}

internal sealed class TanTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Tan";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.tan()";
}

[TorchSharpOp("Acos")]
internal sealed class AcosTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Acos";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.acos()";
}

[TorchSharpOp("Acosh")]
internal sealed class AcoshTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Acosh";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.acosh()";
}

[TorchSharpOp("Asin")]
internal sealed class AsinTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Asin";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.asin()";
}

[TorchSharpOp("Asinh")]
internal sealed class AsinhTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Asinh";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.asinh()";
}

[TorchSharpOp("Atan")]
internal sealed class AtanTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Atan";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.atan()";
}

[TorchSharpOp("Atanh")]
internal sealed class AtanhTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Atanh";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.atanh()";
}

internal sealed class ErfTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Erf";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.erf()";
}

internal sealed class ReciprocalTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Reciprocal";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.reciprocal()";
}

[TorchSharpOp("Sigmoid")]
internal sealed class SigmoidTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Sigmoid";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.sigmoid()";
}

[TorchSharpOp("Tanh")]
internal sealed class TanhTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Tanh";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.tanh()";
}

[TorchSharpOp("Elu")]
internal sealed class EluTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Elu";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.elu({Input(node, values, 0)}, alpha: {FormatFloat(GetFloatAttribute(node, "alpha", 1.0f))}f)";
}

[TorchSharpOp("HardSigmoid")]
internal sealed class HardSigmoidTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "HardSigmoid";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"({Input(node, values, 0)} * {FormatFloat(GetFloatAttribute(node, "alpha", 0.2f))}f + {FormatFloat(GetFloatAttribute(node, "beta", 0.5f))}f).clamp(0.0f, 1.0f)";
}

[TorchSharpOp("LeakyRelu")]
internal sealed class LeakyReluTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "LeakyRelu";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.leaky_relu({Input(node, values, 0)}, negative_slope: {FormatFloat(GetFloatAttribute(node, "alpha", 0.01f))}f)";
}

[TorchSharpOp("Softmax")]
internal sealed class SoftmaxTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Softmax";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.softmax({GetLongAttribute(node, "axis", -1L)}L)";
}

internal sealed class IdentityTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Identity";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => Input(node, values, 0);
}

internal sealed class CastTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Cast";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.to({FormatScalarType((TensorProto.Types.DataType)GetLongAttribute(node, "to", 0L))})";
}

[TorchSharpOp("MatMul")]
internal sealed class MatMulTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "MatMul";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.matmul({Input(node, values, 0)}, {Input(node, values, 1)})";
}

[TorchSharpOp("Max")]
internal sealed class MaxTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Max";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"MaxTensors(new Tensor[] {{ {string.Join(", ", node.Inputs.Select(x => values[x]))} }})";
}

[TorchSharpOp("Min")]
internal sealed class MinTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Min";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"MinTensors(new Tensor[] {{ {string.Join(", ", node.Inputs.Select(x => values[x]))} }})";
}

internal sealed class GemmTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Gemm";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitGemm(node, Input(node, values, 0), Input(node, values, 1), node.Inputs.Length > 2 ? Input(node, values, 2) : null);
}

[TorchSharpOp("Reshape")]
internal sealed class ReshapeTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Reshape";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.reshape({Input(node, values, 1)}.data<long>().ToArray())";
}

[TorchSharpOp("Flatten")]
internal sealed class FlattenTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Flatten";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.flatten({GetLongAttribute(node, "axis", 1L)})";
}

internal sealed class LrnTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "LRN";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitLocalResponseNorm(node, Input(node, values, 0));
}

[TorchSharpOp("AveragePool")]
internal sealed class AveragePoolTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "AveragePool";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitAveragePool2d(node, Input(node, values, 0));
}

[TorchSharpOp("ArgMax")]
internal sealed class ArgMaxTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "ArgMax";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.argmax({GetLongAttribute(node, "axis", 0L)}L, keepdim: {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})";
}

[TorchSharpOp("ArgMin")]
internal sealed class ArgMinTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "ArgMin";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.argmin({GetLongAttribute(node, "axis", 0L)}L, keepdim: {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})";
}

[TorchSharpOp("Celu")]
internal sealed class CeluTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Celu";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.celu({Input(node, values, 0)}, alpha: {FormatFloat(GetFloatAttribute(node, "alpha", 1.0f))}f)";
}

[TorchSharpOp("CumSum")]
internal sealed class CumSumTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "CumSum";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"CumSumTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {FormatBool(GetLongAttribute(node, "reverse", 0L) != 0L)}, {FormatBool(GetLongAttribute(node, "exclusive", 0L) != 0L)})";
}

[TorchSharpOp("DepthToSpace")]
internal sealed class DepthToSpaceTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "DepthToSpace";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"DepthToSpaceTensor({Input(node, values, 0)}, {GetLongAttribute(node, "blocksize", 1L)}L, \"{GetStringAttribute(node, "mode", "DCR")}\")";
}

[TorchSharpOp("Dropout")]
internal sealed class DropoutTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Dropout";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => Input(node, values, 0);
}

[TorchSharpOp("Expand")]
internal sealed class ExpandTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Expand";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.expand({Input(node, values, 1)}.data<long>().ToArray())";
}

[TorchSharpOp("GatherElements")]
internal sealed class GatherElementsTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "GatherElements";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.gather({GetLongAttribute(node, "axis", 0L)}L, {Input(node, values, 1)})";
}

[TorchSharpOp("Gelu")]
internal sealed class GeluTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Gelu";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.gelu({Input(node, values, 0)})";
}

[TorchSharpOp("GroupNormalization")]
internal sealed class GroupNormalizationTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "GroupNormalization";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.group_norm({Input(node, values, 0)}, {GetLongAttribute(node, "num_groups", 1L)}L, {(node.Inputs.Length > 1 ? Input(node, values, 1) : "null")}, {(node.Inputs.Length > 2 ? Input(node, values, 2) : "null")}, eps: {FormatFloat(GetFloatAttribute(node, "epsilon", 1e-5f))}f)";
}

[TorchSharpOp("HardSwish")]
internal sealed class HardSwishTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "HardSwish";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"({Input(node, values, 0)} * ({Input(node, values, 0)} + 3.0f).clamp(0.0f, 6.0f) / 6.0f)";
}

[TorchSharpOp("InstanceNormalization")]
internal sealed class InstanceNormalizationTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "InstanceNormalization";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.instance_norm({Input(node, values, 0)}, weight: {Input(node, values, 1)}, bias: {Input(node, values, 2)}, eps: {FormatFloat(GetFloatAttribute(node, "epsilon", 1e-5f))}f)";
}

[TorchSharpOp("LayerNormalization")]
internal sealed class LayerNormalizationTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "LayerNormalization";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"LayerNormTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {(node.Inputs.Length > 2 ? Input(node, values, 2) : "null")}, {GetLongAttribute(node, "axis", -1L)}L, {FormatFloat(GetFloatAttribute(node, "epsilon", 1e-5f))}f)";
}

[TorchSharpOp("LogSoftmax")]
internal sealed class LogSoftmaxTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "LogSoftmax";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.log_softmax({GetLongAttribute(node, "axis", -1L)}L)";
}

[TorchSharpOp("Mish")]
internal sealed class MishTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Mish";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"({Input(node, values, 0)} * {Input(node, values, 0)}.softplus().tanh())";
}

[TorchSharpOp("PRelu")]
internal sealed class PReluTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "PRelu";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.prelu({Input(node, values, 0)}, {Input(node, values, 1)})";
}

[TorchSharpOp("Pad")]
internal sealed class PadTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Pad";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"PadTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {(node.Inputs.Length > 2 ? Input(node, values, 2) : "null")}, \"{GetStringAttribute(node, "mode", "constant")}\")";
}

[TorchSharpOp("ReduceMax")]
internal sealed class ReduceMaxTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "ReduceMax";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => node.Inputs.Length > 1
        ? $"ReduceMaxTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})"
        : $"ReduceMaxTensor({Input(node, values, 0)}, {FormatNullableLongArray(GetLongArrayAttributeOrNull(node, "axes"))}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})";
}

[TorchSharpOp("ReduceMin")]
internal sealed class ReduceMinTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "ReduceMin";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => node.Inputs.Length > 1
        ? $"ReduceMinTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})"
        : $"ReduceMinTensor({Input(node, values, 0)}, {FormatNullableLongArray(GetLongArrayAttributeOrNull(node, "axes"))}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})";
}

[TorchSharpOp("ReduceProd")]
internal sealed class ReduceProdTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "ReduceProd";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => node.Inputs.Length > 1
        ? $"ReduceProdTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})"
        : $"ReduceProdTensor({Input(node, values, 0)}, {FormatNullableLongArray(GetLongArrayAttributeOrNull(node, "axes"))}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})";
}

[TorchSharpOp("Resize")]
internal sealed class ResizeTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Resize";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"ResizeTensor({Input(node, values, 0)}, {(node.Inputs.Length > 2 ? Input(node, values, 2) : "null")}, {(node.Inputs.Length > 3 ? Input(node, values, 3) : "null")}, \"{GetStringAttribute(node, "mode", "nearest")}\", \"{GetStringAttribute(node, "coordinate_transformation_mode", "half_pixel")}\")";
}

[TorchSharpOp("Selu")]
internal sealed class SeluTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Selu";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.selu({Input(node, values, 0)})";
}

[TorchSharpOp("Slice")]
internal sealed class SliceTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Slice";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"SliceTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {Input(node, values, 2)}, {(node.Inputs.Length > 3 ? Input(node, values, 3) : "null")}, {(node.Inputs.Length > 4 ? Input(node, values, 4) : "null")})";
}

[TorchSharpOp("Softplus")]
internal sealed class SoftplusTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Softplus";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.softplus()";
}

[TorchSharpOp("SpaceToDepth")]
internal sealed class SpaceToDepthTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "SpaceToDepth";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"SpaceToDepthTensor({Input(node, values, 0)}, {GetLongAttribute(node, "blocksize", 1L)}L)";
}

[TorchSharpOp("Split")]
internal sealed class SplitTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Split";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"SplitTensor({Input(node, values, 0)}, {(node.Inputs.Length > 1 ? Input(node, values, 1) : "null")}, {GetLongAttribute(node, "axis", 0L)}L, {node.Outputs.Length}L)";
}

[TorchSharpOp("Tile")]
internal sealed class TileTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Tile";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.repeat({Input(node, values, 1)}.data<long>().ToArray())";
}

[TorchSharpOp("TopK")]
internal sealed class TopKTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "TopK";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"TopKTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {GetLongAttribute(node, "axis", -1L)}L, {FormatBool(GetLongAttribute(node, "largest", 1L) != 0L)}, {FormatBool(GetLongAttribute(node, "sorted", 1L) != 0L)})";
}

[TorchSharpOp("Trilu")]
internal sealed class TriluTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Trilu";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"TriluTensor({Input(node, values, 0)}, {(node.Inputs.Length > 1 ? Input(node, values, 1) : "null")}, {FormatBool(GetLongAttribute(node, "upper", 1L) != 0L)})";
}

[TorchSharpOp("Transpose")]
internal sealed class TransposeTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Transpose";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.permute({FormatLongArray(GetLongArrayAttribute(node, "perm", []))})";
}

internal sealed class ClipTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Clip";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitClip(node, Input(node, values, 0), node.Inputs.Length > 1 ? Input(node, values, 1) : null, node.Inputs.Length > 2 ? Input(node, values, 2) : null);
}

internal sealed class ConvTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Conv";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitConv(node, Input(node, values, 0), Input(node, values, 1), node.Inputs.Length > 2 ? Input(node, values, 2) : "null");
}

internal sealed class MaxPoolTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "MaxPool";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitMaxPool2d(node, Input(node, values, 0));
}

internal sealed class BatchNormalizationTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "BatchNormalization";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitBatchNormalization(node, Input(node, values, 0), Input(node, values, 3), Input(node, values, 4), Input(node, values, 1), Input(node, values, 2));
}

internal sealed class GlobalAveragePoolTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "GlobalAveragePool";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.adaptive_avg_pool2d({Input(node, values, 0)}, new long[] {{ 1L, 1L }})";
}

[TorchSharpOp("Shape")]
internal sealed class ShapeTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Shape";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"CreateShapeTensor({Input(node, values, 0)})";
}

[TorchSharpOp("Gather")]
internal sealed class GatherTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Gather";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"GatherTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {GetLongAttribute(node, "axis", 0L)}L)";
}

[TorchSharpOp("Squeeze")]
internal sealed class SqueezeTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Squeeze";

    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values)
    {
        return node.Inputs.Length > 1
            ? $"SqueezeTensor({Input(node, values, 0)}, {Input(node, values, 1)})"
            : $"SqueezeTensor({Input(node, values, 0)}, {FormatLongArray(GetLongArrayAttribute(node, "axes", []))})";
    }
}

[TorchSharpOp("Unsqueeze")]
internal sealed class UnsqueezeTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Unsqueeze";

    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values)
    {
        return node.Inputs.Length > 1
            ? $"UnsqueezeTensor({Input(node, values, 0)}, {Input(node, values, 1)})"
            : $"UnsqueezeTensor({Input(node, values, 0)}, {FormatLongArray(GetLongArrayAttribute(node, "axes", []))})";
    }
}

[TorchSharpOp("Concat")]
internal sealed class ConcatTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Concat";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"ConcatTensors(new Tensor[] {{ {string.Join(", ", node.Inputs.Select(x => values[x]))} }}, {GetLongAttribute(node, "axis", 0L)}L)";
}

[TorchSharpOp("ReduceMean")]
internal sealed class ReduceMeanTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "ReduceMean";

    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values)
    {
        return node.Inputs.Length > 1
            ? $"ReduceMeanTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})"
            : $"ReduceMeanTensor({Input(node, values, 0)}, {FormatNullableLongArray(GetLongArrayAttributeOrNull(node, "axes"))}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})";
    }
}

[TorchSharpOp("ReduceSum")]
internal sealed class ReduceSumTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "ReduceSum";

    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values)
    {
        return node.Inputs.Length > 1
            ? $"ReduceSumTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})"
            : $"ReduceSumTensor({Input(node, values, 0)}, {FormatNullableLongArray(GetLongArrayAttributeOrNull(node, "axes"))}, {FormatBool(GetLongAttribute(node, "keepdims", 1L) != 0L)})";
    }
}

internal sealed class GreaterTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Greater";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.gt({Input(node, values, 1)})";
}

internal sealed class LessTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Less";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.lt({Input(node, values, 1)})";
}

[TorchSharpOp("GreaterOrEqual")]
internal sealed class GreaterOrEqualTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "GreaterOrEqual";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.ge({Input(node, values, 1)})";
}

[TorchSharpOp("LessOrEqual")]
internal sealed class LessOrEqualTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "LessOrEqual";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.le({Input(node, values, 1)})";
}

internal sealed class EqualTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Equal";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.eq({Input(node, values, 1)})";
}

[TorchSharpOp("Where")]
internal sealed class WhereTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Where";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.where({Input(node, values, 0)}, {Input(node, values, 1)}, {Input(node, values, 2)})";
}

[TorchSharpOp("Not")]
internal sealed class NotTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Not";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.logical_not()";
}

internal sealed class ConstantTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Constant";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitConstant(node);
}

internal sealed class QuantizeLinearTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "QuantizeLinear";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"QuantizeLinearTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {(node.Inputs.Length > 2 ? Input(node, values, 2) : "null")}, {GetLongAttribute(node, "axis", 1L)}L)";
}

internal sealed class DequantizeLinearTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "DequantizeLinear";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"DequantizeLinearTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {(node.Inputs.Length > 2 ? Input(node, values, 2) : "null")}, {GetLongAttribute(node, "axis", 1L)}L)";
}
