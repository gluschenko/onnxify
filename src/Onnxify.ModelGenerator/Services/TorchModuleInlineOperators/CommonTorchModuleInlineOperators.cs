using Onnx;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleInlineOperators;

internal sealed class AddTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Add";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)} + {Input(node, values, 1)}";
}

internal sealed class SubTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Sub";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)} - {Input(node, values, 1)}";
}

internal sealed class MulTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Mul";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)} * {Input(node, values, 1)}";
}

internal sealed class DivTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Div";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)} / {Input(node, values, 1)}";
}

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

internal sealed class LogTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Log";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.log()";
}

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

internal sealed class SigmoidTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Sigmoid";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.sigmoid()";
}

internal sealed class TanhTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Tanh";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.tanh()";
}

internal sealed class EluTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Elu";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.elu({Input(node, values, 0)}, alpha: {FormatFloat(GetFloatAttribute(node, "alpha", 1.0f))}f)";
}

internal sealed class HardSigmoidTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "HardSigmoid";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"({Input(node, values, 0)} * {FormatFloat(GetFloatAttribute(node, "alpha", 0.2f))}f + {FormatFloat(GetFloatAttribute(node, "beta", 0.5f))}f).clamp(0.0f, 1.0f)";
}

internal sealed class LeakyReluTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "LeakyRelu";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.nn.functional.leaky_relu({Input(node, values, 0)}, negative_slope: {FormatFloat(GetFloatAttribute(node, "alpha", 0.01f))}f)";
}

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

internal sealed class MatMulTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "MatMul";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.matmul({Input(node, values, 0)}, {Input(node, values, 1)})";
}

internal sealed class GemmTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Gemm";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitGemm(node, Input(node, values, 0), Input(node, values, 1), node.Inputs.Length > 2 ? Input(node, values, 2) : null);
}

internal sealed class ReshapeTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Reshape";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.reshape({Input(node, values, 1)}.data<long>().ToArray())";
}

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

internal sealed class AveragePoolTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "AveragePool";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => EmitAveragePool2d(node, Input(node, values, 0));
}

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

internal sealed class ShapeTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Shape";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"CreateShapeTensor({Input(node, values, 0)})";
}

internal sealed class GatherTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Gather";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"GatherTensor({Input(node, values, 0)}, {Input(node, values, 1)}, {GetLongAttribute(node, "axis", 0L)}L)";
}

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

internal sealed class ConcatTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Concat";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"ConcatTensors(new Tensor[] {{ {string.Join(", ", node.Inputs.Select(x => values[x]))} }}, {GetLongAttribute(node, "axis", 0L)}L)";
}

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

internal sealed class EqualTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Equal";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"{Input(node, values, 0)}.eq({Input(node, values, 1)})";
}

internal sealed class WhereTorchModuleInlineOperator : TorchModuleInlineOperator
{
    internal override string OnnxOpType => "Where";
    internal override string Emit(TorchNodeSpecification node, IReadOnlyDictionary<string, string> values) => $"torch.where({Input(node, values, 0)}, {Input(node, values, 1)}, {Input(node, values, 2)})";
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
