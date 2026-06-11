using Onnx;
using static Onnxify.ModelGenerator.Helpers.TextHelper;
using static Onnxify.ModelGenerator.OnnxModelGenerator;

namespace Onnxify.ModelGenerator.Services.TorchModuleInlineOperators;

internal abstract class TorchModuleInlineOperator
{
    internal abstract string OnnxOpType { get; }

    internal abstract string Emit(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, string> values
    );

    protected static string Input(
        TorchNodeSpecification node,
        IReadOnlyDictionary<string, string> values,
        int index
    )
    {
        if (index >= node.Inputs.Length)
        {
            throw new InvalidOperationException($"Node '{node.Name}' does not have input {index}.");
        }

        return values[node.Inputs[index]];
    }

    protected static string EmitAveragePool2d(TorchNodeSpecification node, string input)
    {
        var kernelShape = GetLongArrayAttribute(node, "kernel_shape", []);
        var strides = GetLongArrayAttribute(node, "strides", kernelShape);
        var pads = GetLongArrayAttribute(node, "pads", [0L, 0L, 0L, 0L]);
        var ceilMode = GetLongAttribute(node, "ceil_mode", 0L) != 0L;
        var countIncludePad = GetLongAttribute(node, "count_include_pad", 0L) != 0L;
        return $"torch.nn.functional.avg_pool2d({input}, kernel_size: {FormatModuleArgument(kernelShape)}, stride: {FormatModuleArgument(strides)}, padding: {FormatModuleArgument(pads.Take(2))}, ceil_mode: {FormatBool(ceilMode)}, count_include_pad: {FormatBool(countIncludePad)})";
    }

    protected static string EmitMaxPool2d(TorchNodeSpecification node, string input)
    {
        var kernelShape = GetLongArrayAttribute(node, "kernel_shape", []);
        var strides = GetLongArrayAttribute(node, "strides", kernelShape);
        var pads = GetLongArrayAttribute(node, "pads", [0L, 0L, 0L, 0L]);
        var dilations = GetLongArrayAttribute(node, "dilations", [1L, 1L]);
        var ceilMode = GetLongAttribute(node, "ceil_mode", 0L) != 0L;
        return $"torch.nn.functional.max_pool2d({input}, kernel_size: {FormatModuleArgument(kernelShape)}, stride: {FormatModuleArgument(strides)}, padding: {FormatModuleArgument(pads.Take(2))}, dilation: {FormatModuleArgument(dilations)}, ceil_mode: {FormatBool(ceilMode)})";
    }

    protected static string EmitLocalResponseNorm(TorchNodeSpecification node, string input)
    {
        var size = GetLongAttribute(node, "size", 0L);
        var alpha = GetFloatAttribute(node, "alpha", 0.0001f);
        var beta = GetFloatAttribute(node, "beta", 0.75f);
        var bias = GetFloatAttribute(node, "bias", 1.0f);
        return $"torch.nn.functional.local_response_norm({input}, {size}L, alpha: {FormatFloat(alpha)}f, beta: {FormatFloat(beta)}f, k: {FormatFloat(bias)}f)";
    }

    protected static string EmitGemm(
        TorchNodeSpecification node,
        string a,
        string b,
        string? c
    )
    {
        var alpha = GetFloatAttribute(node, "alpha", 1f);
        var beta = GetFloatAttribute(node, "beta", 1f);
        var transA = GetLongAttribute(node, "transA", 0L) != 0;
        var transB = GetLongAttribute(node, "transB", 0L) != 0;
        var left = transA ? $"{a}.transpose(0, 1)" : a;
        var right = transB ? $"{b}.transpose(0, 1)" : b;
        var expression = $"torch.matmul({left}, {right})";

        if (alpha != 1f)
        {
            expression = $"({FormatFloat(alpha)}f * {expression})";
        }

        if (c is not null)
        {
            var bias = beta == 1f ? c : $"({FormatFloat(beta)}f * {c})";
            expression = $"({expression} + {bias})";
        }

        return expression;
    }

    protected static string EmitClip(
        TorchNodeSpecification node,
        string input,
        string? min,
        string? max
    )
    {
        if (min is null && node.Attributes.TryGetValue("min", out var minAttribute))
        {
            min = FormatAttributeScalar(minAttribute);
        }

        if (max is null && node.Attributes.TryGetValue("max", out var maxAttribute))
        {
            max = FormatAttributeScalar(maxAttribute);
        }

        return $"{input}.clamp({min ?? "null"}, {max ?? "null"})";
    }

    protected static string EmitConv(
        TorchNodeSpecification node,
        string input,
        string weight,
        string bias
    )
    {
        var strides = GetLongArrayAttribute(node, "strides", [1L, 1L]);
        var pads = GetLongArrayAttribute(node, "pads", [0L, 0L, 0L, 0L]);
        var dilations = GetLongArrayAttribute(node, "dilations", [1L, 1L]);
        var group = GetLongAttribute(node, "group", 1L);
        var padding = pads.Length >= 2 ? pads.Take(pads.Length / 2).ToArray() : pads;

        return $"torch.nn.functional.conv2d({input}, {weight}, {bias}, {FormatLongArray(strides)}, {FormatLongArray(padding)}, {FormatLongArray(dilations)}, {group}L)";
    }

    protected static string EmitBatchNormalization(
        TorchNodeSpecification node,
        string input,
        string runningMean,
        string runningVar,
        string weight,
        string bias
    )
    {
        var epsilon = GetFloatAttribute(node, "epsilon", 1e-5f);
        var momentum = GetFloatAttribute(node, "momentum", 0.9f);

        return $"torch.nn.functional.batch_norm({input}, {runningMean}, {runningVar}, {weight}, {bias}, training: false, momentum: {FormatFloat(momentum)}f, eps: {FormatFloat(epsilon)}f)";
    }

    protected static string EmitConstant(TorchNodeSpecification node)
    {
        if (!node.Attributes.TryGetValue("value", out var value) || value is not TensorProto tensor)
        {
            throw new NotSupportedException($"Constant node '{node.Name}' does not contain a tensor value attribute.");
        }

        return FormatTensorProtoExpression(tensor);
    }

    protected static string FormatScalarType(TensorProto.Types.DataType dataType)
    {
        return dataType switch
        {
            TensorProto.Types.DataType.Float => "ScalarType.Float32",
            TensorProto.Types.DataType.Double => "ScalarType.Float64",
            TensorProto.Types.DataType.Uint8 => "ScalarType.Byte",
            TensorProto.Types.DataType.Int8 => "ScalarType.Int8",
            TensorProto.Types.DataType.Int16 => "ScalarType.Int16",
            TensorProto.Types.DataType.Int32 => "ScalarType.Int32",
            TensorProto.Types.DataType.Int64 => "ScalarType.Int64",
            TensorProto.Types.DataType.Bool => "ScalarType.Bool",
            _ => throw new NotSupportedException($"Unsupported Cast target tensor data type '{dataType}'."),
        };
    }
}
