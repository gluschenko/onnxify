using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class GenericOnnxOperator(string name) : CompilerOperator(CompilerOperatorIdentity.Onnx(name))
{
    public override Type NodeType => typeof(OnnxNode);

    public override CompilerOnnxStep ScanOnnx(OnnxNode node, string nodeName, CompilerSourceSpan? span) =>
        new(node, Descriptor, span, nodeName);

    public override CompilerOnnxStep CreateNode(
        string nodeName,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        IEnumerable<CompilerAttribute>? attributes = null,
        CompilerSourceSpan? span = null)
    {
        var node = new OnnxNode(
            nodeName,
            OnnxName,
            OnnxDomain,
            string.Empty,
            inputs.Select(static value => (IOnnxGraphEdge)new OnnxEdge(value.IsEmptyOptional ? string.Empty : value.Name)),
            outputs.Select(static value => (IOnnxGraphEdge)new OnnxEdge(value.IsEmptyOptional ? string.Empty : value.Name)),
            (attributes ?? Array.Empty<CompilerAttribute>())
                .Select(attribute => OnnxCompilerBackend.CreateOnnxAttribute(attribute, [], null)));
        return new CompilerOnnxStep(node, Descriptor, span, nodeName);
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation) =>
        throw Unsupported(operation.Span, $"TorchSharp emission for ONNX '{OnnxName}' is not implemented.");
}

internal sealed class QuantizeLinearOperator() : GenericOnnxOperator("QuantizeLinear")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 3;
    protected override bool AllowsEmptyOptionalInputs => true;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];
    public override IReadOnlyList<string> AttributeNames { get; } = ["axis", "block_size", "output_dtype"];

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        var input = QuantizationCompilerHelpers.Name(operation, 0);
        var scale = QuantizationCompilerHelpers.Name(operation, 1);
        var zeroPoint = operation.Inputs.Count > 2 ? QuantizationCompilerHelpers.Name(operation, 2) : "null";
        var zp = zeroPoint == "null" ? "0f" : $"{zeroPoint}.to(global::TorchSharp.torch.ScalarType.Float32)";
        // The fixture uses uint8 affine quantization. Keep the rounded code values as float
        // tensors because the following DequantizeLinear consumes them immediately.
        return $"torch.clamp(torch.round(torch.div({input}, {scale})) + {zp}, min: 0, max: 255)";
    }
}

internal sealed class DequantizeLinearOperator() : GenericOnnxOperator("DequantizeLinear")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override int MinimumInputCount => 2;
    public override int MaximumInputCount => 3;
    protected override bool AllowsEmptyOptionalInputs => true;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];
    public override IReadOnlyList<string> AttributeNames { get; } = ["axis", "block_size"];

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        var input = QuantizationCompilerHelpers.Name(operation, 0);
        var scale = QuantizationCompilerHelpers.Name(operation, 1);
        var zeroPoint = operation.Inputs.Count > 2 ? QuantizationCompilerHelpers.Name(operation, 2) : "null";
        var zp = zeroPoint == "null" ? "0f" : $"{zeroPoint}.to(global::TorchSharp.torch.ScalarType.Float32)";
        return $"torch.mul(torch.sub({input}.to(global::TorchSharp.torch.ScalarType.Float32), {zp}), {scale})";
    }
}

internal sealed class LrnOperator() : GenericOnnxOperator("LRN")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];
    public override IReadOnlyList<string> AttributeNames { get; } = ["alpha", "beta", "bias", "size"];

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        var node = operation.Node;
        var size = QuantizationCompilerHelpers.Attribute(node, "size", 5L);
        var alpha = QuantizationCompilerHelpers.Attribute(node, "alpha", 0.0001f);
        var beta = QuantizationCompilerHelpers.Attribute(node, "beta", 0.75f);
        var bias = QuantizationCompilerHelpers.Attribute(node, "bias", 1f);
        return $"torch.nn.functional.local_response_norm({QuantizationCompilerHelpers.Name(operation, 0)}, {size}L, alpha: {QuantizationCompilerHelpers.Float(alpha)}, beta: {QuantizationCompilerHelpers.Float(beta)}, k: {QuantizationCompilerHelpers.Float(bias)})";
    }
}

internal sealed class MaxPoolOperator() : GenericOnnxOperator("MaxPool")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];
    public override IReadOnlyList<string> AttributeNames { get; } = ["auto_pad", "ceil_mode", "dilations", "kernel_shape", "pads", "storage_order", "strides"];

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        var node = operation.Node;
        var kernel = QuantizationCompilerHelpers.Vector(node, "kernel_shape", [1L, 1L]);
        var strides = QuantizationCompilerHelpers.Vector(node, "strides", kernel);
        var pads = QuantizationCompilerHelpers.Vector(node, "pads", [0L, 0L, 0L, 0L]);
        var dilation = QuantizationCompilerHelpers.Vector(node, "dilations", [1L, 1L]);
        if (pads.Length == 2) pads = [pads[0], pads[1], pads[0], pads[1]];
        if (pads.Length != 4)
            throw Unsupported(operation.Span, "MaxPool requires two-dimensional pads in ONNX begin/end order.");
        var input = QuantizationCompilerHelpers.Name(operation, 0);
        var padding = new[] { pads[0], pads[1] };
        if (pads[0] != pads[2] || pads[1] != pads[3])
        {
            // ONNX pads with negative infinity for MaxPool. Use the same value when
            // decomposing asymmetric ONNX padding into TorchSharp's symmetric pool API.
            input = $"torch.nn.functional.pad({input}, {QuantizationCompilerHelpers.Longs([pads[1], pads[3], pads[0], pads[2]])}, global::TorchSharp.PaddingModes.Constant, float.NegativeInfinity)";
            padding = [0L, 0L];
        }
        var ceil = QuantizationCompilerHelpers.Attribute(node, "ceil_mode", 0L) is long ceilMode && ceilMode != 0 ? "true" : "false";
        if (QuantizationCompilerHelpers.Attribute(node, "storage_order", 0L) is long storageOrder && storageOrder != 0)
            throw Unsupported(operation.Span, "MaxPool storage_order values other than 0 are unsupported.");
        return $"torch.nn.functional.max_pool2d({input}, {QuantizationCompilerHelpers.Longs(kernel)}, {QuantizationCompilerHelpers.Longs(strides)}, {QuantizationCompilerHelpers.Longs(padding)}, {QuantizationCompilerHelpers.Longs(dilation)}, {ceil})";
    }
}

internal sealed class ConstantOperator() : GenericOnnxOperator("Constant")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override int MinimumInputCount => 0;
    public override int MaximumInputCount => 0;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];
    public override IReadOnlyList<string> AttributeNames { get; } = ["value", "value_float", "value_floats", "value_int", "value_ints", "value_string", "value_strings"];

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        var tensor = operation.Node.Attributes.FirstOrDefault(attribute => attribute.Name == "value")?.GetValue() as OnnxTensor;
        if (tensor is null)
        {
            throw Unsupported(operation.Span, "Constant TorchSharp emission currently requires a tensor-valued 'value' attribute.");
        }

        var type = tensor.DataType;
        var (elementName, torchType) = type == typeof(float) ? ("float", "Float32")
            : type == typeof(double) ? ("double", "Float64")
            : type == typeof(long) ? ("long", "Int64")
            : type == typeof(int) ? ("int", "Int32")
            : type == typeof(byte) ? ("byte", "UInt8")
            : type == typeof(sbyte) ? ("sbyte", "Int8")
            : type == typeof(short) ? ("short", "Int16")
            : type == typeof(ushort) ? ("ushort", "UInt16")
            : type == typeof(uint) ? ("uint", "UInt32")
            : type == typeof(ulong) ? ("ulong", "UInt64")
            : throw Unsupported(operation.Span, $"Constant dtype '{type.Name}' is unsupported.");
        var values = string.Join(", ", tensor.Values.Select(value =>
            Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture)
            + (type == typeof(float) ? "f" : type == typeof(long) ? "L" : string.Empty)));
        var shape = QuantizationCompilerHelpers.Longs(tensor.Shape);
        return $"torch.tensor(new {elementName}[] {{ {values} }}, {shape}, dtype: torch.ScalarType.{torchType})";
    }
}

internal static class QuantizationCompilerHelpers
{
    public static string Name(CompilerOnnxStep operation, int index) => CompilerCSharpNaming.Identifier(operation.Inputs[index].Name);
    public static object Attribute(OnnxNode node, string name, object fallback) =>
        node.Attributes.FirstOrDefault(attribute => attribute.Name == name)?.GetValue() ?? fallback;
    public static long[] Vector(OnnxNode node, string name, long[] fallback) =>
        node.Attributes.FirstOrDefault(attribute => attribute.Name == name)?.GetValue() as long[] ?? fallback;
    public static string Longs(IEnumerable<long> values) =>
        $"new long[] {{ {string.Join(", ", values.Select(value => value.ToString(System.Globalization.CultureInfo.InvariantCulture) + "L"))} }}";
    public static string Float(object value) =>
        Convert.ToSingle(value, System.Globalization.CultureInfo.InvariantCulture)
            .ToString("R", System.Globalization.CultureInfo.InvariantCulture) + "f";
}
