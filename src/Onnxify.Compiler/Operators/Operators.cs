using Onnxify;

namespace Onnxify.Compiler.Operators;

internal static class PointwiseSupportedElementTypes
{
    public static IReadOnlyCollection<CompilerElementType> Boolean { get; } = [CompilerElementType.Boolean];

    public static IReadOnlyCollection<CompilerElementType> FloatUnaryCpu { get; } =
    [CompilerElementType.Float32, CompilerElementType.Float64];

    public static IReadOnlyCollection<CompilerElementType> Abs { get; } =
    [CompilerElementType.Float32, CompilerElementType.Float64, CompilerElementType.Int8,
        CompilerElementType.Int16, CompilerElementType.Int32, CompilerElementType.Int64, CompilerElementType.UInt8];

    public static IReadOnlyCollection<CompilerElementType> Neg { get; } =
    [CompilerElementType.Float32, CompilerElementType.Float64, CompilerElementType.Int8,
        CompilerElementType.Int16, CompilerElementType.Int32, CompilerElementType.Int64];

    public static IReadOnlyCollection<CompilerElementType> Sign { get; } =
    [CompilerElementType.Float16, CompilerElementType.Float32, CompilerElementType.Float64,
        CompilerElementType.BFloat16,
        CompilerElementType.Int8, CompilerElementType.Int16, CompilerElementType.Int32, CompilerElementType.Int64,
        CompilerElementType.UInt8];

    public static IReadOnlyCollection<CompilerElementType> Float16AndFloatCpu { get; } =
    [CompilerElementType.Float16, CompilerElementType.Float32, CompilerElementType.Float64];

    public static IReadOnlyCollection<CompilerElementType> Relu { get; } =
    [CompilerElementType.Float16, CompilerElementType.Float32, CompilerElementType.Float64,
        CompilerElementType.Int8, CompilerElementType.Int32];

    public static IReadOnlyCollection<CompilerElementType> Float32ActivationCpu { get; } = [CompilerElementType.Float32];

    public static IReadOnlyCollection<CompilerElementType> LeakyReluCpu { get; } =
    [CompilerElementType.Float16, CompilerElementType.Float32];

    public static IReadOnlyCollection<CompilerElementType> Comparisons { get; } = new HashSet<CompilerElementType>
    {
        CompilerElementType.Float32,
        CompilerElementType.Float64,
        CompilerElementType.Int8,
        CompilerElementType.Int16,
        CompilerElementType.Int32,
        CompilerElementType.Int64,
        CompilerElementType.UInt8,
        CompilerElementType.UInt16,
        CompilerElementType.UInt32,
        CompilerElementType.UInt64,
    };

    public static IReadOnlyCollection<CompilerElementType> Equality { get; } = new HashSet<CompilerElementType>(Comparisons)
    {
        CompilerElementType.Boolean,
    };

    public static IReadOnlyCollection<CompilerElementType> MaxMin { get; } = new HashSet<CompilerElementType>
    {
        CompilerElementType.Float16,
        CompilerElementType.Float32,
        CompilerElementType.Float64,
        CompilerElementType.Int8,
        CompilerElementType.Int32,
        CompilerElementType.Int64,
        CompilerElementType.UInt8,
    };

    public static IReadOnlyCollection<CompilerElementType> Pow { get; } = new HashSet<CompilerElementType>
    {
        CompilerElementType.Float16,
        CompilerElementType.Float32,
        CompilerElementType.Float64,
        CompilerElementType.Int32,
        CompilerElementType.Int64,
    };
}

// Groups operators that only configure shared behavior through constructor arguments or simple metadata overrides.
// Operators with custom scanning, validation, or emission logic live in dedicated files.
[CompilerTorchOperator("aten::abs")]
internal sealed class AbsOperator() : UnaryMethodOperator<Onnxify.Abs>("Abs", "abs", supportedInputTypes: PointwiseSupportedElementTypes.Abs);

[CompilerTorchOperator("aten::acosh")]
internal sealed class AcoshOperator() : UnaryMethodOperator<Onnxify.Acosh>("Acosh", "acosh", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::acos")]
internal sealed class AcosOperator() : UnaryMethodOperator<Onnxify.Acos>("Acos", "acos", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::add.Tensor")]
[CompilerTorchOperator("aten::add.Scalar")]
internal sealed class AddOperator() : BinaryArithmeticOperator<Onnxify.Add>("Add", "add", "+")
{
    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        if (broadcastError is not null)
        {
            return broadcastError;
        }

        return ValidateSameTypedNumericInputs(
            node,
            knownTypes,
            "the ONNXScript TorchSharp converter and ONNX Runtime CPU kernel intersection does not include this type.",
            RealOrUInt8CpuTypes);
    }
}

[CompilerTorchOperator("aten::asinh")]
internal sealed class AsinhOperator() : UnaryMethodOperator<Onnxify.Asinh>("Asinh", "asinh", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::asin")]
internal sealed class AsinOperator() : UnaryMethodOperator<Onnxify.Asin>("Asin", "asin", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::atanh")]
internal sealed class AtanhOperator() : UnaryMethodOperator<Onnxify.Atanh>("Atanh", "atanh", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::atan")]
internal sealed class AtanOperator() : UnaryMethodOperator<Onnxify.Atan>("Atan", "atan", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::logical_and")]
internal sealed class AndOperator()
    : BinaryMethodOperator<Onnxify.And>("And", "logical_and", inputTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(node, knownTypes, PointwiseSupportedElementTypes.Boolean,
            "the ONNX Runtime CPU logical kernel accepts Boolean inputs only.");
    }
}

[CompilerTorchOperator("aten::ceil")]
internal sealed class CeilOperator() : UnaryMethodOperator<Onnxify.Ceil>("Ceil", "ceil", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::cosh")]
internal sealed class CoshOperator() : UnaryMethodOperator<Onnxify.Cosh>("Cosh", "cosh", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::cos")]
internal sealed class CosOperator() : UnaryMethodOperator<Onnxify.Cos>("Cos", "cos", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::div.Tensor")]
[CompilerTorchOperator("aten::div.Scalar")]
[CompilerTorchOperator("aten::divide.Tensor")]
[CompilerTorchOperator("aten::divide.Scalar")]
[CompilerTorchOperator("aten::true_divide.Tensor")]
[CompilerTorchOperator("aten::true_divide.Scalar")]
internal sealed class DivOperator() : BinaryArithmeticOperator<Onnxify.Div>("Div", "div", "/")
{
    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        if (broadcastError is not null)
        {
            return broadcastError;
        }

        return ValidateSameTypedNumericInputs(
            node,
            knownTypes,
            "TorchSharp aten_div promotion semantics differ because it promotes integer inputs to floating point.",
            FloatCpuTypes);
    }
}

[CompilerTorchOperator("aten::eq")]
[CompilerTorchOperator("aten::eq.Tensor")]
[CompilerTorchOperator("aten::eq.Scalar")]
internal sealed class EqualOperator() : BinaryOperator<Onnxify.Equal>("Equal", "eq", true, CompilerElementType.Boolean, binaryToken: "==")
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(
            node,
            knownTypes,
            PointwiseSupportedElementTypes.Equality,
            "the ONNX Runtime CPU Equal kernel does not register this element type.");
    }
}

[CompilerTorchOperator("aten::erf")]
[CompilerTorchOperator("aten::special_erf")]
internal sealed class ErfOperator() : UnaryMethodOperator<Onnxify.Erf>("Erf", "erf", supportedInputTypes: PointwiseSupportedElementTypes.Float16AndFloatCpu);

[CompilerTorchOperator("aten::exp")]
internal sealed class ExpOperator() : UnaryMethodOperator<Onnxify.Exp>("Exp", "exp", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::floor")]
internal sealed class FloorOperator() : UnaryMethodOperator<Onnxify.Floor>("Floor", "floor", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::gt.Tensor")]
[CompilerTorchOperator("aten::gt.Scalar")]
[CompilerTorchOperator("aten::greater.Tensor")]
internal sealed class GreaterOperator() : BinaryOperator<Onnxify.Greater>("Greater", "gt", true, CompilerElementType.Boolean, binaryToken: ">")
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes) => ValidateComparison(node, knownTypes);
    private static string? ValidateComparison(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = ValidateBroadcast(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(node, knownTypes, PointwiseSupportedElementTypes.Comparisons,
            "the ONNX Runtime CPU comparison kernel does not register this element type.");
    }
}

[CompilerTorchOperator("aten::ge.Tensor")]
[CompilerTorchOperator("aten::ge.Scalar")]
[CompilerTorchOperator("aten::greater_equal.Tensor")]
internal sealed class GreaterOrEqualOperator() : BinaryOperator<Onnxify.GreaterOrEqual>("GreaterOrEqual", "ge", true, CompilerElementType.Boolean, binaryToken: ">=")
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(node, knownTypes, PointwiseSupportedElementTypes.Comparisons,
            "the ONNX Runtime CPU comparison kernel does not register this element type.");
    }
}

[CompilerTorchOperator("aten::isnan")]
internal sealed class IsNaNOperator() : UnaryMethodOperator<Onnxify.IsNaN>("IsNaN", "isnan", CompilerElementType.Boolean);

[CompilerTorchOperator("aten::lt.Tensor")]
[CompilerTorchOperator("aten::lt.Scalar")]
[CompilerTorchOperator("aten::less.Tensor")]
internal sealed class LessOperator() : BinaryOperator<Onnxify.Less>("Less", "lt", true, CompilerElementType.Boolean, binaryToken: "<")
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(node, knownTypes, PointwiseSupportedElementTypes.Comparisons,
            "the ONNX Runtime CPU comparison kernel does not register this element type.");
    }
}

[CompilerTorchOperator("aten::le.Tensor")]
[CompilerTorchOperator("aten::le.Scalar")]
[CompilerTorchOperator("aten::less_equal.Tensor")]
internal sealed class LessOrEqualOperator() : BinaryOperator<Onnxify.LessOrEqual>("LessOrEqual", "le", true, CompilerElementType.Boolean, binaryToken: "<=")
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(node, knownTypes, PointwiseSupportedElementTypes.Comparisons,
            "the ONNX Runtime CPU comparison kernel does not register this element type.");
    }
}

[CompilerTorchOperator("aten::log")]
internal sealed class LogOperator() : UnaryMethodOperator<Onnxify.Log>("Log", "log", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::maximum")]
internal sealed class MaxOperator() : BinaryMethodOperator<Onnxify.Max>("Max", "maximum")
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(
            node,
            knownTypes,
            PointwiseSupportedElementTypes.MaxMin,
            "the ONNXScript aten_maximum converter and ONNX Runtime CPU kernel intersection does not include this type.");
    }
}

[CompilerTorchOperator("aten::minimum")]
internal sealed class MinOperator() : BinaryMethodOperator<Onnxify.Min>("Min", "minimum")
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(
            node,
            knownTypes,
            PointwiseSupportedElementTypes.MaxMin,
            "the ONNXScript aten_minimum converter and ONNX Runtime CPU kernel intersection does not include this type.");
    }
}

[CompilerTorchOperator("aten::mul")]
[CompilerTorchOperator("aten::mul.Tensor")]
[CompilerTorchOperator("aten::multiply.Tensor")]
internal sealed class MulOperator() : BinaryArithmeticOperator<Onnxify.Mul>("Mul", "mul", "*")
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateSameTypedNumericInputs(
            node,
            knownTypes,
            "the ONNXScript TorchSharp converter and ONNX Runtime CPU kernel intersection does not include this type.",
            RealOrUInt8CpuTypes);
    }
}

[CompilerTorchOperator("aten::logical_or")]
internal sealed class OrOperator()
    : BinaryMethodOperator<Onnxify.Or>("Or", "logical_or", inputTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(node, knownTypes, PointwiseSupportedElementTypes.Boolean,
            "the ONNX Runtime CPU logical kernel accepts Boolean inputs only.");
    }
}

[CompilerTorchOperator("aten::logical_xor")]
internal sealed class XorOperator()
    : BinaryMethodOperator<Onnxify.Xor>("Xor", "logical_xor", inputTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateKnownInputElementTypes(node, knownTypes, PointwiseSupportedElementTypes.Boolean,
            "the ONNX Runtime CPU logical kernel accepts Boolean inputs only.");
    }
}

[CompilerTorchOperator("aten::reciprocal")]
internal sealed class ReciprocalOperator() : UnaryMethodOperator<Onnxify.Reciprocal>("Reciprocal", "reciprocal", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::sign")]
internal sealed class SignOperator() : UnaryMethodOperator<Onnxify.Sign>("Sign", "sign", supportedInputTypes: PointwiseSupportedElementTypes.Sign);

[CompilerTorchOperator("aten::sinh")]
internal sealed class SinhOperator() : UnaryMethodOperator<Onnxify.Sinh>("Sinh", "sinh", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::sin")]
internal sealed class SinOperator() : UnaryMethodOperator<Onnxify.Sin>("Sin", "sin", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::sqrt")]
internal sealed class SqrtOperator() : UnaryMethodOperator<Onnxify.Sqrt>("Sqrt", "sqrt", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

[CompilerTorchOperator("aten::sub.Tensor")]
[CompilerTorchOperator("aten::sub.Scalar")]
[CompilerTorchOperator("aten::subtract.Tensor")]
[CompilerTorchOperator("aten::subtract.Scalar")]
internal sealed class SubOperator() : BinaryArithmeticOperator<Onnxify.Sub>("Sub", "sub", "-")
{
    public override string? ValidateOnnxNode(OnnxNode node, IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        var broadcastError = base.ValidateOnnxNode(node, knownTypes);
        return broadcastError ?? ValidateSameTypedNumericInputs(
            node,
            knownTypes,
            "the ONNXScript aten_sub real-type converter and ONNX Runtime CPU kernel intersection does not include this type.",
            RealCpuTypes);
    }
}

[CompilerTorchOperator("aten::tan")]
internal sealed class TanOperator() : UnaryMethodOperator<Onnxify.Tan>("Tan", "tan", supportedInputTypes: PointwiseSupportedElementTypes.FloatUnaryCpu);

