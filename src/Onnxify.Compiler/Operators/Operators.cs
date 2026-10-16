using Onnxify;

namespace Onnxify.Compiler.Operators;

// Groups operators that only configure shared behavior through constructor arguments or simple metadata overrides.
// Operators with custom scanning, validation, or emission logic live in dedicated files.
[CompilerTorchOperator("aten::abs")]
internal sealed class AbsOperator() : UnaryMethodOperator<Onnxify.Abs>("Abs", "abs");

[CompilerTorchOperator("aten::acosh")]
internal sealed class AcoshOperator() : UnaryMethodOperator<Onnxify.Acosh>("Acosh", "acosh");

[CompilerTorchOperator("aten::acos")]
internal sealed class AcosOperator() : UnaryMethodOperator<Onnxify.Acos>("Acos", "acos");

[CompilerTorchOperator("aten::add.Tensor")]
[CompilerTorchOperator("aten::add.Scalar")]
internal sealed class AddOperator() : BinaryArithmeticOperator<Onnxify.Add>("Add", "add", "+");

[CompilerTorchOperator("aten::asinh")]
internal sealed class AsinhOperator() : UnaryMethodOperator<Onnxify.Asinh>("Asinh", "asinh");

[CompilerTorchOperator("aten::asin")]
internal sealed class AsinOperator() : UnaryMethodOperator<Onnxify.Asin>("Asin", "asin");

[CompilerTorchOperator("aten::atanh")]
internal sealed class AtanhOperator() : UnaryMethodOperator<Onnxify.Atanh>("Atanh", "atanh");

[CompilerTorchOperator("aten::atan")]
internal sealed class AtanOperator() : UnaryMethodOperator<Onnxify.Atan>("Atan", "atan");

[CompilerTorchOperator("aten::logical_and")]
internal sealed class AndOperator()
    : BinaryMethodOperator<Onnxify.And>("And", "logical_and", inputTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
}

[CompilerTorchOperator("aten::ceil")]
internal sealed class CeilOperator() : UnaryMethodOperator<Onnxify.Ceil>("Ceil", "ceil");

[CompilerTorchOperator("aten::cosh")]
internal sealed class CoshOperator() : UnaryMethodOperator<Onnxify.Cosh>("Cosh", "cosh");

[CompilerTorchOperator("aten::cos")]
internal sealed class CosOperator() : UnaryMethodOperator<Onnxify.Cos>("Cos", "cos");

[CompilerTorchOperator("aten::div.Tensor")]
[CompilerTorchOperator("aten::div.Scalar")]
[CompilerTorchOperator("aten::divide.Tensor")]
[CompilerTorchOperator("aten::divide.Scalar")]
[CompilerTorchOperator("aten::true_divide.Tensor")]
[CompilerTorchOperator("aten::true_divide.Scalar")]
internal sealed class DivOperator() : BinaryArithmeticOperator<Onnxify.Div>("Div", "div", "/");

[CompilerTorchOperator("aten::eq")]
[CompilerTorchOperator("aten::eq.Tensor")]
[CompilerTorchOperator("aten::eq.Scalar")]
internal sealed class EqualOperator() : BinaryOperator<Onnxify.Equal>("Equal", "eq", true, CompilerElementType.Boolean, binaryToken: "==");

[CompilerTorchOperator("aten::erf")]
[CompilerTorchOperator("aten::special_erf")]
internal sealed class ErfOperator() : UnaryMethodOperator<Onnxify.Erf>("Erf", "erf");

[CompilerTorchOperator("aten::exp")]
internal sealed class ExpOperator() : UnaryMethodOperator<Onnxify.Exp>("Exp", "exp");

[CompilerTorchOperator("aten::floor")]
internal sealed class FloorOperator() : UnaryMethodOperator<Onnxify.Floor>("Floor", "floor");

[CompilerTorchOperator("aten::gt.Tensor")]
[CompilerTorchOperator("aten::gt.Scalar")]
[CompilerTorchOperator("aten::greater.Tensor")]
internal sealed class GreaterOperator() : BinaryOperator<Onnxify.Greater>("Greater", "gt", true, CompilerElementType.Boolean, binaryToken: ">");

[CompilerTorchOperator("aten::ge.Tensor")]
[CompilerTorchOperator("aten::ge.Scalar")]
[CompilerTorchOperator("aten::greater_equal.Tensor")]
internal sealed class GreaterOrEqualOperator() : BinaryOperator<Onnxify.GreaterOrEqual>("GreaterOrEqual", "ge", true, CompilerElementType.Boolean, binaryToken: ">=");

[CompilerTorchOperator("aten::isnan")]
internal sealed class IsNaNOperator() : UnaryMethodOperator<Onnxify.IsNaN>("IsNaN", "isnan", CompilerElementType.Boolean);

[CompilerTorchOperator("aten::lt.Tensor")]
[CompilerTorchOperator("aten::lt.Scalar")]
[CompilerTorchOperator("aten::less.Tensor")]
internal sealed class LessOperator() : BinaryOperator<Onnxify.Less>("Less", "lt", true, CompilerElementType.Boolean, binaryToken: "<");

[CompilerTorchOperator("aten::le.Tensor")]
[CompilerTorchOperator("aten::le.Scalar")]
[CompilerTorchOperator("aten::less_equal.Tensor")]
internal sealed class LessOrEqualOperator() : BinaryOperator<Onnxify.LessOrEqual>("LessOrEqual", "le", true, CompilerElementType.Boolean, binaryToken: "<=");

[CompilerTorchOperator("aten::log")]
internal sealed class LogOperator() : UnaryMethodOperator<Onnxify.Log>("Log", "log");

[CompilerTorchOperator("aten::maximum")]
internal sealed class MaxOperator() : BinaryMethodOperator<Onnxify.Max>("Max", "maximum");

[CompilerTorchOperator("aten::minimum")]
internal sealed class MinOperator() : BinaryMethodOperator<Onnxify.Min>("Min", "minimum");

[CompilerTorchOperator("aten::mul")]
[CompilerTorchOperator("aten::mul.Tensor")]
[CompilerTorchOperator("aten::multiply.Tensor")]
internal sealed class MulOperator() : BinaryArithmeticOperator<Onnxify.Mul>("Mul", "mul", "*");

[CompilerTorchOperator("aten::logical_or")]
internal sealed class OrOperator()
    : BinaryMethodOperator<Onnxify.Or>("Or", "logical_or", inputTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
}

[CompilerTorchOperator("aten::logical_xor")]
internal sealed class XorOperator()
    : BinaryMethodOperator<Onnxify.Xor>("Xor", "logical_xor", inputTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
}

[CompilerTorchOperator("aten::reciprocal")]
internal sealed class ReciprocalOperator() : UnaryMethodOperator<Onnxify.Reciprocal>("Reciprocal", "reciprocal");

[CompilerTorchOperator("aten::round")]
internal sealed class RoundOperator() : UnaryMethodOperator<Onnxify.Round>("Round", "round");

[CompilerTorchOperator("aten::sign")]
internal sealed class SignOperator() : UnaryMethodOperator<Onnxify.Sign>("Sign", "sign");

[CompilerTorchOperator("aten::sinh")]
internal sealed class SinhOperator() : UnaryMethodOperator<Onnxify.Sinh>("Sinh", "sinh");

[CompilerTorchOperator("aten::sin")]
internal sealed class SinOperator() : UnaryMethodOperator<Onnxify.Sin>("Sin", "sin");

[CompilerTorchOperator("aten::sqrt")]
internal sealed class SqrtOperator() : UnaryMethodOperator<Onnxify.Sqrt>("Sqrt", "sqrt");

[CompilerTorchOperator("aten::sub.Tensor")]
[CompilerTorchOperator("aten::sub.Scalar")]
[CompilerTorchOperator("aten::subtract.Tensor")]
[CompilerTorchOperator("aten::subtract.Scalar")]
internal sealed class SubOperator() : BinaryArithmeticOperator<Onnxify.Sub>("Sub", "sub", "-");

[CompilerTorchOperator("aten::tan")]
internal sealed class TanOperator() : UnaryMethodOperator<Onnxify.Tan>("Tan", "tan");

