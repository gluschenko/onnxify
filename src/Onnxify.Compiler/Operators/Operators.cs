using Onnxify;

namespace Onnxify.Compiler.Operators;

// Groups operators that only configure shared behavior through constructor arguments or simple metadata overrides.
// Operators with custom scanning, validation, or emission logic live in dedicated files.
internal sealed class AbsOperator() : UnaryMethodOperator<Onnxify.Abs>("Abs", "abs");

internal sealed class AcoshOperator() : UnaryMethodOperator<Onnxify.Acosh>("Acosh", "acosh");

internal sealed class AcosOperator() : UnaryMethodOperator<Onnxify.Acos>("Acos", "acos");

internal sealed class AddOperator() : BinaryArithmeticOperator<Onnxify.Add>("Add", "add", "+");

internal sealed class AsinhOperator() : UnaryMethodOperator<Onnxify.Asinh>("Asinh", "asinh");

internal sealed class AsinOperator() : UnaryMethodOperator<Onnxify.Asin>("Asin", "asin");

internal sealed class AtanhOperator() : UnaryMethodOperator<Onnxify.Atanh>("Atanh", "atanh");

internal sealed class AtanOperator() : UnaryMethodOperator<Onnxify.Atan>("Atan", "atan");

internal sealed class AndOperator()
    : BinaryMethodOperator<Onnxify.And>("And", "logical_and", inputTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
}

internal sealed class CeilOperator() : UnaryMethodOperator<Onnxify.Ceil>("Ceil", "ceil");

internal sealed class CoshOperator() : UnaryMethodOperator<Onnxify.Cosh>("Cosh", "cosh");

internal sealed class CosOperator() : UnaryMethodOperator<Onnxify.Cos>("Cos", "cos");

internal sealed class DivOperator() : BinaryArithmeticOperator<Onnxify.Div>("Div", "div", "/");

internal sealed class EqualOperator() : BinaryOperator<Onnxify.Equal>("Equal", "eq", true, CompilerElementType.Boolean, binaryToken: "==");

internal sealed class ErfOperator() : UnaryMethodOperator<Onnxify.Erf>("Erf", "erf");

internal sealed class ExpOperator() : UnaryMethodOperator<Onnxify.Exp>("Exp", "exp");

internal sealed class FloorOperator() : UnaryMethodOperator<Onnxify.Floor>("Floor", "floor");

internal sealed class GreaterOperator() : BinaryOperator<Onnxify.Greater>("Greater", "gt", true, CompilerElementType.Boolean, binaryToken: ">");

internal sealed class GreaterOrEqualOperator() : BinaryOperator<Onnxify.GreaterOrEqual>("GreaterOrEqual", "ge", true, CompilerElementType.Boolean, binaryToken: ">=");

internal sealed class IsNaNOperator() : UnaryMethodOperator<Onnxify.IsNaN>("IsNaN", "isnan", CompilerElementType.Boolean);

internal sealed class LessOperator() : BinaryOperator<Onnxify.Less>("Less", "lt", true, CompilerElementType.Boolean, binaryToken: "<");

internal sealed class LessOrEqualOperator() : BinaryOperator<Onnxify.LessOrEqual>("LessOrEqual", "le", true, CompilerElementType.Boolean, binaryToken: "<=");

internal sealed class LogOperator() : UnaryMethodOperator<Onnxify.Log>("Log", "log");

internal sealed class MaxOperator() : BinaryMethodOperator<Onnxify.Max>("Max", "maximum");

internal sealed class MinOperator() : BinaryMethodOperator<Onnxify.Min>("Min", "minimum");

internal sealed class MulOperator() : BinaryArithmeticOperator<Onnxify.Mul>("Mul", "mul", "*");

internal sealed class OrOperator()
    : BinaryMethodOperator<Onnxify.Or>("Or", "logical_or", inputTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
}

internal sealed class XorOperator()
    : BinaryMethodOperator<Onnxify.Xor>("Xor", "logical_xor", inputTypes: [CompilerElementType.Boolean, CompilerElementType.Boolean])
{
    public override CompilerElementType? OutputElementType => CompilerElementType.Boolean;
}

internal sealed class ReciprocalOperator() : UnaryMethodOperator<Onnxify.Reciprocal>("Reciprocal", "reciprocal");

internal sealed class RoundOperator() : UnaryMethodOperator<Onnxify.Round>("Round", "round");

internal sealed class SignOperator() : UnaryMethodOperator<Onnxify.Sign>("Sign", "sign");

internal sealed class SinhOperator() : UnaryMethodOperator<Onnxify.Sinh>("Sinh", "sinh");

internal sealed class SinOperator() : UnaryMethodOperator<Onnxify.Sin>("Sin", "sin");

internal sealed class SqrtOperator() : UnaryMethodOperator<Onnxify.Sqrt>("Sqrt", "sqrt");

internal sealed class SubOperator() : BinaryArithmeticOperator<Onnxify.Sub>("Sub", "sub", "-");

internal sealed class TanOperator() : UnaryMethodOperator<Onnxify.Tan>("Tan", "tan");

