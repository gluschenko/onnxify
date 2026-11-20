using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::round")]
[CompilerTorchOperator("aten::round.decimals")]
internal sealed class RoundOperator() : UnaryMethodOperator<Onnxify.Round>(
    "Round",
    "round",
    supportedInputTypes: PointwiseSupportedElementTypes.Float16AndFloatCpu)
{
    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (context.Invocation is null)
        {
            return base.TryScanTorchSharp(context, expression);
        }

        var input = context.Receiver ?? context.Invocation.Arguments.FirstOrDefault();
        var decimalsExpression = context.Receiver is null
            ? context.Invocation.Arguments.ElementAtOrDefault(1)
            : context.Invocation.Arguments.FirstOrDefault();
        if (input is null)
        {
            throw context.Unsupported(expression, "TorchSharp round requires one tensor input and an optional compile-time decimals argument.");
        }

        if (decimalsExpression is null || RoundDecimalsOperator.TryReadInteger(decimalsExpression, out var decimals) && decimals == 0)
        {
            if (decimalsExpression is not null)
            {
                var argumentCountIsValid = context.Receiver is null
                    ? context.Invocation.Arguments.Count == 2
                    : context.Invocation.Arguments.Count == 1;
                if (!argumentCountIsValid)
                {
                    throw context.Unsupported(expression, "TorchSharp round accepts only one optional decimals argument.");
                }
            }

            var standardInputReference = context.RequireTensorReference(input, OnnxName);
            context.AddOperation(this, [standardInputReference], span: expression.Span);
            return true;
        }

        if (!RoundDecimalsOperator.TryReadInteger(decimalsExpression, out decimals))
        {
            throw context.Unsupported(decimalsExpression, "TorchSharp round decimals must be a compile-time integer.");
        }

        var validArgumentCount = context.Receiver is null
            ? context.Invocation.Arguments.Count == 2
            : context.Invocation.Arguments.Count == 1;
        if (!validArgumentCount)
        {
            throw context.Unsupported(expression, "TorchSharp round accepts only one optional decimals argument.");
        }

        var inputReference = context.RequireTensorReference(input, OnnxName);
        var inputType = context.Inputs.FirstOrDefault(value => string.Equals(value.Name, inputReference.Name, StringComparison.Ordinal))?.Type;
        if (inputType is not CompilerTensorType { ElementType: CompilerElementType.Float16 or CompilerElementType.Float32 or CompilerElementType.Float64 } tensorType)
        {
            throw context.Unsupported(input, "TorchSharp round with decimals supports Float16, Float32, and Float64 tensors only.");
        }

        var scale = Math.Pow(10d, decimals);
        var floatScale = (float)scale;
        if (double.IsNaN(scale)
            || double.IsInfinity(scale)
            || scale == 0d
            || tensorType.ElementType == CompilerElementType.Float32 && (float.IsNaN(floatScale) || float.IsInfinity(floatScale) || floatScale == 0f)
            || tensorType.ElementType == CompilerElementType.Float16 && (scale > 65504d || scale < 2.98023223876953125e-8d))
        {
            throw context.Unsupported(decimalsExpression, $"TorchSharp round decimals '{decimals}' produces a scale outside the supported tensor dtype range.");
        }

        context.AddOperation(
            RoundDecimalsOperator.Instance,
            [inputReference],
            [new CompilerAttribute("decimals", new CompilerSignedIntegerLiteral(CompilerElementType.Int64, decimals))],
            expression.Span);
        return true;
    }
}
