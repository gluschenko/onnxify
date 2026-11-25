namespace Onnxify.Compiler.Operators.Helpers;

internal static class StaticInt64Vector
{
    public static long[] ReadVector(CompilerExpression expression, TorchSharpOperatorScanContext context, string description)
    {
        if (expression is not CompilerArrayExpression array)
        {
            throw context.Unsupported(expression, $"{description} must be a compile-time integer array.");
        }

        return array.Items.Select(item => ReadValue(item, context, description)).ToArray();
    }

    public static long ReadScalar(CompilerExpression expression, TorchSharpOperatorScanContext context, string description)
    {
        return ReadValue(expression, context, description);
    }

    public static void AddInitializer(TorchSharpOperatorScanContext context, string name, IReadOnlyList<long> values)
    {
        context.Builder.AddStateMember(
            new CompilerStateMember(
                name,
                CompilerStateMemberKind.Initializer,
                new CompilerTensorType(CompilerElementType.Int64, [new CompilerFixedDimension(values.Count)]),
                new CompilerTensorLiteral(
                    CompilerElementType.Int64,
                    [new CompilerFixedDimension(values.Count)],
                    values.Select(static value => (CompilerScalarLiteral)new CompilerSignedIntegerLiteral(
                        CompilerElementType.Int64,
                        value)))));
    }

    public static string Format(IReadOnlyList<long> values)
    {
        return string.Join(", ", values.Select(static value => value.ToString(System.Globalization.CultureInfo.InvariantCulture) + "L"));
    }

    private static long ReadValue(CompilerExpression expression, TorchSharpOperatorScanContext context, string description)
    {
        return expression switch
        {
            CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signed } => signed.Value,
            CompilerLiteralExpression { Literal: CompilerUnsignedIntegerLiteral unsigned } when unsigned.Value <= long.MaxValue => (long)unsigned.Value,
            CompilerUnaryExpression { Operator: "-", Expression: CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral signed } }
                when signed.Value != long.MinValue => -signed.Value,
            _ => throw context.Unsupported(expression, $"{description} values must be compile-time integers."),
        };
    }
}
