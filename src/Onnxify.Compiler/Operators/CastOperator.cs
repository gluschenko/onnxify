using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class CastOperator() : UnaryOperator<Onnxify.Cast>("Cast", "to_type", ["Tensor.to_type"])
{
    public override IReadOnlyList<string> AttributeNames { get; } = ["to"];
    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation || context.Receiver is null || invocation.Arguments.Count != 1
            || !CompilerOperatorRegistry.TryGetMemberPath(invocation.Arguments[0], out var targetType)
            || !TryGetTargetType(targetType, out var code))
        {
            throw context.Unsupported(expression, "Tensor.to_type requires a tensor receiver and a supported constant ScalarType value.");
        }

        var node = new Onnxify.Cast(
            name: OnnxName.ToLowerInvariant(),
            options: new CastInputOutputOptions
            {
                Input = new OnnxEdge(context.RequireTensorReference(context.Receiver, "Tensor.to_type").Name),
                RoundMode = null,
                Saturate = null,
                To = code,
                Output = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }

    private static bool TryGetTargetType(string path, out long code)
    {
        var name = path.Split('.').Last();
        (code, _) = name switch
        {
            "Float32" or "Float" => (1L, true),
            "UInt8" or "Byte" => (2L, true),
            "Int8" or "SByte" => (3L, true),
            "UInt16" => (4L, true),
            "Int16" or "Short" => (5L, true),
            "Int32" or "Int" => (6L, true),
            "Int64" or "Long" => (7L, true),
            "Bool" or "Boolean" => (9L, true),
            "Float16" or "Half" => (10L, true),
            "Float64" or "Double" => (11L, true),
            "UInt32" => (12L, true),
            "UInt64" => (13L, true),
            "BFloat16" => (16L, true),
            _ => (0L, false),
        };
        return code != 0;
    }

    public override CompilerElementType? GetTorchSharpOutputElementType(CompilerInvocationExpression invocation)
    {
        if (invocation.Arguments.Count != 1 || !CompilerOperatorRegistry.TryGetMemberPath(invocation.Arguments[0], out var targetType)
            || !TryGetTargetType(targetType, out var code))
        {
            return null;
        }

        return code switch
        {
            1 => CompilerElementType.Float32,
            2 => CompilerElementType.UInt8,
            3 => CompilerElementType.Int8,
            4 => CompilerElementType.UInt16,
            5 => CompilerElementType.Int16,
            6 => CompilerElementType.Int32,
            7 => CompilerElementType.Int64,
            9 => CompilerElementType.Boolean,
            10 => CompilerElementType.Float16,
            11 => CompilerElementType.Float64,
            12 => CompilerElementType.UInt32,
            13 => CompilerElementType.UInt64,
            16 => CompilerElementType.BFloat16,
            _ => null,
        };
    }
    protected override string PrintTorchSharp(Onnxify.Cast node, CompilerSourceSpan? span)
    {
        var typeName = node.To switch
        {
            1 => "Float32",
            2 => "UInt8",
            3 => "Int8",
            4 => "UInt16",
            5 => "Int16",
            6 => "Int32",
            7 => "Int64",
            9 => "Bool",
            10 => "Float16",
            11 => "Float64",
            12 => "UInt32",
            13 => "UInt64",
            16 => "BFloat16",
            _ => throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"ONNX Cast target element type '{node.To}' is unsupported.",
                CompilerDiagnosticStage.Emit,
                span),
        };
        return $"{node.Inputs[0].Name}.to_type(torch.ScalarType.{typeName})";
    }
}

