using System.Globalization;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class TorchSharpOperatorScanContext
{
    public TorchSharpOperatorScanContext(
        string document,
        MethodDeclarationSyntax method,
        IReadOnlyList<CSharpSyntaxScanner.CompilerScanValue> inputs,
        IReadOnlyList<CSharpSyntaxScanner.CompilerScanValue> outputs,
        CompilerComputationTreeBuilder builder,
        CompilerInvocationExpression? invocation = null,
        CompilerExpression? receiver = null,
        string? callName = null)
    {
        Document = document;
        Method = method;
        Inputs = inputs;
        Outputs = outputs;
        Builder = builder;
        Invocation = invocation;
        Receiver = receiver;
        CallName = callName ?? string.Empty;
    }

    public string Document { get; }
    public MethodDeclarationSyntax Method { get; }
    public IReadOnlyList<CSharpSyntaxScanner.CompilerScanValue> Inputs { get; }
    public IReadOnlyList<CSharpSyntaxScanner.CompilerScanValue> Outputs { get; }
    public CompilerComputationTreeBuilder Builder { get; }
    public CompilerInvocationExpression? Invocation { get; }
    public CompilerExpression? Receiver { get; }
    public string CallName { get; }

    public CompilerValueReference RequireTensorReference(CompilerExpression expression, string operation)
    {
        if (expression is CompilerReferenceExpression reference
            && Inputs.Any(input => string.Equals(input.Name, reference.Name, StringComparison.Ordinal)))
        {
            return new CompilerValueReference(reference.Name);
        }

        throw Unsupported(expression, $"TorchSharp call '{operation}' requires tensor parameters for its tensor operands.");
    }

    public bool TryAddScalarInitializer(
        CompilerOperator compilerOperator,
        int index,
        CompilerExpression expression,
        out CompilerValueReference reference)
    {
        if (expression is CompilerLiteralExpression { Literal: CompilerScalarLiteral scalarLiteral })
        {
            var elementType = Inputs
                .Select(static input => input.Type)
                .OfType<CompilerTensorType>()
                .Select(static tensorType => (CompilerElementType?)tensorType.ElementType)
                .FirstOrDefault();
            if (elementType is null)
            {
                throw Unsupported(
                    expression,
                    $"TorchSharp operator '{compilerOperator.OnnxName}' cannot infer scalar dtype without a tensor parameter.");
            }

            var name = $"__{compilerOperator.OnnxName.ToLowerInvariant()}_scalar{index}";
            var value = ConvertScalarLiteral(scalarLiteral, elementType.Value, expression);
            var tensorLiteral = new CompilerTensorLiteral(elementType.Value, Array.Empty<CompilerDimension>(), [value]);
            Builder.AddStateMember(
                new CompilerStateMember(
                    name,
                    CompilerStateMemberKind.Initializer,
                    new CompilerTensorType(elementType.Value, Array.Empty<CompilerDimension>()),
                    tensorLiteral));
            reference = new CompilerValueReference(name);
            return true;
        }

        reference = new CompilerValueReference(string.Empty, isEmptyOptional: true);
        return false;
    }

    public float RequireNumericLiteral(CompilerExpression expression, string parameterName)
    {
        if (expression is CompilerLiteralExpression literal)
        {
            return literal.Literal switch
            {
                CompilerFloatingPointLiteral floatingPoint => (float)floatingPoint.Value,
                CompilerSignedIntegerLiteral integer => integer.Value,
                CompilerUnsignedIntegerLiteral integer => integer.Value,
                _ => throw Unsupported(expression, $"Gemm parameter '{parameterName}' must be a numeric compile-time constant."),
            };
        }

        throw Unsupported(expression, $"Gemm parameter '{parameterName}' must be a numeric compile-time constant.");
    }

    public OnnxEdge RequireSingleOutputEdge(CompilerOperator compilerOperator)
    {
        if (Outputs.Count != 1)
        {
            throw Unsupported(
                Method,
                $"TorchSharp operator '{compilerOperator.OnnxName}' requires exactly one output.");
        }

        return new OnnxEdge(Outputs[0].Name);
    }

    public void AddOperation(
        CompilerOperator compilerOperator,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerAttribute>? attributes = null,
        CompilerSourceSpan? span = null)
    {
        if (Outputs.Count != 1)
        {
            throw Unsupported(
                Method,
                $"TorchSharp operator '{compilerOperator.OnnxName}' requires exactly one output.");
        }

        var outputReferences = Outputs.Select(output => new CompilerValueReference(output.Name)).ToArray();
        Builder.AddOperation(
            compilerOperator.CreateNode(
                name: compilerOperator.OnnxName.ToLowerInvariant(),
                inputs: inputs,
                outputs: outputReferences,
                attributes: attributes,
                span: span));
    }

    public void AddOperation<TNode>(
        CompilerOperator<TNode> compilerOperator,
        TNode node,
        CompilerSourceSpan? span = null)
        where TNode : OnnxNode
    {
        if (Outputs.Count != 1)
        {
            throw Unsupported(
                Method,
                $"TorchSharp operator '{compilerOperator.OnnxName}' requires exactly one output.");
        }

        if (node.Outputs.Count != 1
            || !string.Equals(node.Outputs[0].Name, Outputs[0].Name, StringComparison.Ordinal))
        {
            throw Unsupported(
                Method,
                $"Typed ONNX node '{compilerOperator.OnnxName}' does not target the expected TorchSharp output '{Outputs[0].Name}'.");
        }

        Builder.AddOperation(new CompilerOnnxStep(node, compilerOperator.Descriptor, span));
    }

    public CSharpCompilerDiagnosticException Unsupported(SyntaxNode source, string message)
    {
        return new(
            CompilerDiagnosticCodes.Unsupported,
            message,
            CompilerDiagnosticStage.Analyze,
            CSharpCompilerFrontend.Span(Document, source.GetLocation()));
    }

    public CSharpCompilerDiagnosticException Unsupported(CompilerExpression source, string message)
    {
        return new(CompilerDiagnosticCodes.Unsupported, message, CompilerDiagnosticStage.Analyze, source.Span);
    }

    private static CompilerScalarLiteral ConvertScalarLiteral(
        CompilerScalarLiteral literal,
        CompilerElementType targetType,
        CompilerExpression source)
    {
        try
        {
            var value = literal switch
            {
                CompilerFloatingPointLiteral floatingPoint => (object)floatingPoint.Value,
                CompilerSignedIntegerLiteral integer => integer.Value,
                CompilerUnsignedIntegerLiteral integer => integer.Value,
                CompilerBooleanLiteral boolean => boolean.Value,
                _ => throw new InvalidCastException(),
            };
            return targetType switch
            {
                CompilerElementType.Float16 or CompilerElementType.BFloat16 or CompilerElementType.Float32 or CompilerElementType.Float64
                    => new CompilerFloatingPointLiteral(targetType, Convert.ToDouble(value, CultureInfo.InvariantCulture)),
                CompilerElementType.Int8 or CompilerElementType.Int16 or CompilerElementType.Int32 or CompilerElementType.Int64
                    => new CompilerSignedIntegerLiteral(targetType, Convert.ToInt64(value, CultureInfo.InvariantCulture)),
                CompilerElementType.UInt8 or CompilerElementType.UInt16 or CompilerElementType.UInt32 or CompilerElementType.UInt64
                    => new CompilerUnsignedIntegerLiteral(targetType, Convert.ToUInt64(value, CultureInfo.InvariantCulture)),
                _ => throw new CSharpCompilerDiagnosticException(
                    CompilerDiagnosticCodes.Unsupported,
                    $"Scalar operands are unsupported for tensor element type '{targetType}'.",
                    CompilerDiagnosticStage.Analyze,
                    source.Span),
            };
        }
        catch (Exception exception) when (exception is FormatException or InvalidCastException or OverflowException)
        {
            throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"Scalar operand cannot be represented as '{targetType}'.",
                CompilerDiagnosticStage.Analyze,
                source.Span);
        }
    }
}

