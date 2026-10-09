using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class ActivationOperator<TNode> : CompilerOperator<TNode>
    where TNode : OnnxNode
{
    protected ActivationOperator(
        string name,
        string form,
        IEnumerable<string>? attributes = null,
        bool includeInstance = false,
        int inputCount = 1,
        IEnumerable<float>? fixedArguments = null)
        : base(CompilerOperatorIdentity.Onnx(name))
    {
        IEnumerable<string> names = includeInstance ? new[] { form, $"Tensor.{name.ToLowerInvariant()}" } : new[] { form };
        TorchSharpForms = names.Select(CompilerTorchSharpForm.Call).ToArray();
        AttributeNames = attributes?.ToArray() ?? Array.Empty<string>();
        MinimumInputCount = inputCount;
        MaximumInputCount = inputCount;
        FixedTorchSharpArguments = fixedArguments?.ToArray() ?? Array.Empty<float>();
    }

    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; }
    public override IReadOnlyList<string> AttributeNames { get; }
    public override int MinimumInputCount { get; }
    public override int MaximumInputCount { get; }
    public override IReadOnlyList<float> FixedTorchSharpArguments { get; }
    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            throw context.Unsupported(expression, $"TorchSharp activation '{OnnxName}' must be invoked as a method.");
        }

        var values = (context.Receiver is null ? Array.Empty<CompilerExpression>() : [context.Receiver]).Concat(invocation.Arguments).ToArray();
        var inputCount = InputCount;
        if (values.Length < inputCount + FixedTorchSharpArguments.Count
            || values.Length > inputCount + AttributeNames.Count + FixedTorchSharpArguments.Count)
        {
            throw context.Unsupported(
                expression,
                $"TorchSharp activation '{OnnxName}' requires {(InputCount == 1 ? "one tensor input" : $"{InputCount} tensor inputs")}, " +
                "one output, and supported literal activation arguments.");
        }

        var inputs = values.Take(inputCount).Select(value => context.RequireTensorReference(value, OnnxName)).ToArray();
        var attributes = new List<CompilerAttribute>();
        var attributeValues = values.Skip(inputCount).Take(AttributeNames.Count).ToArray();
        for (var index = 0; index < attributeValues.Length; index++)
        {
            if (attributeValues[index] is not CompilerLiteralExpression literal)
            {
                throw context.Unsupported(attributeValues[index], $"TorchSharp activation '{OnnxName}' requires compile-time literal attributes.");
            }

            attributes.Add(new CompilerAttribute(AttributeNames[index], literal.Literal));
        }
        var fixedValues = values.Skip(values.Length - FixedTorchSharpArguments.Count).ToArray();
        for (var index = 0; index < fixedValues.Length; index++)
        {
            if (fixedValues[index] is not CompilerLiteralExpression { Literal: CompilerFloatingPointLiteral number }
                || (float)number.Value != FixedTorchSharpArguments[index])
            {
                throw context.Unsupported(fixedValues[index], $"TorchSharp activation '{OnnxName}' only supports its fixed trailing arguments.");
            }
        }
        context.AddOperation(this, inputs, attributes, expression.Span);
        return true;
    }
    protected static string Input(TNode node)
    {
        return CompilerCSharpNaming.Identifier(node.Inputs[0].Name);
    }

}

