using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Operators;

internal sealed class IsInfOperator() : UnaryMethodOperator<Onnxify.IsInf>("IsInf", "isinf", CompilerElementType.Boolean)
{
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.isinf"),
        CompilerTorchSharpForm.Call("Tensor.isinf"),
        CompilerTorchSharpForm.Call("torch.isposinf"),
        CompilerTorchSharpForm.Call("torch.isneginf"),
    ];

    public override IReadOnlyList<string> AttributeNames { get; } = ["detect_negative", "detect_positive"];

    protected override string PrintTorchSharp(Onnxify.IsInf node, CompilerSourceSpan? span)
    {
        var detectNegative = node.DetectNegative ?? 1;
        var detectPositive = node.DetectPositive ?? 1;
        var function = (detectNegative, detectPositive) switch
        {
            (1, 1) => "isinf",
            (0, 1) => "isposinf",
            (1, 0) => "isneginf",
            _ => throw Unsupported(span, "ONNX IsInf must detect at least one infinity sign."),
        };

        return $"torch.{function}({CompilerCSharpNaming.Identifier(node.X.Name)})";
    }

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        var inputExpression = context.Receiver ?? invocation.Arguments.FirstOrDefault();
        if (inputExpression is null
            || (context.Receiver is null && invocation.Arguments.Count != 1)
            || (context.Receiver is not null && invocation.Arguments.Count != 0))
        {
            throw context.Unsupported(expression, $"TorchSharp call '{context.CallName}' requires one tensor input.");
        }

        var (detectNegative, detectPositive) = context.CallName switch
        {
            "torch.isinf" or "Tensor.isinf" => (1L, 1L),
            "torch.isposinf" => (0L, 1L),
            "torch.isneginf" => (1L, 0L),
            _ => throw context.Unsupported(expression, $"Unsupported TorchSharp IsInf form '{context.CallName}'."),
        };
        var input = context.RequireTensorReference(inputExpression, OnnxName);
        var node = new Onnxify.IsInf(
            OnnxName.ToLowerInvariant(),
            new Onnxify.IsInfInputOutputOptions
            {
                X = new OnnxEdge(input.Name),
                DetectNegative = detectNegative,
                DetectPositive = detectPositive,
                Y = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.IsInf isInf)
        {
            return $"IsInf requires the generated node type '{nameof(Onnxify.IsInf)}'.";
        }

        return isInf.DetectNegative is (null or 0 or 1) && isInf.DetectPositive is (null or 0 or 1)
            && (isInf.DetectNegative ?? 1) + (isInf.DetectPositive ?? 1) > 0
                ? null
                : "ONNX IsInf detection attributes must be 0 or 1 and at least one sign must be enabled.";
    }
}
