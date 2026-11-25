using Onnxify;
using Onnxify.Compiler.Operators.Helpers;

namespace Onnxify.Compiler.Operators;

internal sealed class GlobalAveragePoolOperator() : CompilerOperator<Onnxify.GlobalAveragePool>(CompilerOperatorIdentity.Onnx("GlobalAveragePool"))
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } =
    [
        CompilerTorchSharpForm.Call("torch.nn.functional.adaptive_avg_pool2d"),
    ];

    public override bool TryScanTorchSharp(TorchSharpOperatorScanContext context, CompilerExpression expression)
    {
        if (expression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        if (context.Receiver is not null || invocation.Arguments.Count != 2)
        {
            throw context.Unsupported(expression, "adaptive_avg_pool2d requires an input and a static output size.");
        }

        var outputSize = StaticInt64Vector.ReadVector(invocation.Arguments[1], context, "adaptive_avg_pool2d output size");
        if (!outputSize.SequenceEqual([1L, 1L]))
        {
            throw context.Unsupported(invocation.Arguments[1], "GlobalAveragePool supports only a static output size of [1, 1].");
        }

        var input = context.RequireTensorReference(invocation.Arguments[0], OnnxName);
        var node = new Onnxify.GlobalAveragePool(
            OnnxName.ToLowerInvariant(),
            new Onnxify.GlobalAveragePoolInputOutputOptions
            {
                X = new OnnxEdge(input.Name),
                Y = context.RequireSingleOutputEdge(this),
            });
        context.AddOperation(this, node, expression.Span);
        return true;
    }

    protected override string PrintTorchSharp(Onnxify.GlobalAveragePool node, CompilerSourceSpan? span)
    {
        return $"torch.nn.functional.adaptive_avg_pool2d({CompilerCSharpNaming.Identifier(node.X.Name)}, new long[] {{ 1L, 1L }})";
    }

    public override string? ValidateOnnxNode(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes)
    {
        if (node is not Onnxify.GlobalAveragePool pool)
        {
            return $"GlobalAveragePool requires the generated node type '{nameof(Onnxify.GlobalAveragePool)}'.";
        }

        if (knownTypes.TryGetValue(pool.X.Name, out var inputType)
            && inputType is CompilerTensorType { Dimensions: { Count: not 4 } })
        {
            return "Compiler GlobalAveragePool currently supports rank-4 NCHW tensors only.";
        }

        return null;
    }
}
