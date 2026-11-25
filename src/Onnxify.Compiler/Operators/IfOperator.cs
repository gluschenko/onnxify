using Onnxify;

namespace Onnxify.Compiler.Operators;

/// <summary>Imports ONNX If subgraphs as lazy C# TorchSharp branches.</summary>
internal sealed class IfOperator() : CompilerOperator(CompilerOperatorIdentity.Onnx("If"))
{
    public override Type NodeType => typeof(OnnxNode);
    public override IReadOnlyCollection<CompilerTorchSharpForm> TorchSharpForms { get; } = [];
    public override CompilerOperationCapability Capability => CompilerOperationCapability.ImportOnly;
    public override int MinimumInputCount => 1;
    public override int MaximumInputCount => 1;
    public override int MinimumOutputCount => 1;
    public override int MaximumOutputCount => int.MaxValue;
    public override IReadOnlyList<string> AttributeNames { get; } = ["then_branch", "else_branch"];

    public override CompilerOnnxStep CreateNode(
        string name,
        IEnumerable<CompilerValueReference> inputs,
        IEnumerable<CompilerValueReference> outputs,
        IEnumerable<CompilerAttribute>? attributes = null,
        CompilerSourceSpan? span = null)
    {
        var node = new OnnxNode(
            name,
            OnnxName,
            OnnxDomain,
            string.Empty,
            inputs.Select(input => (IOnnxGraphEdge)new OnnxEdge(input.IsEmptyOptional ? string.Empty : input.Name)),
            outputs.Select(output => (IOnnxGraphEdge)new OnnxEdge(output.IsEmptyOptional ? string.Empty : output.Name)),
            (attributes ?? []).Select(attribute => OnnxCompilerBackend.CreateOnnxAttribute(attribute, [], null)));
        return new CompilerOnnxStep(node, Descriptor, span, name);
    }

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        var node = operation.Node;
        var span = operation.Span;
        var condition = CompilerCSharpNaming.Identifier(node.Inputs[0].Name);
        var thenGraph = GetBranch(node, "then_branch", span);
        var elseGraph = GetBranch(node, "else_branch", span);
        var thenExpression = CompilerOnnxGraphCSharpPrinter.PrintTensorArray(thenGraph, span);
        var elseExpression = CompilerOnnxGraphCSharpPrinter.PrintTensorArray(elseGraph, span);
        var conditional = $"({condition}.item<bool>() ? {thenExpression} : {elseExpression})";
        return node.Outputs.Count == 1 ? $"{conditional}[0]" : conditional;
    }

    private static OnnxGraph GetBranch(OnnxNode node, string name, CompilerSourceSpan? span)
    {
        if (node.Attributes.FirstOrDefault(attribute => string.Equals(attribute.Name, name, StringComparison.Ordinal))?.GetValue() is OnnxGraph graph)
        {
            return graph;
        }

        throw Unsupported(span, $"ONNX If requires a graph-valued '{name}' attribute.");
    }
}
