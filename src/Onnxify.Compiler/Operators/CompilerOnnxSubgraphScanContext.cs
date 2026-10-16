using System.Globalization;
using Onnxify;

namespace Onnxify.Compiler.Operators;

internal sealed class CompilerOnnxSubgraphScanContext(
    OnnxGraph graph,
    int startIndex,
    CompilerSourceSpan? span
)
{
    public OnnxGraph Graph { get; } = graph;
    public int StartIndex { get; } = startIndex;
    public CompilerSourceSpan? Span { get; } = span;

    public bool TryGetNode<TNode>(int relativeIndex, out TNode? node)
        where TNode : OnnxNode
    {
        var index = StartIndex + relativeIndex;
        if (index >= 0 && index < Graph.Nodes.Count && Graph.Nodes[index] is TNode typedNode)
        {
            node = typedNode;
            return true;
        }

        node = null;
        return false;
    }

    public bool CanCollapse(int nodeCount)
    {
        if (nodeCount <= 0 || StartIndex + nodeCount > Graph.Nodes.Count)
        {
            return false;
        }

        var firstInternalIndex = StartIndex;
        var lastInternalIndex = StartIndex + nodeCount - 1;
        var finalNode = Graph.Nodes[lastInternalIndex];
        var finalOutputs = finalNode.Outputs.Select(static output => output.Name).ToHashSet(StringComparer.Ordinal);

        for (var relativeIndex = 0; relativeIndex < nodeCount - 1; relativeIndex++)
        {
            foreach (var output in Graph.Nodes[StartIndex + relativeIndex].Outputs)
            {
                if (finalOutputs.Contains(output.Name)
                    || Graph.Outputs.Any(value => string.Equals(value.Name, output.Name, StringComparison.Ordinal)))
                {
                    return false;
                }

                for (var consumerIndex = 0; consumerIndex < Graph.Nodes.Count; consumerIndex++)
                {
                    if (consumerIndex >= firstInternalIndex && consumerIndex <= lastInternalIndex)
                    {
                        continue;
                    }

                    if (Graph.Nodes[consumerIndex].Inputs.Any(input =>
                            string.Equals(input.Name, output.Name, StringComparison.Ordinal)))
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    public bool IsScalarConstant(string name, double expectedValue)
    {
        if (Graph.GetValue(name) is not OnnxTensor tensor
            || tensor.Shape.Length != 0
            || tensor.Values.Count != 1
            || tensor.Values[0] is not IConvertible value)
        {
            return false;
        }

        try
        {
            var actualValue = value.ToDouble(CultureInfo.InvariantCulture);
            return actualValue.Equals(expectedValue);
        }
        catch (Exception)
        {
            return false;
        }
    }

    public bool IsScalarInitializer(string name)
    {
        return Graph.GetValue(name) is OnnxTensor { Shape.Length: 0, Values.Count: 1 };
    }

    public CompilerOnnxStep CreateCompositeStep(
        CompilerOperator compilerOperator,
        IEnumerable<string> inputNames,
        string outputName,
        int consumedNodeCount
    )
    {
        var finalNode = Graph.Nodes[StartIndex + consumedNodeCount - 1];
        var name = string.IsNullOrWhiteSpace(finalNode.Name)
            ? $"__compiler_composite_{StartIndex}"
            : finalNode.Name;
        return compilerOperator.CreateNode(
            name,
            inputNames.Select(static input => new CompilerValueReference(input)),
            [new CompilerValueReference(outputName)],
            span: Span);
    }
}
