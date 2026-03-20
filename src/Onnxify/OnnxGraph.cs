using System.Xml.Linq;
using Google.Protobuf.WellKnownTypes;
using Onnx;

namespace Onnxify;

public class OnnxGraph
{
    public string Name { get; init; }

    private readonly GraphProto _graph;

    public IReadOnlyList<OnnxValue> Inputs => _inputs;
    public IReadOnlyList<OnnxValue> Outputs => _outputs;
    public IReadOnlyList<OnnxTensor> Initializers => _initializers;
    public IReadOnlyList<OnnxValue> Placeholders => _placeholders;
    public IReadOnlyList<OnnxNode> Nodes => _nodes;

    private readonly LazyDictionary<string, OnnxValue> _inputs = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxValue> _outputs = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxTensor> _initializers = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxValue> _placeholders = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxNode> _nodes = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxEdge> _edges = new(x => x.Name, StringComparer.Ordinal);

    internal OnnxGraph(GraphProto graph)
    {
        _graph = graph;

        foreach (var tensor in graph.Initializer)
        {
            _initializers.Add(OnnxHelper.FromProto(tensor));
        }

        foreach (var value in graph.ValueInfo)
        {
            _placeholders.Add(OnnxHelper.FromProto(value));
        }

        foreach (var input in graph.Input)
        {
            _inputs.Add(OnnxHelper.FromProto(input));
        }

        foreach (var output in graph.Output)
        {
            _outputs.Add(OnnxHelper.FromProto(output));
        }

        foreach (var node in graph.Node)
        {
            foreach (var x in node.Input.Concat(node.Output))
            {
                if (GetValue(x) is null)
                {
                    _edges.Add(new OnnxEdge(x));
                }
            }
        }

        foreach (var node in graph.Node)
        {
            _nodes.Add(OnnxNode.FromProto(node, this));
        }

        Name = graph.Name;
    }

    public OnnxNode? GetNode(string name)
    {
        if (_nodes.TryGetValue(name, out var result))
        {
            return result;
        }

        return null;
    }

    public IOnnxGraphEdge? GetValue(string name)
    {
        if (_inputs.TryGetValue(name, out var input))
        {
            return input;
        }

        if (_outputs.TryGetValue(name, out var output))
        {
            return output;
        }

        if (_placeholders.TryGetValue(name, out var value))
        {
            return value;
        }

        if (_initializers.TryGetValue(name, out var tensor))
        {
            return tensor;
        }

        if (_edges.TryGetValue(name, out var edge))
        {
            return edge;
        }

        return null;
    }

    public OnnxTensor AddTensor<T>(
        string name,
        long[] shape,
        T[] value
    )
    {
        if (_initializers.Contains(name))
        {
            throw new InvalidOperationException($"Tensor '{name}' is already added into graph");
        }

        var tensor = new OnnxTensor<T>(
            name: name,
            dataLocation: OnnxTensor.TensorDataLocation.Default,
            shape: shape,
            value: value,
            null
        );

        _initializers.Add(tensor);
        return tensor;
    }

    public OnnxValue AddInput<T>(string name, T type) where T : OnnxValueType
    {
        if (_inputs.Contains(name))
        {
            throw new InvalidOperationException($"Value '{name}' is already added into graph");
        }

        var placeholder = new OnnxValue<T>(name, type, null);
        _inputs.Add(placeholder);
        return placeholder;
    }

    public OnnxValue AddOutput<T>(string name, T type) where T : OnnxValueType
    {
        if (_outputs.Contains(name))
        {
            throw new InvalidOperationException($"Value '{name}' is already added into graph");
        }

        var placeholder = new OnnxValue<T>(name, type, null);
        _outputs.Add(placeholder);
        return placeholder;
    }

    public OnnxValue AddValue<T>(string name, T type) where T : OnnxValueType
    {
        if (_placeholders.Contains(name))
        {
            throw new InvalidOperationException($"Value '{name}' is already added into graph");
        }

        var placeholder = new OnnxValue<T>(name, type, null);
        _placeholders.Add(placeholder);
        return placeholder;
    }

    public OnnxEdge AddEdge(string name)
    {
        if (_edges.Contains(name))
        {
            throw new InvalidOperationException($"Edge '{name}' is already added into graph");
        }

        var edge = new OnnxEdge(name);
        _edges.Add(edge);
        return edge;
    }

    public OnnxNode AddNode(
        string name,
        string opType,
        string domain,
        string docString,
        IEnumerable<IOnnxGraphEdge> inputs,
        IEnumerable<IOnnxGraphEdge> outputs,
        IEnumerable<OnnxAttribute> attributes
    )
    {
        var node = new OnnxNode(
            name: name,
            opType: opType,
            domain: domain,
            docString: docString,
            inputs: inputs,
            outputs: outputs,
            attributes: attributes,
            proto: null
        );

        return AddNode(node);
    }

    public OnnxNode AddNode(OnnxNode node)
    {
        if (_nodes.Contains(node.Name))
        {
            throw new InvalidOperationException($"Node '{node.Name}' is already added into graph");
        }

        _nodes.Add(node);
        return node;
    }

    internal GraphProto ToProto()
    {
        var newGraph = _graph.Clone();
        newGraph.Name = Name;

        newGraph.Initializer.Set(_initializers.Select(x => x.ToProto()));
        newGraph.ValueInfo.Set(_placeholders.Select(x => x.ToProto()));
        newGraph.Input.Set(_inputs.Select(x => x.ToProto()));
        newGraph.Output.Set(_outputs.Select(x => x.ToProto()));
        newGraph.Node.Set(_nodes.Select(x => x.ToProto()));

        return newGraph;
    }
}
