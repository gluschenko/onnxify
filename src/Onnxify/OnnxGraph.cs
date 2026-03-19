using Onnx;

namespace Onnxify;

public class OnnxGraph
{
    public string Name { get; init; }

    private readonly GraphProto _graph;

    public IReadOnlyList<OnnxValue> Inputs => _inputs;
    public IReadOnlyList<OnnxValue> Outputs => _outputs;
    public IReadOnlyList<OnnxNode> Nodes => _nodes;
    public IReadOnlyList<OnnxTensor> Tensors => _tensors;

    private readonly LazyDictionary<string, OnnxEdge> _edges = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, OnnxTensor> _tensors = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, OnnxValue> _constraints = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, OnnxValue> _inputs = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, OnnxValue> _outputs = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, OnnxNode> _nodes = new(x => x.Name, EqualityComparer<string>.Default);

    internal OnnxGraph(GraphProto graph)
    {
        _graph = graph;

        foreach (var tensor in graph.Initializer)
        {
            _tensors.Add(OnnxHelper.FromProto(tensor));
        }

        foreach (var value in graph.ValueInfo)
        {
            _constraints.Add(new OnnxValue(value));
        }

        foreach (var input in graph.Input)
        {
            _inputs.Add(new OnnxValue(input));
        }

        foreach (var output in graph.Output)
        {
            _outputs.Add(new OnnxValue(output));
        }

        foreach (var node in graph.Node)
        {
            foreach (var x in node.Input.Concat(node.Output))
            {
                if (GetValue(x) is null)
                {
                    _edges[x] = new OnnxEdge(x);
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

        if (_constraints.TryGetValue(name, out var value))
        {
            return value;
        }

        if (_tensors.TryGetValue(name, out var tensor))
        {
            return tensor;
        }

        if (_edges.TryGetValue(name, out var edge))
        {
            return edge;
        }

        return null;
    }

    public OnnxTensor? GetTensor(string name)
    {
        if (_tensors.TryGetValue(name, out var result))
        {
            return result;
        }

        return null;
    }

    public OnnxTensor AddValue<T>(
        string name,
        long[] shape,
        T[] value
    )
    {
        if (_tensors.Contains(name))
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

        _tensors.Add(tensor);
        return tensor;
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
        if (_nodes.Contains(name))
        {
            throw new InvalidOperationException($"Node '{name}' is already added into graph");
        }

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

        _nodes.Add(node);
        return node;
    }

    internal GraphProto ToProto()
    {
        var newGraph = _graph.Clone();
        newGraph.Name = Name;

        newGraph.Initializer.Set(_tensors.Select(x => x.ToProto()));
        newGraph.ValueInfo.Set(_constraints.Select(x => x.ToProto()));
        newGraph.Input.Set(_inputs.Select(x => x.ToProto()));
        newGraph.Output.Set(_outputs.Select(x => x.ToProto()));
        newGraph.Node.Set(_nodes.Select(x => x.ToProto()));

        return newGraph;
    }
}
