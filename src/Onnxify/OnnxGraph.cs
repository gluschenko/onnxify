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
            _nodes.Add(new OnnxNode(node, this));
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
        string name
    )
    {
        if (_tensors.Contains(name))
        {
            throw new InvalidOperationException($"Tensor '{name}' is already added into graph");
        }

        var tensor = new OnnxTensor<T>(
            name: name,
            dataType: TensorProto.Types.DataType.Float,
            dataLocation: TensorProto.Types.DataLocation.Default,
            shape: [1, 3, 256, 256],
            value: [],
            null
        );

        _tensors.Add(tensor);
        return tensor;
    }

    public OnnxNode AddNode(
        string name,
        string opType,
        IEnumerable<string> inputs,
        IEnumerable<string> outputs,
        IEnumerable<string> attributes
    )
    {
        if (_nodes.Contains(name))
        {
            throw new InvalidOperationException($"Node '{name}' is already added into graph");
        }

        var proto = new NodeProto
        {
            Name = name,
            OpType = opType,
            Input = { inputs },
            Output = { outputs },
        };

        var node = new OnnxNode(proto, this);
        _nodes.Add(node);
        return node;
    }

    internal GraphProto ToProto()
    {
        var newGraph = _graph.Clone();
        newGraph.Name = Name;

        newGraph.Initializer.Clear();
        foreach (var x in _tensors)
        {
            newGraph.Initializer.Add(x.ToProto());
        }

        newGraph.ValueInfo.Clear();
        foreach (var x in _constraints)
        {
            newGraph.ValueInfo.Add(x.ToProto());
        }

        newGraph.Input.Clear();
        foreach (var x in _inputs)
        {
            newGraph.Input.Add(x.ToProto());
        }

        newGraph.Output.Clear();
        foreach (var x in _outputs)
        {
            newGraph.Output.Add(x.ToProto());
        }

        newGraph.Node.Clear();
        foreach (var x in _nodes)
        {
            newGraph.Node.Add(x.ToProto());
        }

        return newGraph;
    }
}
