using Onnx;

namespace Onnxify;

public class OnnxNode : IOnnxGraphNode
{
    public string Name { get; init; }
    public string OpType { get; set; }
    public string Domain { get; set; }
    public string DocString { get; set; }

    public IReadOnlyList<IOnnxGraphEdge> Inputs => _inputs;
    public IReadOnlyList<IOnnxGraphEdge> Outputs => _outputs;
    public IReadOnlyList<OnnxAttribute> Attributes => _attributes;

    private readonly LazyDictionary<string, IOnnxGraphEdge> _inputs = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, IOnnxGraphEdge> _outputs = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, OnnxAttribute> _attributes = new(x => x.Name, EqualityComparer<string>.Default);

    private readonly NodeProto _node;
    private readonly OnnxGraph _graph;

    internal OnnxNode(NodeProto node, OnnxGraph graph)
    {
        _node = node;
        _graph = graph;

        Name = node.Name;
        OpType = node.OpType;
        Domain = node.Domain;
        DocString = node.DocString;

        foreach (var x in node.Input)
        {
            var value = graph.GetValue(x) ?? throw new InvalidOperationException($"No graph value found for '{x}'.");
            _inputs.Add(value);
        }

        foreach (var x in node.Output)
        {
            var value = graph.GetValue(x) ?? throw new InvalidOperationException($"No graph value found for '{x}'.");
            _outputs.Add(value);
        }

        foreach (var attribute in node.Attribute)
        {
            var value = OnnxHelper.FromProto(attribute);
            _attributes.Add(value);
        }
    }

    public OnnxGraph GetGraph()
    {
        return _graph;
    }

    internal NodeProto ToProto()
    {
        var newNode = _node.Clone();
        newNode.Name = Name;
        newNode.OpType = OpType;
        newNode.Domain = Domain;
        newNode.DocString = DocString;

        newNode.Input.Clear();
        foreach (var x in Inputs)
        {
            newNode.Input.Add(x.Name);
        }

        newNode.Output.Clear();
        foreach (var x in Outputs)
        {
            newNode.Output.Add(x.Name);
        }

        newNode.Attribute.Clear();
        foreach (var attribute in Attributes)
        {
            newNode.Attribute.Add(attribute.ToProto());
        }

        return newNode;
    }
}
