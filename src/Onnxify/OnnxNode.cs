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

    internal OnnxNode(
        string name,
        string opType,
        string domain,
        string docString,
        IEnumerable<IOnnxGraphEdge> inputs,
        IEnumerable<IOnnxGraphEdge> outputs,
        IEnumerable<OnnxAttribute> attributes,
        NodeProto? proto
    )
    {
        _node = proto ?? new NodeProto
        {
            Name = name,
        };

        Name = name;
        OpType = opType;
        Domain = domain;
        DocString = docString;

        foreach (var x in inputs)
        {
            _inputs.Add(x);
        }

        foreach (var x in outputs)
        {
            _outputs.Add(x);
        }

        foreach (var x in attributes)
        {
            _attributes.Add(x);
        }
    }

    internal NodeProto ToProto()
    {
        var newNode = _node.Clone();
        newNode.Name = Name;
        newNode.OpType = OpType;
        newNode.Domain = Domain;
        newNode.DocString = DocString;

        newNode.Input.Set(Inputs.Select(x => x.Name));
        newNode.Output.Set(Outputs.Select(x => x.Name));
        newNode.Attribute.Set(Attributes.Select(x => x.ToProto()));

        return newNode;
    }

    internal static OnnxNode FromProto(NodeProto node, OnnxGraph graph)
    {
        var name = node.Name;
        var opType = node.OpType;
        var domain = node.Domain;
        var docString = node.DocString;
        var inputs = new List<IOnnxGraphEdge>();
        var outputs = new List<IOnnxGraphEdge>();
        var attributes = new List<OnnxAttribute>();

        foreach (var x in node.Input)
        {
            var value = graph.GetValue(x) ?? throw new InvalidOperationException($"No graph value found for '{x}'.");
            inputs.Add(value);
        }

        foreach (var x in node.Output)
        {
            var value = graph.GetValue(x) ?? throw new InvalidOperationException($"No graph value found for '{x}'.");
            outputs.Add(value);
        }

        foreach (var attribute in node.Attribute)
        {
            var value = OnnxHelper.FromProto(attribute);
            attributes.Add(value);
        }

        return new OnnxNode(
            name: name,
            opType: opType,
            domain: domain,
            docString: docString,
            inputs: inputs,
            outputs: outputs,
            attributes: attributes,
            proto: node
        );
    }
}
