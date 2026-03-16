using Google.Protobuf;
using Onnx;
using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;

namespace Onnxify;

public class OnnxModelCreateOptions
{
    public int Opset { get; set; } = 13;
    public long IrVersion { get; set; } = 8;
    public string ProducerName { get; set; } = "onnxify";
}

public class OnnxModel
{
    public string ProducerName
    {
        get => _model.ProducerName;
        set => _model.ProducerName = value;
    }

    public string ProducerVersion
    {
        get => _model.ProducerVersion;
        set => _model.ProducerVersion = value;
    }

    public long ModelVersion
    {
        get => _model.ModelVersion;
        set => _model.ModelVersion = value;
    }

    public long IrVersion
    {
        get => _model.IrVersion;
        set => _model.IrVersion = value;
    }

    public string Document
    {
        get => _model.DocString;
        set => _model.DocString = value;
    }

    public string Domain
    {
        get => _model.Domain;
        set => _model.Domain = value;
    }

    public IList<StringStringEntryProto> MetadataProps { get; }
    public IList<TrainingInfoProto> TrainingInfo { get; }
    public IList<OperatorSetIdProto> OpsetImport { get; }

    public OnnxGraph Graph => _graph;

    private readonly ModelProto _model;
    private readonly OnnxGraph _graph;

    internal OnnxModel(ModelProto model)
    {
        _model = model;
        _graph = new OnnxGraph(model.Graph);

        MetadataProps = new List<StringStringEntryProto>(model.MetadataProps);
        TrainingInfo = new List<TrainingInfoProto>(model.TrainingInfo);
        OpsetImport = new List<OperatorSetIdProto>(model.OpsetImport);
    }

    public static OnnxModel Create(OnnxModelCreateOptions? options = null)
    {
        options ??= new OnnxModelCreateOptions();

        var model = new ModelProto
        {
            IrVersion = options.IrVersion,
            ProducerName = options.ProducerName,
            Graph = new GraphProto(),
        };

        model.OpsetImport.Add(new OperatorSetIdProto
        {
            Domain = "",
            Version = options.Opset,
        });

        return new OnnxModel(model);
    }

    public static OnnxModel FromFile(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("File not found", path);
        }

        var data = File.ReadAllBytes(path);
        var model = ModelProto.Parser.ParseFrom(data);
        return new OnnxModel(model);
    }

    public void Save(string path, bool overwrite = false)
    {
        if (File.Exists(path) && !overwrite)
        {
            throw new IOException($"File already exists at '{path}'");
        }

        using var fileStream = File.Create(path);

        var newModel = ToProto();
        newModel.WriteTo(fileStream);
    }

    internal ModelProto ToProto()
    {
        var newModel = _model.Clone();
        newModel.Graph = _graph.ToProto();

        newModel.MetadataProps.Clear();
        newModel.MetadataProps.AddRange(MetadataProps);

        newModel.TrainingInfo.Clear();
        newModel.TrainingInfo.AddRange(TrainingInfo);

        newModel.OpsetImport.Clear();
        newModel.OpsetImport.AddRange(OpsetImport);

        return newModel;
    }
}

public class OnnxGraph
{
    public string Name { get; init; }

    private readonly GraphProto _graph;

    public IReadOnlyList<OnnxValue> Inputs => _inputs;
    public IReadOnlyList<OnnxValue> Outputs => _outputs;
    public IReadOnlyList<OnnxNode> Nodes => _nodes;

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
            _tensors.Add(new OnnxTensor(tensor, this));
        }

        foreach (var value in graph.ValueInfo)
        {
            _constraints.Add(new OnnxValue(value, this));
        }

        foreach (var input in graph.Input)
        {
            _inputs.Add(new OnnxValue(input, this));
        }

        foreach (var output in graph.Output)
        {
            _outputs.Add(new OnnxValue(output, this));
        }

        foreach (var node in graph.Node)
        {
            foreach (var x in node.Input.Concat(node.Output))
            {
                if (GetValue(x) is null)
                {
                    _edges[x] = new OnnxEdge(this, x);
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

    public GraphProto ToProto()
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

public interface IOnnxGraphNode
{
    public string Name { get; }
    public OnnxGraph GetGraph();
}

public interface IOnnxGraphEdge
{
    public string Name { get; }
    public OnnxGraph GetGraph();
}

public class OnnxNode : IOnnxGraphNode
{
    public string Name { get; init; }
    public string OpType { get; set; }
    public string Domain { get; set; }
    public string DocString { get; set; }

    public List<IOnnxGraphEdge> Inputs { get; private set; } = [];
    public List<IOnnxGraphEdge> Outputs { get; private set; } = [];
    public List<AttributeProto> Attributes { get; private set; } = [];

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
            Inputs.Add(value);
        }

        foreach (var x in node.Output)
        {
            var value = graph.GetValue(x) ?? throw new InvalidOperationException($"No graph value found for '{x}'.");
            Outputs.Add(value);
        }

        foreach (var attribute in node.Attribute)
        {
            Attributes.Add(attribute.Clone());
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
            newNode.Attribute.Add(attribute.Clone());
        }

        return newNode;
    }
}

public class OnnxTensor : IOnnxGraphEdge
{
    public string Name { get; init; }
    public TensorProto.Types.DataLocation DataLocation { set; get; }

    private readonly TensorProto _tensor;
    private readonly OnnxGraph _graph;

    internal OnnxTensor(TensorProto tensor, OnnxGraph graph)
    {
        _tensor = tensor;
        _graph = graph;

        Name = tensor.Name;
        DataLocation = tensor.DataLocation;
    }

    public OnnxGraph GetGraph()
    {
        return _graph;
    }

    internal TensorProto ToProto()
    {
        var newTensor = _tensor.Clone();
        newTensor.Name = Name;
        newTensor.DataLocation = DataLocation;

        return newTensor;
    }
}

public class OnnxValue : IOnnxGraphEdge
{
    public string Name { get; init; }

    private readonly ValueInfoProto _valueInfo;
    private readonly OnnxGraph _graph;

    internal OnnxValue(ValueInfoProto valueInfo, OnnxGraph graph)
    {
        _valueInfo = valueInfo;
        _graph = graph;

        Name = valueInfo.Name;
    }

    public OnnxGraph GetGraph()
    {
        return _graph;
    }

    internal ValueInfoProto ToProto()
    {
        var newValue = _valueInfo.Clone();
        newValue.Name = Name;

        return newValue;
    }
}

public class OnnxEdge : IOnnxGraphEdge
{
    public string Name { get; init; }
    private readonly OnnxGraph _graph;

    internal OnnxEdge(OnnxGraph graph, string name)
    {
        _graph = graph;

        Name = name;
    }

    public OnnxGraph GetGraph()
    {
        return _graph;
    }
}

public class LazyDictionary<TKey, TValue> : KeyedCollection<TKey, TValue> where TKey : notnull
{
    private readonly Func<TValue, TKey> _keySelector;
    private readonly IEqualityComparer<TKey> _equalityComparer;

    public LazyDictionary(Func<TValue, TKey> keySelector, IEqualityComparer<TKey> comparer) : base(comparer)
    {
        _keySelector = keySelector;
        _equalityComparer = comparer;
    }

    protected override TKey GetKeyForItem(TValue item)
    {
        return _keySelector(item);
    }

    public new TValue this[TKey key]
    {
        get => base[key];
        set
        {
            var newKey = _keySelector(value);

            if (!_equalityComparer.Equals(key, newKey))
            {
                throw new ArgumentException("Key of value does not match indexer key.");
            }

            if (TryGetValue(key, out var existing))
            {
                var index = Items.IndexOf(existing);
                SetItem(index, value);
            }
            else
            {
                Add(value);
            }
        }
    }
}
