using Google.Protobuf;
using Onnx;
using System.Collections.ObjectModel;
using System.Xml.Linq;
using static Onnx.TensorProto.Types;
using static Tensorboard.ApiDef.Types;

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
            _tensors.Add(new OnnxTensor(tensor));
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
}

public abstract class BaseOnnxAttribute
{
    public abstract string Name { get; }
    internal abstract AttributeProto ToProto();
}

public class OnnxNode : IOnnxGraphNode
{
    public string Name { get; init; }
    public string OpType { get; set; }
    public string Domain { get; set; }
    public string DocString { get; set; }

    public IReadOnlyList<IOnnxGraphEdge> Inputs => _inputs;
    public IReadOnlyList<IOnnxGraphEdge> Outputs => _outputs;
    public IReadOnlyList<BaseOnnxAttribute> Attributes => _attributes;

    private readonly LazyDictionary<string, IOnnxGraphEdge> _inputs = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, IOnnxGraphEdge> _outputs = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, BaseOnnxAttribute> _attributes = new(x => x.Name, EqualityComparer<string>.Default);

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
            var value = OnnxAttributeHelper.FromProto(attribute);
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


public class OnnxAttribute<T> : BaseOnnxAttribute
{
    public override string Name => _attribute.Name;
    public AttributeProto.Types.AttributeType Type { get; init; }
    public T Value { get; set; }

    private readonly AttributeProto _attribute;

    public OnnxAttribute(AttributeProto attribute)
    {
        _attribute = attribute;

        Type = _attribute.Type;
        Value = OnnxAttributeHelper.GetValue<T>(attribute);
    }

    internal override AttributeProto ToProto()
    {
        var newAttribute = _attribute.Clone();
        newAttribute.Name = Name;
        newAttribute.Type = Type;
        newAttribute.SetValue(Value);

        return newAttribute;
    }
}

public static class OnnxAttributeHelper
{
    internal static BaseOnnxAttribute FromProto(AttributeProto attribute)
    {
        if (attribute.Type == AttributeProto.Types.AttributeType.Undefined)
        {
            return new OnnxAttribute<object>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.Float)
        {
            return new OnnxAttribute<float>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.Int)
        {
            return new OnnxAttribute<long>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.String)
        {
            return new OnnxAttribute<string>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.Tensor)
        {
            return new OnnxAttribute<OnnxTensor>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.Graph)
        {
            return new OnnxAttribute<OnnxGraph>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.SparseTensor)
        {
            return new OnnxAttribute<OnnxSparseTensor>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.Floats)
        {
            return new OnnxAttribute<float[]>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.Ints)
        {
            return new OnnxAttribute<long[]>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.Strings)
        {
            return new OnnxAttribute<string[]>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.Tensors)
        {
            return new OnnxAttribute<OnnxTensor[]>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.Graphs)
        {
            return new OnnxAttribute<OnnxGraph[]>(attribute);
        }
        else if (attribute.Type == AttributeProto.Types.AttributeType.SparseTensors)
        {
            return new OnnxAttribute<OnnxSparseTensor[]>(attribute);
        }
        else
        {
            throw new NotImplementedException($"Not implemented for '{attribute.Type}'");
        }
    }

    internal static T GetValue<T>(this AttributeProto attribute)
    {
        return (T)GetValue(attribute);
    }

    internal static object GetValue(this AttributeProto attribute)
    {
        return attribute.Type switch
        {
            AttributeProto.Types.AttributeType.Float => attribute.F,
            AttributeProto.Types.AttributeType.Int => attribute.I,
            AttributeProto.Types.AttributeType.String => attribute.S.ToStringUtf8(),
            AttributeProto.Types.AttributeType.Tensor => new OnnxTensor(attribute.T),
            AttributeProto.Types.AttributeType.Graph => new OnnxGraph(attribute.G),
            AttributeProto.Types.AttributeType.SparseTensor => new OnnxSparseTensor(attribute.SparseTensor),

            AttributeProto.Types.AttributeType.Floats => attribute.Floats.ToArray(),
            AttributeProto.Types.AttributeType.Ints => attribute.Ints.ToArray(),
            AttributeProto.Types.AttributeType.Strings => attribute.Strings.Select(x => x.ToStringUtf8()).ToArray(),
            AttributeProto.Types.AttributeType.Tensors => attribute.Tensors.Select(x => new OnnxTensor(x)).ToArray(),
            AttributeProto.Types.AttributeType.Graphs => attribute.Graphs.Select(x => new OnnxGraph(x)).ToArray(),
            AttributeProto.Types.AttributeType.SparseTensors => attribute.SparseTensors.Select(x => new OnnxSparseTensor(x)).ToArray(),

            _ => throw new NotImplementedException($"Unsupported attribute type {attribute.Type}")
        };
    }

    internal static void SetValue<T>(this AttributeProto attribute, T value)
    {
        switch (value)
        {
            case float f:
                attribute.F = f;
                attribute.Type = AttributeProto.Types.AttributeType.Float;
                break;

            case long i:
                attribute.I = i;
                attribute.Type = AttributeProto.Types.AttributeType.Int;
                break;

            case string s:
                attribute.S = ByteString.CopyFromUtf8(s);
                attribute.Type = AttributeProto.Types.AttributeType.String;
                break;

            case OnnxTensor t:
                attribute.T = t.ToProto();
                attribute.Type = AttributeProto.Types.AttributeType.String;
                break;

            case OnnxGraph g:
                attribute.G = g.ToProto();
                attribute.Type = AttributeProto.Types.AttributeType.String;
                break;

            case OnnxSparseTensor sparseTensor:
                attribute.SparseTensor = sparseTensor.ToProto();
                attribute.Type = AttributeProto.Types.AttributeType.String;
                break;

            case float[] floatArray:
                attribute.Floats.Clear();
                attribute.Floats.AddRange(floatArray);
                attribute.Type = AttributeProto.Types.AttributeType.Floats;
                break;

            case long[] intArray:
                attribute.Ints.Clear();
                attribute.Ints.AddRange(intArray);
                attribute.Type = AttributeProto.Types.AttributeType.Ints;
                break;

            case string[] stringArray:
                attribute.Strings.Clear();
                attribute.Strings.AddRange(stringArray.Select(ByteString.CopyFromUtf8));
                attribute.Type = AttributeProto.Types.AttributeType.Strings;
                break;

            case OnnxTensor[] tensorArray:
                foreach (var x in tensorArray)
                {
                    attribute.Tensors.Add(x.ToProto());
                }

                attribute.Type = AttributeProto.Types.AttributeType.String;
                break;

            case OnnxGraph[] graphArray:
                foreach (var x in graphArray)
                {
                    attribute.Graphs.Add(x.ToProto());
                }

                attribute.Type = AttributeProto.Types.AttributeType.String;
                break;

            case OnnxSparseTensor[] sparseTensorArray:
                foreach (var x in sparseTensorArray)
                {
                    attribute.SparseTensors.Add(x.ToProto());
                }

                attribute.Type = AttributeProto.Types.AttributeType.String;
                break;

            default:
                throw new NotSupportedException($"Unsupported attribute type {typeof(T).Name}");
        }
    }
}

public class OnnxTensor : IOnnxGraphEdge
{
    public string Name { get; init; }
    public TensorProto.Types.DataLocation DataLocation { set; get; }

    private readonly TensorProto _tensor;

    internal OnnxTensor(TensorProto tensor)
    {
        _tensor = tensor;

        Name = tensor.Name;
        DataLocation = tensor.DataLocation;
    }

    internal TensorProto ToProto()
    {
        var newTensor = _tensor.Clone();
        newTensor.Name = Name;
        newTensor.DataLocation = DataLocation;

        return newTensor;
    }
}

public class OnnxSparseTensor : IOnnxGraphEdge
{
    public string Name => _value.Name;
    public OnnxTensor Value => _value;

    private readonly SparseTensorProto _tensor;
    private readonly OnnxTensor _value;

    internal OnnxSparseTensor(SparseTensorProto tensor)
    {
        _tensor = tensor;
        _value = new OnnxTensor(tensor.Values);
    }

    internal SparseTensorProto ToProto()
    {
        var newTensor = _tensor.Clone();
        newTensor.Values = _value.ToProto();

        return newTensor;
    }
}

public class OnnxValue : IOnnxGraphEdge
{
    public string Name { get; init; }

    private readonly ValueInfoProto _valueInfo;

    internal OnnxValue(ValueInfoProto valueInfo)
    {
        _valueInfo = valueInfo;

        Name = valueInfo.Name;
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

    internal OnnxEdge(string name)
    {
        Name = name;
    }
}

public class LazyDictionary<TKey, TValue> : KeyedCollection<TKey, TValue> where TKey : notnull
{
    private readonly Func<TValue, TKey> _keySelector;

    public LazyDictionary(Func<TValue, TKey> keySelector, IEqualityComparer<TKey>? comparer = null) : base(comparer)
    {
        _keySelector = keySelector;
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

            if (!Comparer.Equals(key, newKey))
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
