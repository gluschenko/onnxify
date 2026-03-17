using Google.Protobuf;
using Onnx;
using System.Collections.ObjectModel;
using System.Numerics;
using System.Runtime.InteropServices;

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

public abstract class OnnxAttribute
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


public class OnnxAttribute<T> : OnnxAttribute
{
    public override string Name => _attribute.Name;
    public AttributeProto.Types.AttributeType Type { get; init; }
    public T Value { get; set; }

    private readonly AttributeProto _attribute;

    public OnnxAttribute(AttributeProto attribute)
    {
        _attribute = attribute;

        Type = _attribute.Type;
        Value = OnnxHelper.GetValue<T>(attribute);
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

public static class OnnxHelper
{
    internal static OnnxTensor FromProto(TensorProto tensor)
    {
        var type = (TensorProto.Types.DataType)tensor.DataType;

        return type switch
        {
            TensorProto.Types.DataType.Undefined => new OnnxTensor<object>(tensor),
            TensorProto.Types.DataType.Float => new OnnxTensor<float>(tensor),
            TensorProto.Types.DataType.Uint8 => new OnnxTensor<byte>(tensor),
            TensorProto.Types.DataType.Int8 => new OnnxTensor<sbyte>(tensor),
            TensorProto.Types.DataType.Uint16 => new OnnxTensor<ushort>(tensor),
            TensorProto.Types.DataType.Int16 => new OnnxTensor<short>(tensor),
            TensorProto.Types.DataType.Int32 => new OnnxTensor<int>(tensor),
            TensorProto.Types.DataType.Int64 => new OnnxTensor<long>(tensor),
            TensorProto.Types.DataType.String => new OnnxTensor<string>(tensor),
            TensorProto.Types.DataType.Bool => new OnnxTensor<bool>(tensor),
            TensorProto.Types.DataType.Float16 => new OnnxTensor<Half>(tensor),
            TensorProto.Types.DataType.Double => new OnnxTensor<double>(tensor),
            TensorProto.Types.DataType.Uint32 => new OnnxTensor<uint>(tensor),
            TensorProto.Types.DataType.Uint64 => new OnnxTensor<ulong>(tensor),
            TensorProto.Types.DataType.Complex64 => new OnnxTensor<Complex64>(tensor),
            TensorProto.Types.DataType.Complex128 => new OnnxTensor<Complex>(tensor),
            TensorProto.Types.DataType.Bfloat16 => new OnnxTensor<BFloat16>(tensor),
            _ => throw new NotImplementedException($"Not implemented for '{type}'"),
        };
    }

    internal static OnnxSparseTensorBase FromProto(SparseTensorProto tensor)
    {
        var type = (TensorProto.Types.DataType)tensor.Values.DataType;

        return type switch
        {
            TensorProto.Types.DataType.Undefined => new OnnxSparseTensor<object>(tensor),
            TensorProto.Types.DataType.Float => new OnnxSparseTensor<float>(tensor),
            TensorProto.Types.DataType.Uint8 => new OnnxSparseTensor<byte>(tensor),
            TensorProto.Types.DataType.Int8 => new OnnxSparseTensor<sbyte>(tensor),
            TensorProto.Types.DataType.Uint16 => new OnnxSparseTensor<ushort>(tensor),
            TensorProto.Types.DataType.Int16 => new OnnxSparseTensor<short>(tensor),
            TensorProto.Types.DataType.Int32 => new OnnxSparseTensor<int>(tensor),
            TensorProto.Types.DataType.Int64 => new OnnxSparseTensor<long>(tensor),
            TensorProto.Types.DataType.String => new OnnxSparseTensor<string>(tensor),
            TensorProto.Types.DataType.Bool => new OnnxSparseTensor<bool>(tensor),
            TensorProto.Types.DataType.Float16 => new OnnxSparseTensor<Half>(tensor),
            TensorProto.Types.DataType.Double => new OnnxSparseTensor<double>(tensor),
            TensorProto.Types.DataType.Uint32 => new OnnxSparseTensor<uint>(tensor),
            TensorProto.Types.DataType.Uint64 => new OnnxSparseTensor<ulong>(tensor),
            TensorProto.Types.DataType.Complex64 => new OnnxSparseTensor<Complex64>(tensor),
            TensorProto.Types.DataType.Complex128 => new OnnxSparseTensor<Complex>(tensor),
            TensorProto.Types.DataType.Bfloat16 => new OnnxSparseTensor<BFloat16>(tensor),
            _ => throw new NotImplementedException($"Not implemented for '{type}'"),
        };
    }

    internal static OnnxAttribute FromProto(AttributeProto attribute)
    {
        return attribute.Type switch
        {
            AttributeProto.Types.AttributeType.Float => new OnnxAttribute<float>(attribute),
            AttributeProto.Types.AttributeType.Int => new OnnxAttribute<long>(attribute),
            AttributeProto.Types.AttributeType.String => new OnnxAttribute<string>(attribute),

            AttributeProto.Types.AttributeType.Tensor => new OnnxAttribute<OnnxTensor>(attribute),
            AttributeProto.Types.AttributeType.Graph => new OnnxAttribute<OnnxGraph>(attribute),
            AttributeProto.Types.AttributeType.SparseTensor => new OnnxAttribute<OnnxSparseTensorBase>(attribute),

            AttributeProto.Types.AttributeType.Floats => new OnnxAttribute<float[]>(attribute),
            AttributeProto.Types.AttributeType.Ints => new OnnxAttribute<long[]>(attribute),
            AttributeProto.Types.AttributeType.Strings => new OnnxAttribute<string[]>(attribute),

            AttributeProto.Types.AttributeType.Tensors => new OnnxAttribute<OnnxTensor[]>(attribute),
            AttributeProto.Types.AttributeType.Graphs => new OnnxAttribute<OnnxGraph[]>(attribute),
            AttributeProto.Types.AttributeType.SparseTensors => new OnnxAttribute<OnnxSparseTensorBase[]>(attribute),

            _ => throw new NotImplementedException($"Not implemented for '{attribute.Type}'"),
        };
    }

    internal static object GetValue(this TensorProto tensor)
    {
        var type = (TensorProto.Types.DataType)tensor.DataType;

        if (tensor.RawData.Length > 0)
        {
            var span = tensor.RawData.Span;

            return type switch
            {
                TensorProto.Types.DataType.Float => MemoryMarshal.Cast<byte, float>(span).ToArray(),
                TensorProto.Types.DataType.Double => MemoryMarshal.Cast<byte, double>(span).ToArray(),
                TensorProto.Types.DataType.Int32 => MemoryMarshal.Cast<byte, int>(span).ToArray(),
                TensorProto.Types.DataType.Int64 => MemoryMarshal.Cast<byte, long>(span).ToArray(),
                TensorProto.Types.DataType.Uint32 => MemoryMarshal.Cast<byte, uint>(span).ToArray(),
                TensorProto.Types.DataType.Uint64 => MemoryMarshal.Cast<byte, ulong>(span).ToArray(),
                TensorProto.Types.DataType.Int16 => MemoryMarshal.Cast<byte, short>(span).ToArray(),
                TensorProto.Types.DataType.Uint16 => MemoryMarshal.Cast<byte, ushort>(span).ToArray(),
                TensorProto.Types.DataType.Int8 => MemoryMarshal.Cast<byte, sbyte>(span).ToArray(),
                TensorProto.Types.DataType.Uint8 => span.ToArray(),
                TensorProto.Types.DataType.Bool => span.ToArray().Select(x => x != 0).ToArray(),

                TensorProto.Types.DataType.Float16 => ConvertHalf(span),
                TensorProto.Types.DataType.Bfloat16 => ConvertBFloat16(span),

                TensorProto.Types.DataType.Complex64 => ConvertComplex64(span),
                TensorProto.Types.DataType.Complex128 => ConvertComplex128(span),

                _ => throw new NotImplementedException($"Unsupported raw tensor type {type}")
            };
        }

        return type switch
        {
            TensorProto.Types.DataType.Float => tensor.FloatData.ToArray(),
            TensorProto.Types.DataType.Double => tensor.DoubleData.ToArray(),
            TensorProto.Types.DataType.Uint8 => tensor.Int32Data.Select(x => (byte)x).ToArray(),
            TensorProto.Types.DataType.Int8 => tensor.Int32Data.Select(x => (sbyte)x).ToArray(),
            TensorProto.Types.DataType.Int32 => tensor.Int32Data.ToArray(),
            TensorProto.Types.DataType.Int64 => tensor.Int64Data.ToArray(),
            TensorProto.Types.DataType.String => tensor.StringData.Select(x => x.ToStringUtf8()).ToArray(),
            _ => throw new NotImplementedException($"Unsupported non-raw tensor type {type}")
        };
    }

    internal static IEnumerable<T> GetValue<T>(this TensorProto tensor)
    {
        var value = GetValue(tensor);

        if (value is IEnumerable<T> typed)
        {
            return typed;
        }

        throw new InvalidCastException($"Tensor '{tensor.Name}' is {value.GetType().Name}, not {typeof(T).Name}");
    }

    private static BFloat16[] ConvertBFloat16(ReadOnlySpan<byte> data)
    {
        var ushortSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(data);
        var result = new BFloat16[ushortSpan.Length];

        for (int i = 0; i < ushortSpan.Length; i++)
        {
            uint value = (uint)ushortSpan[i] << 16;
            result[i] = new BFloat16(BitConverter.Int32BitsToSingle((int)value));
        }

        return result;
    }

    private static Half[] ConvertHalf(ReadOnlySpan<byte> data)
    {
        var ushortSpan = MemoryMarshal.Cast<byte, ushort>(data);
        var result = new Half[ushortSpan.Length];

        for (int i = 0; i < ushortSpan.Length; i++)
        {
            result[i] = BitConverter.UInt16BitsToHalf(ushortSpan[i]);
        }

        return result;
    }

    private static Complex[] ConvertComplex64(ReadOnlySpan<byte> data)
    {
        var floatSpan = MemoryMarshal.Cast<byte, float>(data);
        var result = new Complex[floatSpan.Length / 2];

        for (int i = 0; i < result.Length; i++)
        {
            result[i] = new Complex(floatSpan[i * 2], floatSpan[i * 2 + 1]);
        }

        return result;
    }

    private static Complex[] ConvertComplex128(ReadOnlySpan<byte> data)
    {
        var doubleSpan = MemoryMarshal.Cast<byte, double>(data);
        var result = new Complex[doubleSpan.Length / 2];

        for (int i = 0; i < result.Length; i++)
        {
            result[i] = new Complex(
                doubleSpan[i * 2],
                doubleSpan[i * 2 + 1]
            );
        }

        return result;
    }

    internal static T GetValue<T>(this AttributeProto attribute)
    {
        var value = GetValue(attribute);

        if (value is T typed)
        {
            return typed;
        }

        throw new InvalidCastException($"Attribute '{attribute.Name}' is {value.GetType().Name}, not {typeof(T).Name}");
    }

    internal static object GetValue(this AttributeProto attribute)
    {
        return attribute.Type switch
        {
            AttributeProto.Types.AttributeType.Float => attribute.F,
            AttributeProto.Types.AttributeType.Int => attribute.I,
            AttributeProto.Types.AttributeType.String => attribute.S.ToStringUtf8(),

            AttributeProto.Types.AttributeType.Tensor => FromProto(attribute.T),
            AttributeProto.Types.AttributeType.Graph => new OnnxGraph(attribute.G),
            AttributeProto.Types.AttributeType.SparseTensor => FromProto(attribute.SparseTensor),

            AttributeProto.Types.AttributeType.Floats => attribute.Floats.ToArray(),
            AttributeProto.Types.AttributeType.Ints => attribute.Ints.ToArray(),
            AttributeProto.Types.AttributeType.Strings => attribute.Strings.Select(x => x.ToStringUtf8()).ToArray(),

            AttributeProto.Types.AttributeType.Tensors => attribute.Tensors.Select(x => FromProto(x)).ToArray(),
            AttributeProto.Types.AttributeType.Graphs => attribute.Graphs.Select(x => new OnnxGraph(x)).ToArray(),
            AttributeProto.Types.AttributeType.SparseTensors => attribute.SparseTensors.Select(x => FromProto(x)).ToArray(),

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
                attribute.Type = AttributeProto.Types.AttributeType.Tensor;
                break;

            case OnnxGraph g:
                attribute.G = g.ToProto();
                attribute.Type = AttributeProto.Types.AttributeType.Graph;
                break;

            case OnnxSparseTensorBase sparseTensor:
                attribute.SparseTensor = sparseTensor.ToProto();
                attribute.Type = AttributeProto.Types.AttributeType.SparseTensor;
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
                attribute.Tensors.Clear();
                foreach (var x in tensorArray)
                {
                    attribute.Tensors.Add(x.ToProto());
                }

                attribute.Type = AttributeProto.Types.AttributeType.Tensors;
                break;

            case OnnxGraph[] graphArray:
                attribute.Graphs.Clear();
                foreach (var x in graphArray)
                {
                    attribute.Graphs.Add(x.ToProto());
                }

                attribute.Type = AttributeProto.Types.AttributeType.Graphs;
                break;

            case OnnxSparseTensorBase[] sparseTensorArray:
                attribute.SparseTensors.Clear();
                foreach (var x in sparseTensorArray)
                {
                    attribute.SparseTensors.Add(x.ToProto());
                }

                attribute.Type = AttributeProto.Types.AttributeType.SparseTensors;
                break;

            default:
                throw new NotSupportedException($"Unsupported attribute type {typeof(T).Name}");
        }
    }

    internal static void SetValue<T>(this TensorProto tensor, T value, params long[] shape)
    {
        tensor.Dims.Clear();
        tensor.Dims.AddRange(shape);

        tensor.RawData = ByteString.Empty;

        switch (value)
        {
            case float[] f:
                tensor.DataType = (int)TensorProto.Types.DataType.Float;
                tensor.RawData = Pack(f);
                break;

            case double[] d:
                tensor.DataType = (int)TensorProto.Types.DataType.Double;
                tensor.RawData = Pack(d);
                break;

            case int[] i32:
                tensor.DataType = (int)TensorProto.Types.DataType.Int32;
                tensor.RawData = Pack(i32);
                break;

            case long[] i64:
                tensor.DataType = (int)TensorProto.Types.DataType.Int64;
                tensor.RawData = Pack(i64);
                break;

            case byte[] u8:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint8;
                tensor.RawData = ByteString.CopyFrom(u8);
                break;

            case sbyte[] i8:
                tensor.DataType = (int)TensorProto.Types.DataType.Int8;
                tensor.RawData = Pack(i8);
                break;

            case short[] i16:
                tensor.DataType = (int)TensorProto.Types.DataType.Int16;
                tensor.RawData = Pack(i16);
                break;

            case ushort[] u16:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint16;
                tensor.RawData = Pack(u16);
                break;

            case uint[] u32:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint32;
                tensor.RawData = Pack(u32);
                break;

            case ulong[] u64:
                tensor.DataType = (int)TensorProto.Types.DataType.Uint64;
                tensor.RawData = Pack(u64);
                break;

            case bool[] b:
                tensor.DataType = (int)TensorProto.Types.DataType.Bool;
                tensor.RawData = ByteString.CopyFrom(b.Select(x => (byte)(x ? 1 : 0)).ToArray());
                break;

            case Half[] h:
                tensor.DataType = (int)TensorProto.Types.DataType.Float16;
                tensor.RawData = PackHalf(h);
                break;

            case BFloat16[] bf when typeof(T) == typeof(BFloat16[]):
                tensor.DataType = (int)TensorProto.Types.DataType.Bfloat16;
                tensor.RawData = PackBFloat16(bf);
                break;

            case Complex64[] c64:
                tensor.DataType = (int)TensorProto.Types.DataType.Complex64;
                tensor.RawData = PackComplex64(c64);
                break;

            case System.Numerics.Complex[] c128:
                tensor.DataType = (int)TensorProto.Types.DataType.Complex128;
                tensor.RawData = PackComplex128(c128);
                break;

            case string[] s:
                tensor.DataType = (int)TensorProto.Types.DataType.String;
                tensor.StringData.Clear();
                tensor.StringData.AddRange(s.Select(ByteString.CopyFromUtf8));
                break;

            default:
                throw new NotSupportedException($"Unsupported tensor type {typeof(T)}");
        }
    }

    private static ByteString Pack<T>(T[] data) where T : struct
    {
        var span = System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan());
        return ByteString.CopyFrom(span.ToArray());
    }

    private static ByteString PackHalf(Half[] data)
    {
        var buffer = new byte[data.Length * 2];

        for (int i = 0; i < data.Length; i++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(data[i]);
            buffer[i * 2] = (byte)(bits & 0xFF);
            buffer[i * 2 + 1] = (byte)(bits >> 8);
        }

        return ByteString.CopyFrom(buffer);
    }

    private static ByteString PackBFloat16(BFloat16[] data)
    {
        var buffer = new byte[data.Length * 2];

        for (int i = 0; i < data.Length; i++)
        {
            uint bits = (uint)BitConverter.SingleToInt32Bits(data[i].ToSingle());
            ushort bf = (ushort)(bits >> 16);

            buffer[i * 2] = (byte)(bf & 0xFF);
            buffer[i * 2 + 1] = (byte)(bf >> 8);
        }

        return ByteString.CopyFrom(buffer);
    }

    private static ByteString PackComplex64(Complex64[] data)
    {
        var buffer = new float[data.Length * 2];

        for (int i = 0; i < data.Length; i++)
        {
            buffer[i * 2] = (float)data[i].Real;
            buffer[i * 2 + 1] = (float)data[i].Imaginary;
        }

        return Pack(buffer);
    }

    private static ByteString PackComplex128(System.Numerics.Complex[] data)
    {
        var buffer = new double[data.Length * 2];

        for (int i = 0; i < data.Length; i++)
        {
            buffer[i * 2] = data[i].Real;
            buffer[i * 2 + 1] = data[i].Imaginary;
        }

        return Pack(buffer);
    }
}

public abstract class OnnxTensor : IOnnxGraphEdge
{
    public abstract string Name { get; }
    public abstract TensorProto.Types.DataType DataType { get; }
    internal abstract TensorProto ToProto();
}

public class OnnxTensor<T> : OnnxTensor
{
    public override string Name => _tensor.Name;
    public override TensorProto.Types.DataType DataType => (TensorProto.Types.DataType)_tensor.DataType;
    public TensorProto.Types.DataLocation DataLocation { set; get; }
    public IEnumerable<long> Shape { get; set; }
    public IEnumerable<T> Value { get; set; }

    private readonly TensorProto _tensor;

    internal OnnxTensor(TensorProto tensor)
    {
        _tensor = tensor;

        DataLocation = tensor.DataLocation;
        Shape = tensor.Dims.ToList();
        Value = OnnxHelper.GetValue<T>(tensor);
    }

    internal override TensorProto ToProto()
    {
        var newTensor = _tensor.Clone();
        newTensor.Name = Name;
        newTensor.DataType = (int)DataType;
        newTensor.DataLocation = DataLocation;

        newTensor.Dims.Clear();
        foreach (var size in Shape)
        {
            newTensor.Dims.Add(size);
        }

        var data = Value.ToArray() ?? [];
        newTensor.SetValue(data, _tensor.Dims.ToArray());

        return newTensor;
    }
}

public abstract class OnnxSparseTensorBase : IOnnxGraphEdge
{
    public abstract string Name { get; }
    public abstract TensorProto.Types.DataType DataType { get; }
    public abstract OnnxTensor Value { get; }
    internal abstract SparseTensorProto ToProto();
}

public class OnnxSparseTensor<T> : OnnxSparseTensorBase
{
    public override string Name => _value.Name;
    public override TensorProto.Types.DataType DataType => _value.DataType;
    public override OnnxTensor<T> Value => _value;

    private readonly SparseTensorProto _tensor;
    private readonly OnnxTensor<T> _value;

    internal OnnxSparseTensor(SparseTensorProto tensor)
    {
        _tensor = tensor;
        _value = (OnnxTensor<T>)OnnxHelper.FromProto(tensor.Values);
    }

    internal override SparseTensorProto ToProto()
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

internal class LazyDictionary<TKey, TValue> : KeyedCollection<TKey, TValue> where TKey : notnull
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

public readonly struct BFloat16
{
    public ushort Value { get; }

    public BFloat16(float value)
    {
        uint bits = (uint)BitConverter.SingleToInt32Bits(value);
        Value = (ushort)(bits >> 16);
    }

    public float ToSingle()
    {
        uint bits = (uint)Value << 16;
        return BitConverter.Int32BitsToSingle((int)bits);
    }
}

public readonly struct Complex64
{
    public double Real { get; }
    public double Imaginary { get; }

    public Complex64(double real, double imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }
}
