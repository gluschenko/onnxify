using Google.Protobuf;
using Onnx;
using System.Collections.ObjectModel;
using System.Xml.Linq;
using static Onnx.TensorProto.Types;
using static Tensorboard.CostGraphDef.Types;

namespace Onnxify
{
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

        // TODO: incapsulate
        public IReadOnlyList<StringStringEntryProto> MetadataProps => _metadataProps;
        public IReadOnlyList<TrainingInfoProto> TrainingInfo => _trainingInfo;
        public IReadOnlyList<OperatorSetIdProto> OpsetImport => _opsetImport;

        private readonly ObservableCollection<StringStringEntryProto> _metadataProps;
        private readonly ObservableCollection<TrainingInfoProto> _trainingInfo;
        private readonly ObservableCollection<OperatorSetIdProto> _opsetImport;

        public OnnxGraph Graph => _graph;

        private readonly ModelProto _model;
        private readonly OnnxGraph _graph;

        internal OnnxModel(ModelProto model)
        {
            _model = model;
            _graph = new OnnxGraph(model.Graph);

            _metadataProps = [];
            foreach (var x in _model.MetadataProps)
            {
                _metadataProps.Add(x);
            }

            _trainingInfo = [];
            foreach (var x in _model.TrainingInfo)
            {
                _trainingInfo.Add(x);
            }

            _opsetImport = [];
            foreach (var x in _model.OpsetImport)
            {
                _opsetImport.Add(x);
            }
        }

        public static OnnxModel Create(OnnxModelCreateOptions? options)
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
                throw new Exception($"File already exists at '{path}'");
            }

            using var fileStream = File.Create(path);

            var newModel = ToProto();
            newModel.WriteTo(fileStream);
        }

        internal ModelProto ToProto()
        {
            var newModel = _model.Clone();
            newModel.Graph = _graph.ToProto();
            return newModel;
        }
    }

    public class OnnxGraph
    {
        public string Name { get; init; }

        private readonly GraphProto _graph;

        private readonly Dictionary<string, OnnxTensor> _tensors;
        private readonly Dictionary<string, OnnxValue> _constraints;
        private readonly Dictionary<string, OnnxValue> _inputs;
        private readonly Dictionary<string, OnnxValue> _outputs;
        private readonly Dictionary<string, OnnxNode> _nodes;

        internal OnnxGraph(GraphProto graph)
        {
            _graph = graph;

            _tensors = graph.Initializer.ToDictionary(x => x.Name, x => new OnnxTensor(x, this));
            _constraints = graph.ValueInfo.ToDictionary(x => x.Name, x => new OnnxValue(x, this));
            _inputs = graph.Input.ToDictionary(x => x.Name, x => new OnnxValue(x, this));
            _outputs = graph.Output.ToDictionary(x => x.Name, x => new OnnxValue(x, this));
            _nodes = graph.Node.ToDictionary(x => x.Name, x => new OnnxNode(x, this));

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

        public IOnnxValue? GetValue(string name)
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
            foreach (var (_, x) in _tensors)
            {
                newGraph.Initializer.Add(x.ToProto());
            }

            newGraph.ValueInfo.Clear();
            foreach (var (_, x) in _constraints)
            {
                newGraph.ValueInfo.Add(x.ToProto());
            }

            newGraph.Input.Clear();
            foreach (var (_, x) in _inputs)
            {
                newGraph.Input.Add(x.ToProto());
            }

            newGraph.Output.Clear();
            foreach (var (_, x) in _outputs)
            {
                newGraph.Output.Add(x.ToProto());
            }

            newGraph.Node.Clear();
            foreach (var (_, x) in _nodes)
            {
                newGraph.Node.Add(x.ToProto());
            }

            return newGraph;
        }
    }

    public class OnnxNode
    {
        public string Name { get; init; }
        public string OpType { get; set; }
        public IReadOnlyList<string> Inputs { get; set; }
        public IReadOnlyList<string> Outputs { get; set; }

        private readonly NodeProto _node;
        private readonly OnnxGraph _graph;

        internal OnnxNode(NodeProto node, OnnxGraph graph)
        {
            _node = node;
            _graph = graph;

            Name = node.Name;
            OpType = node.OpType;
            Inputs = node.Input.ToArray();
            Outputs = node.Output.ToArray();
        }

        public OnnxGraph GetGraph()
        {
            return _graph;
        }

        public IEnumerable<IOnnxValue> GetInputValues()
        {
            foreach (var x in Inputs)
            {
                var value = _graph.GetValue(x);
                if (value is not null)
                {
                    yield return value;
                }
            }
        }

        public IEnumerable<IOnnxValue> GetOutputValues()
        {
            foreach (var x in Outputs)
            {
                var value = _graph.GetValue(x);
                if (value is not null)
                {
                    yield return value;
                }
            }
        }

        internal NodeProto ToProto()
        {
            var newNode = _node.Clone();
            newNode.Name = Name;
            newNode.OpType = OpType;

            newNode.Input.Clear();
            foreach (var x in Inputs)
            {
                newNode.Input.Add(x);
            }

            newNode.Output.Clear();
            foreach (var x in Outputs)
            {
                newNode.Output.Add(x);
            }

            return newNode;
        }
    }

    public interface IOnnxValue
    {
        public string Name { get; }
        public OnnxGraph GetGraph();
    }

    public class OnnxTensor : IOnnxValue
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

    public class OnnxValue : IOnnxValue
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
}
