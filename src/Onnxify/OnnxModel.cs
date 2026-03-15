using Google.Protobuf;
using Onnx;
using System.Collections.ObjectModel;

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
            _model.WriteTo(fileStream);
        }
    }

    public class OnnxGraph
    {
        public string Name
        {
            get => _graph.Name;
            set => _graph.Name = value;
        }

        public IReadOnlyCollection<OnnxTensor> Tensors => _tensors.Values;
        public IReadOnlyCollection<OnnxValue> Constraints => _constraints.Values;
        public IReadOnlyCollection<OnnxValue> Inputs => _inputs.Values;
        public IReadOnlyCollection<OnnxValue> Outputs => _outputs.Values;
        public IReadOnlyCollection<OnnxNode> Nodes => _nodes.Values;

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
    }

    public class OnnxNode
    {
        public string Name => _node.Name;
        public string OpType => _node.OpType;
        public IReadOnlyList<string> Inputs => _node.Input;
        public IReadOnlyList<string> Outputs => _node.Output;

        private readonly NodeProto _node;
        private readonly OnnxGraph _graph;

        internal OnnxNode(NodeProto node, OnnxGraph graph)
        {
            _node = node;
            _graph = graph;
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
            var node = _node.Clone();


            return node;
        }
    }

    public interface IOnnxValue
    {
        public string Name { get; }
        public OnnxGraph GetGraph();
    }

    public class OnnxTensor : IOnnxValue
    {
        public string Name => _tensor.Name;

        private readonly TensorProto _tensor;
        private readonly OnnxGraph _graph;

        internal OnnxTensor(TensorProto tensor, OnnxGraph graph)
        {
            _tensor = tensor;
            _graph = graph;
        }

        public OnnxGraph GetGraph()
        {
            return _graph;
        }
    }

    public class OnnxValue : IOnnxValue
    {
        public string Name => _valueInfo.Name;

        private readonly ValueInfoProto _valueInfo;
        private readonly OnnxGraph _graph;

        internal OnnxValue(ValueInfoProto valueInfo, OnnxGraph graph)
        {
            _valueInfo = valueInfo;
            _graph = graph;
        }

        public OnnxGraph GetGraph()
        {
            return _graph;
        }
    }
}
