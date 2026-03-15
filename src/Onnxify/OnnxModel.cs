using Google.Protobuf;
using Google.Protobuf.Collections;
using Onnx;
using Onnxify.Abstractions;
using static TorchSharp.torch.optim.lr_scheduler.impl.CyclicLR;

namespace Onnxify
{
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
        public RepeatedField<StringStringEntryProto> MetadataProps => _model.MetadataProps;
        public RepeatedField<TrainingInfoProto> TrainingInfo => _model.TrainingInfo;
        public RepeatedField<OperatorSetIdProto> OpsetImport => _model.OpsetImport;

        public OnnxGraph Graph => _graph;

        private readonly ModelProto _model;
        private readonly OnnxGraph _graph;

        internal OnnxModel(ModelProto model)
        {
            _model = model;
            _graph = new OnnxGraph(model.Graph);
        }

        public static OnnxModel Create()
        {
            var model = new ModelProto
            {
                Graph = new GraphProto(),
            };

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

        public IReadOnlyCollection<OnnxNode> Nodes => _nodes.Values;
        public IReadOnlyCollection<OnnxTensor> Tensors => _tensors.Values;
        public IReadOnlyCollection<OnnxValue> Constraints => _constraints.Values;

        private readonly GraphProto _graph;
        private readonly Dictionary<string, OnnxNode> _nodes;
        private readonly Dictionary<string, OnnxTensor> _tensors;
        private readonly Dictionary<string, OnnxValue> _constraints;

        internal OnnxGraph(GraphProto graph)
        {
            _graph = graph;
            _nodes = graph.Node.ToDictionary(x => x.Name, x => new OnnxNode(x));
            _tensors = graph.Initializer.ToDictionary(x => x.Name, x => new OnnxTensor(x));
            _constraints = graph.ValueInfo.ToDictionary(x => x.Name, x => new OnnxValue(x));
        }
    }

    public class OnnxNode
    {
        public string Name
        {
            get => _node.Name;
            set => _node.Name = value;
        }

        private readonly NodeProto _node;

        internal OnnxNode(NodeProto node)
        {
            _node = node;
        }
    }

    public class OnnxTensor
    {
        private readonly TensorProto _tensor;

        internal OnnxTensor(TensorProto tensor)
        {
            _tensor = tensor;
        }
    }

    public class OnnxValue
    {
        private readonly ValueInfoProto _valueInfo;

        internal OnnxValue(ValueInfoProto valueInfo)
        {
            _valueInfo = valueInfo;
        }
    }
}
