using Google.Protobuf;
using Onnx;
using Onnxify.Data;

namespace Onnxify;

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

    public IReadOnlyList<KeyValuePair<string, string>> MetadataProps => _metadataProps;
    public IReadOnlyList<KeyValuePair<string, long>> OpsetImport => _opsetImport;
    public IReadOnlyList<TrainingInfo> TrainingInfo => _trainingInfo;

    public OnnxGraph Graph => _graph;

    private readonly ModelProto _model;
    private readonly OnnxGraph _graph;
    private readonly OnnxModelBaseOptions _options;

    private readonly LazyDictionary<string, KeyValuePair<string, string>> _metadataProps = new(x => x.Key, StringComparer.Ordinal);
    private readonly LazyDictionary<string, KeyValuePair<string, long>> _opsetImport = new(x => x.Key, StringComparer.Ordinal);
    private readonly List<TrainingInfo> _trainingInfo = new();

    internal OnnxModel(ModelProto model, OnnxModelBaseOptions options)
    {
        _model = model;
        _options = options;
        _graph = new OnnxGraph(model.Graph, _options);

        foreach (var x in model.MetadataProps)
        {
            _metadataProps.Add(new KeyValuePair<string, string>(x.Key, x.Value));
        }

        foreach (var x in model.OpsetImport)
        {
            _opsetImport.Add(new KeyValuePair<string, long>(x.Domain, x.Version));
        }

        foreach (var x in model.TrainingInfo)
        {
            _trainingInfo.Add(Onnxify.TrainingInfo.FromProto(x, options));
        }
    }

    public static OnnxModel Create(OnnxModelCreationOptions? options = null)
    {
        options ??= new OnnxModelCreationOptions();

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

        return new OnnxModel(model, new OnnxModelBaseOptions
        {
            DataLocation = options.DataLocation,
        });
    }

    public static OnnxModel FromFile(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("File not found", path);
        }

        var data = File.ReadAllBytes(path);
        var model = ModelProto.Parser.ParseFrom(data);
        return new OnnxModel(model, new OnnxModelBaseOptions
        {
            DataLocation = Path.GetDirectoryName(path),
        });
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

        newModel.MetadataProps.Set(_metadataProps.Select(x => new StringStringEntryProto { Key = x.Key, Value = x.Value }));
        newModel.OpsetImport.Set(_opsetImport.Select(x => new OperatorSetIdProto { Domain = x.Key, Version = x.Value }));
        newModel.TrainingInfo.Set(_trainingInfo.Select(x => x.ToProto()));

        return newModel;
    }

    public override string ToString()
    {
        return $"""
        OnnxModel(
            ProducerName={ProducerName},
            ProducerVersion={ProducerVersion},
            ModelVersion={ModelVersion},
            IrVersion={IrVersion},
            Domain={Domain}
        )
        """;
    }
}

public interface IOnnxGraphNode
{
    public string Name { get; }
}

public interface IOnnxGraphEdge
{
    public string Name { get; }
}

public class TrainingInfo
{
    public required OnnxGraph Initialization { get; init; }
    public required OnnxGraph Algorithm { get; init; }
    public required Dictionary<string, string> InitializationBinding { get; init; }
    public required Dictionary<string, string> UpdateBinding { get; init; }

    internal TrainingInfoProto ToProto()
    {
        return new TrainingInfoProto
        {
            Initialization = Initialization.ToProto(),
            Algorithm = Algorithm.ToProto(),
            InitializationBinding = { InitializationBinding.Select(x => new StringStringEntryProto { Key = x.Key, Value = x.Value }).ToList() },
            UpdateBinding = { UpdateBinding.Select(x => new StringStringEntryProto { Key = x.Key, Value = x.Value }).ToList() },
        };
    }

    internal static TrainingInfo FromProto(TrainingInfoProto proto, OnnxModelBaseOptions options)
    {
        return new TrainingInfo 
        {
            Initialization = new OnnxGraph(proto.Initialization, options),
            Algorithm = new OnnxGraph(proto.Algorithm, options),
            InitializationBinding = proto.InitializationBinding.ToDictionary(x => x.Key, x => x.Value),
            UpdateBinding = proto.UpdateBinding.ToDictionary(x => x.Key, x => x.Value),
        };
    }
}
