using Google.Protobuf;
using Onnx;

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
