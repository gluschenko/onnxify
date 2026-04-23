using Google.Protobuf;
using Onnx;
using Onnxify.Data;
using Onnxify.Helpers;

namespace Onnxify;

/// <summary>
/// Represents an ONNX model as editable metadata plus a single graph that can be loaded, inspected, modified, and written back to the ONNX protobuf format.
/// </summary>
/// <remarks>
/// Use <see cref="FromFile"/> when preserving an existing model and <see cref="Create"/> when authoring a graph from scratch. The wrapper keeps the original protobuf data where possible, so edits to exposed collections and metadata are reflected when <see cref="Save"/> is called.
/// </remarks>
public class OnnxModel
{
    /// <summary>
    /// Identifies the tool or application that produced the model; useful for downstream diagnostics when several exporters are present in a pipeline.
    /// </summary>
    public string ProducerName
    {
        get => _model.ProducerName;
        set => _model.ProducerName = value;
    }

    /// <summary>
    /// Records the producer version independently from the model's own semantic version.
    /// </summary>
    public string ProducerVersion
    {
        get => _model.ProducerVersion;
        set => _model.ProducerVersion = value;
    }

    /// <summary>
    /// Carries the version assigned by the model author, not the ONNX IR or opset version.
    /// </summary>
    public long ModelVersion
    {
        get => _model.ModelVersion;
        set => _model.ModelVersion = value;
    }

    /// <summary>
    /// Selects the ONNX IR format version used when the model is serialized.
    /// </summary>
    /// <remarks>
    /// Change this only when you know the target runtime supports the chosen IR version; operator availability is controlled separately through <see cref="OpsetImport"/>.
    /// </remarks>
    public long IrVersion
    {
        get => _model.IrVersion;
        set => _model.IrVersion = value;
    }

    /// <summary>
    /// Stores a free-form model-level note that travels in the ONNX <c>doc_string</c> field.
    /// </summary>
    public string Document
    {
        get => _model.DocString;
        set => _model.DocString = value;
    }

    /// <summary>
    /// Names the model's logical domain, which consumers may use to distinguish model families or vendor-specific packages.
    /// </summary>
    public string Domain
    {
        get => _model.Domain;
        set => _model.Domain = value;
    }

    /// <summary>
    /// Exposes model metadata as key/value pairs keyed by exact, case-sensitive metadata names.
    /// </summary>
    public IReadOnlyList<KeyValuePair<string, string>> MetadataProps => _metadataProps;

    /// <summary>
    /// Lists the operator-set versions the graph expects for each ONNX domain.
    /// </summary>
    /// <remarks>
    /// The empty domain string represents the standard <c>ai.onnx</c> domain.
    /// </remarks>
    public IReadOnlyList<OperationSet> OpsetImport => _opsetImport;

    /// <summary>
    /// Contains optional ONNX training graphs and their parameter bindings when the source model includes training metadata.
    /// </summary>
    public IReadOnlyList<TrainingInfo> TrainingInfo => _trainingInfo;

    /// <summary>
    /// Gets the main computation graph. Mutations made through this graph are included in the next save.
    /// </summary>
    public OnnxGraph Graph => _graph;

    private readonly ModelProto _model;
    private readonly OnnxGraph _graph;
    private readonly OnnxModelBaseOptions _options;

    private readonly LazyDictionary<string, KeyValuePair<string, string>> _metadataProps = new(x => x.Key, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OperationSet> _opsetImport = new(x => x.Domain, StringComparer.Ordinal);
    private readonly List<TrainingInfo> _trainingInfo = [];

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
            _opsetImport.Add(new OperationSet
            {
                Domain = x.Domain,
                Version = x.Version,
            });
        }

        foreach (var x in model.TrainingInfo)
        {
            _trainingInfo.Add(Onnxify.TrainingInfo.FromProto(x, options));
        }
    }

    /// <summary>
    /// Creates an empty ONNX model initialized with a graph and a default opset import.
    /// </summary>
    /// <param name="options">Creation defaults for IR version, standard-domain opset, producer metadata, and external-data handling.</param>
    /// <returns>A model ready for graph construction.</returns>
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

    /// <summary>
    /// Loads an ONNX model from disk and configures relative external tensor data to resolve from the model file's directory.
    /// </summary>
    /// <param name="path">Path to a serialized <c>.onnx</c> file.</param>
    /// <returns>A mutable wrapper around the loaded model.</returns>
    /// <exception cref="FileNotFoundException">Thrown when <paramref name="path"/> does not exist.</exception>
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

    /// <summary>
    /// Serializes the current model state to an ONNX file.
    /// </summary>
    /// <param name="path">Destination path for the serialized model.</param>
    /// <param name="overwrite">Set to <see langword="true"/> to replace an existing file.</param>
    /// <exception cref="IOException">Thrown when the destination exists and <paramref name="overwrite"/> is <see langword="false"/>.</exception>
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
        newModel.OpsetImport.Set(_opsetImport.Select(x => new OperatorSetIdProto { Domain = x.Domain, Version = x.Version }));
        newModel.TrainingInfo.Set(_trainingInfo.Select(x => x.ToProto()));

        return newModel;
    }

    /// <summary>
    /// Returns a diagnostic view of the model metadata and graph contents.
    /// </summary>
    /// <returns>A multiline string intended for inspection, not for stable serialization.</returns>
    public override string ToString()
    {
        var producerName = string.IsNullOrWhiteSpace(ProducerName) ? "<unknown>" : ProducerName;
        var domain = string.IsNullOrWhiteSpace(Domain) ? "<default>" : Domain;
        return $"""
            OnnxModel(
                Producer={producerName},
                Version={ProducerVersion},
                ModelVersion={ModelVersion},
                IrVersion={IrVersion},
                Domain={domain},
                Graph={Graph.ToString().Indent(1)}
            )
            """;
    }

    /// <summary>
    /// Adds or replaces a metadata property using an exact, case-sensitive key.
    /// </summary>
    /// <param name="key">Metadata key as it should appear in the ONNX model.</param>
    /// <param name="value">Metadata value to store.</param>
    public void AddMetadataProps(string key, string value)
    {
        _metadataProps[key] = new KeyValuePair<string, string>(key, value);
    }

    /// <summary>
    /// Removes all opset imports so callers can rebuild the exact domain/version set required by a graph.
    /// </summary>
    /// <remarks>
    /// ONNX runtimes need at least the domains used by graph nodes; call <see cref="SetOpsetImport"/> after clearing unless the model is intentionally incomplete.
    /// </remarks>
    public void ClearOpsetImports()
    {
        _opsetImport.Clear();
    }

    /// <summary>
    /// Adds or replaces the opset version for an operator domain.
    /// </summary>
    /// <param name="domain">Operator domain. Use an empty string for the standard <c>ai.onnx</c> domain.</param>
    /// <param name="version">Opset version required for nodes in that domain.</param>
    public void SetOpsetImport(string domain, long version)
    {
        ArgumentNullException.ThrowIfNull(domain);

        _opsetImport[domain] = new OperationSet
        {
            Domain = domain,
            Version = version,
        };
    }
}

/// <summary>
/// Common contract for graph objects that occupy the node namespace.
/// </summary>
public interface IOnnxGraphNode
{
    /// <summary>
    /// Gets the exact graph-local node name used by node references and diagnostics.
    /// </summary>
    public string Name { get; }
}

/// <summary>
/// Common contract for values that can be connected to node inputs or outputs.
/// </summary>
/// <remarks>
/// Implementations include typed values, initializers, sparse tensors, and anonymous edges created when ONNX only provides a wire name.
/// </remarks>
public interface IOnnxGraphEdge
{
    /// <summary>
    /// Gets the exact wire name written into node input and output lists.
    /// </summary>
    public string Name { get; }
}

/// <summary>
/// Describes one imported ONNX operator-set domain and version.
/// </summary>
public class OperationSet
{
    /// <summary>
    /// Gets the ONNX operator domain; the empty string means the standard <c>ai.onnx</c> domain.
    /// </summary>
    public required string Domain { get; init; }

    /// <summary>
    /// Gets the opset version that operator schemas are resolved against for <see cref="Domain"/>.
    /// </summary>
    public required long Version { get; init; }
}

/// <summary>
/// Represents ONNX training metadata containing initialization and algorithm graphs plus their parameter bindings.
/// </summary>
/// <remarks>
/// Most inference-only models do not include this block. When present, it is preserved during round-tripping so tooling can inspect or carry training graphs forward.
/// </remarks>
public class TrainingInfo
{
    /// <summary>
    /// Gets the graph that initializes training state before the algorithm graph runs.
    /// </summary>
    public required OnnxGraph Initialization { get; init; }

    /// <summary>
    /// Gets the graph that describes training-time updates.
    /// </summary>
    public required OnnxGraph Algorithm { get; init; }

    /// <summary>
    /// Maps names produced by <see cref="Initialization"/> to names consumed by the model graph.
    /// </summary>
    public required Dictionary<string, string> InitializationBinding { get; init; }

    /// <summary>
    /// Maps update outputs from <see cref="Algorithm"/> back to model graph state.
    /// </summary>
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
