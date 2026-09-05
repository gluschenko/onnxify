namespace Onnxify;

/// <summary>Categories of graph data that <see cref="OnnxGraph.Clean"/> may remove.</summary>
[Flags]
public enum OnnxGraphCleanupFlags : byte
{
    Nodes = 1 << 0,
    Values = 1 << 1,
    Initializers = 1 << 2,
    Inputs = 1 << 3,
    Outputs = 1 << 4,
    Subgraphs = 1 << 5,
    Annotations = 1 << 6,
    ValueInfo = Values,
    Constants = Initializers,
    NestedGraphs = Subgraphs,
    QuantizationAnnotations = Annotations,
    All = 0x7F
}

/// <summary>Identifies the graph-member category represented by a cleanup item.</summary>
public enum OnnxGraphCleanupItemType : byte
{
    Node,
    Value,
    Initializer,
    SparseInitializer,
    Input,
    Output,
    Annotation,
}

/// <summary>Describes one named graph member removed during cleanup.</summary>
public sealed class OnnxGraphCleanupItem
{
    public required string Name { get; init; }
    public required OnnxGraphCleanupItemType Type { get; init; }
}

/// <summary>Reports the graph members removed by one cleanup pass.</summary>
public sealed class OnnxGraphCleanupReport
{
    public required int NodesRemoved { get; init; }
    public required int ValuesRemoved { get; init; }
    public required int InitializersRemoved { get; init; }
    public required int SparseInitializersRemoved { get; init; }
    public required int InputsRemoved { get; init; }
    public required int OutputsRemoved { get; init; }
    public required int SubgraphsCleaned { get; init; }
    public required int AnnotationsRemoved { get; init; }
    public required IReadOnlyList<OnnxGraphCleanupItem> RemovedItems { get; init; }

    public int TotalRemoved => RemovedItems.Count;
}
