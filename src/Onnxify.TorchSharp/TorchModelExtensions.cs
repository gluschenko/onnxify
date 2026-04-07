namespace Onnxify.TorchSharp;

public static class TorchModelExtensions
{
    private static readonly IEnumerable<ITorchModuleExporter> _exporters = [
        new ConvExporter(),
        new ReluExporter(),
        new MaxPool2dExporter(),
        new DropoutExporter(),
        new LinearExporter(),
        new AdaptiveAvgPool2dExporter(),
        new SequentialExporter(),
        new EmbeddingExporter(),
    ];

    public static IOnnxGraphEdge Export(
        this TorchModule module,
        OnnxGraph graph,
        IOnnxGraphEdge input
    )
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(input);

        foreach (var exporter in _exporters)
        {
            if (exporter.IsMatch(module))
            {
                var edge = exporter.Export(graph, module, input);
                return edge;
            }
        }

        throw new NotImplementedException($"Not implemented for '{module.GetType().FullName}'");
    }
}
