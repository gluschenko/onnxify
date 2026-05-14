namespace Onnxify;

internal sealed class OnnxBundledOperatorSchema
{
    public required string Domain { get; init; }
    public required string Name { get; init; }
    public required int SinceVersion { get; init; }
    public required int MinimumInputs { get; init; }
    public required int? MaximumInputs { get; init; }
    public required int MinimumOutputs { get; init; }
    public required int? MaximumOutputs { get; init; }
    public required HashSet<string> KnownAttributes { get; init; }
    public required HashSet<string> RequiredAttributes { get; init; }
    public required string SourceDescription { get; init; }
}
