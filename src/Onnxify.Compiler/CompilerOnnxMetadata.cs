namespace Onnxify.Compiler;

/// <summary>Immutable ONNX operator-set import metadata.</summary>
public sealed class CompilerOpsetImport : IEquatable<CompilerOpsetImport>
{
    public CompilerOpsetImport(string domain, long version)
    {
        Domain = domain ?? string.Empty;
        Version = version;
    }

    public string Domain { get; }

    public long Version { get; }

    public bool Equals(CompilerOpsetImport? other)
    {
        return other is not null
            && string.Equals(Domain, other.Domain, StringComparison.Ordinal)
            && Version == other.Version;
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerOpsetImport);

    public override int GetHashCode() => CompilerStructural.Combine(17, Domain, Version);
}

/// <summary>Immutable graph quantization annotation metadata.</summary>
public sealed class CompilerQuantizationAnnotation : IEquatable<CompilerQuantizationAnnotation>
{
    public CompilerQuantizationAnnotation(
        string tensorName,
        IEnumerable<KeyValuePair<string, string>>? parameterTensorNames = null
    )
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name cannot be empty.", nameof(tensorName));
        }

        TensorName = tensorName;
        ParameterTensorNames = CompilerStructural.Copy(
            parameterTensorNames ?? Array.Empty<KeyValuePair<string, string>>(),
            nameof(parameterTensorNames));

        if (ParameterTensorNames.Select(x => x.Key).Distinct(StringComparer.Ordinal).Count() != ParameterTensorNames.Count)
        {
            throw new ArgumentException("Quantization parameter names must be unique.", nameof(parameterTensorNames));
        }
    }

    public string TensorName { get; }

    public IReadOnlyList<KeyValuePair<string, string>> ParameterTensorNames { get; }

    public bool Equals(CompilerQuantizationAnnotation? other)
    {
        return other is not null
            && string.Equals(TensorName, other.TensorName, StringComparison.Ordinal)
            && CompilerStructural.SequenceEqual(ParameterTensorNames, other.ParameterTensorNames);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerQuantizationAnnotation);

    public override int GetHashCode() => CompilerStructural.Combine(
        17,
        TensorName,
        CompilerStructural.GetHashCode(ParameterTensorNames));
}

/// <summary>Immutable model-level metadata carried alongside the main compiler graph.</summary>
public sealed class CompilerModelEnvelope : IEquatable<CompilerModelEnvelope>
{
    public CompilerModelEnvelope(
        string? producerName = null,
        string? producerVersion = null,
        long modelVersion = 0,
        long intermediateRepresentationVersion = 0,
        string? document = null,
        string? domain = null,
        IEnumerable<KeyValuePair<string, string>>? metadata = null,
        IEnumerable<CompilerOpsetImport>? opsetImports = null
    )
    {
        ProducerName = producerName ?? string.Empty;
        ProducerVersion = producerVersion ?? string.Empty;
        ModelVersion = modelVersion;
        IntermediateRepresentationVersion = intermediateRepresentationVersion;
        Document = document ?? string.Empty;
        Domain = domain ?? string.Empty;
        Metadata = CompilerStructural.Copy(
            metadata ?? Array.Empty<KeyValuePair<string, string>>(),
            nameof(metadata));
        OpsetImports = CompilerStructural.Copy(
            opsetImports ?? Array.Empty<CompilerOpsetImport>(),
            nameof(opsetImports));

        if (Metadata.Select(x => x.Key).Distinct(StringComparer.Ordinal).Count() != Metadata.Count)
        {
            throw new ArgumentException("Model metadata keys must be unique.", nameof(metadata));
        }

        if (OpsetImports.Select(x => x.Domain).Distinct(StringComparer.Ordinal).Count() != OpsetImports.Count)
        {
            throw new ArgumentException("Model opset domains must be unique.", nameof(opsetImports));
        }
    }

    public string ProducerName { get; }

    public string ProducerVersion { get; }

    public long ModelVersion { get; }

    public long IntermediateRepresentationVersion { get; }

    public string Document { get; }

    public string Domain { get; }

    public IReadOnlyList<KeyValuePair<string, string>> Metadata { get; }

    public IReadOnlyList<CompilerOpsetImport> OpsetImports { get; }

    public bool Equals(CompilerModelEnvelope? other)
    {
        return other is not null
            && string.Equals(ProducerName, other.ProducerName, StringComparison.Ordinal)
            && string.Equals(ProducerVersion, other.ProducerVersion, StringComparison.Ordinal)
            && ModelVersion == other.ModelVersion
            && IntermediateRepresentationVersion == other.IntermediateRepresentationVersion
            && string.Equals(Document, other.Document, StringComparison.Ordinal)
            && string.Equals(Domain, other.Domain, StringComparison.Ordinal)
            && CompilerStructural.SequenceEqual(Metadata, other.Metadata)
            && CompilerStructural.SequenceEqual(OpsetImports, other.OpsetImports);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerModelEnvelope);

    public override int GetHashCode() => CompilerStructural.Combine(
        17,
        ProducerName,
        ProducerVersion,
        ModelVersion,
        IntermediateRepresentationVersion,
        Document,
        Domain,
        CompilerStructural.GetHashCode(Metadata),
        CompilerStructural.GetHashCode(OpsetImports));
}
