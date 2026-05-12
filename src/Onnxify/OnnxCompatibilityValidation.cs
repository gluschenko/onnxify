using System.Collections.ObjectModel;

namespace Onnxify;

/// <summary>
/// Controls how ONNX compatibility validation should be performed.
/// </summary>
public enum OnnxCompatibilityValidationMode
{
    /// <summary>
    /// Validates the model against its own IR version and imported opsets.
    /// </summary>
    Structural = 0,

    /// <summary>
    /// Validates the model against an external runtime profile in addition to structural checks.
    /// </summary>
    TargetRuntime = 1,
}

/// <summary>
/// Describes the severity of one compatibility diagnostic.
/// </summary>
public enum OnnxCompatibilityDiagnosticSeverity
{
    /// <summary>
    /// Indicates a compatibility problem that should be treated as invalid.
    /// </summary>
    Error = 0,

    /// <summary>
    /// Indicates a best-effort or incomplete compatibility check.
    /// </summary>
    Warning = 1,
}

/// <summary>
/// Identifies one operator signature for runtime-profile validation.
/// </summary>
public sealed class OnnxOperatorSupport : IEquatable<OnnxOperatorSupport>
{
    /// <summary>
    /// Gets the ONNX operator domain; use an empty string for the default domain.
    /// </summary>
    public required string Domain { get; init; }

    /// <summary>
    /// Gets the ONNX operator type, such as <c>Add</c> or <c>Cast</c>.
    /// </summary>
    public required string OpType { get; init; }

    /// <inheritdoc />
    public bool Equals(OnnxOperatorSupport? other)
    {
        return other is not null
            && string.Equals(Domain, other.Domain, StringComparison.Ordinal)
            && string.Equals(OpType, other.OpType, StringComparison.Ordinal);
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        return obj is OnnxOperatorSupport other && Equals(other);
    }

    /// <inheritdoc />
    public override int GetHashCode()
    {
        return HashCode.Combine(Domain, OpType);
    }
}

/// <summary>
/// Provides options for model compatibility validation.
/// </summary>
public sealed class OnnxCompatibilityValidationOptions
{
    /// <summary>
    /// Gets or sets the validation mode.
    /// </summary>
    public OnnxCompatibilityValidationMode Mode { get; init; } = OnnxCompatibilityValidationMode.Structural;

    /// <summary>
    /// Gets or sets the maximum IR version accepted by the target runtime profile.
    /// </summary>
    /// <remarks>
    /// This value is only used when <see cref="Mode"/> is <see cref="OnnxCompatibilityValidationMode.TargetRuntime"/>.
    /// </remarks>
    public long? MaximumTargetIrVersion { get; init; }

    /// <summary>
    /// Gets or sets the opset imports supported by the target runtime profile.
    /// </summary>
    /// <remarks>
    /// When set, nodes are validated against this external domain/version set in addition to the model's own imports.
    /// </remarks>
    public IReadOnlyList<OperationSet>? TargetOpsetImport { get; init; }

    /// <summary>
    /// Gets or sets the operator signatures explicitly supported by the target runtime profile.
    /// </summary>
    /// <remarks>
    /// Leave this unset when opset-level validation is sufficient. When provided, each node must also appear in this list.
    /// </remarks>
    public IReadOnlyCollection<OnnxOperatorSupport>? SupportedOperators { get; init; }
}

/// <summary>
/// Represents one compatibility diagnostic produced by model validation.
/// </summary>
public sealed class OnnxCompatibilityDiagnostic
{
    /// <summary>
    /// Gets the stable diagnostic code.
    /// </summary>
    public required string Code { get; init; }

    /// <summary>
    /// Gets the severity of the diagnostic.
    /// </summary>
    public required OnnxCompatibilityDiagnosticSeverity Severity { get; init; }

    /// <summary>
    /// Gets the diagnostic message.
    /// </summary>
    public required string Message { get; init; }

    /// <summary>
    /// Gets the node name associated with the diagnostic, when applicable.
    /// </summary>
    public string? NodeName { get; init; }

    /// <summary>
    /// Gets the node operator type associated with the diagnostic, when applicable.
    /// </summary>
    public string? OpType { get; init; }

    /// <summary>
    /// Gets the node operator domain associated with the diagnostic, when applicable.
    /// </summary>
    public string? Domain { get; init; }
}

/// <summary>
/// Collects compatibility diagnostics for a model.
/// </summary>
public sealed class OnnxCompatibilityValidationResult
{
    private readonly IReadOnlyList<OnnxCompatibilityDiagnostic> _diagnostics;

    internal OnnxCompatibilityValidationResult(IReadOnlyList<OnnxCompatibilityDiagnostic> diagnostics)
    {
        _diagnostics = diagnostics;
    }

    /// <summary>
    /// Gets all diagnostics emitted during validation.
    /// </summary>
    public IReadOnlyList<OnnxCompatibilityDiagnostic> Diagnostics => _diagnostics;

    /// <summary>
    /// Gets whether validation completed without errors.
    /// </summary>
    public bool IsValid => _diagnostics.All(x => x.Severity != OnnxCompatibilityDiagnosticSeverity.Error);

    /// <summary>
    /// Gets diagnostics with error severity.
    /// </summary>
    public IReadOnlyList<OnnxCompatibilityDiagnostic> Errors => new ReadOnlyCollection<OnnxCompatibilityDiagnostic>(
        _diagnostics.Where(x => x.Severity == OnnxCompatibilityDiagnosticSeverity.Error).ToArray());

    /// <summary>
    /// Gets diagnostics with warning severity.
    /// </summary>
    public IReadOnlyList<OnnxCompatibilityDiagnostic> Warnings => new ReadOnlyCollection<OnnxCompatibilityDiagnostic>(
        _diagnostics.Where(x => x.Severity == OnnxCompatibilityDiagnosticSeverity.Warning).ToArray());
}

internal static class OnnxCompatibilityValidator
{
    private const string MissingOpsetImportCode = "ONNXCOMP001";
    private const string UnsupportedOperatorCode = "ONNXCOMP002";
    private const string InputArityMismatchCode = "ONNXCOMP003";
    private const string OutputArityMismatchCode = "ONNXCOMP004";
    private const string MissingRequiredAttributeCode = "ONNXCOMP005";
    private const string UnknownAttributeCode = "ONNXCOMP006";
    private const string MaximumIrVersionCode = "ONNXCOMP007";
    private const string UnsupportedRuntimeOperatorCode = "ONNXCOMP008";
    private const string IncompleteSchemaHistoryCode = "ONNXCOMP009";

    public static OnnxCompatibilityValidationResult Validate(
        OnnxModel model,
        OnnxCompatibilityValidationOptions? options)
    {
        ArgumentNullException.ThrowIfNull(model);

        options ??= new OnnxCompatibilityValidationOptions();

        var diagnostics = new List<OnnxCompatibilityDiagnostic>();
        var schemaRepository = OnnxOperatorSchemaRepository.Instance;

        var modelOpsets = model.OpsetImport.ToDictionary(x => x.Domain, x => x.Version, StringComparer.Ordinal);
        var targetOpsets = options.TargetOpsetImport?.ToDictionary(x => x.Domain, x => x.Version, StringComparer.Ordinal);
        var supportedOperators = options.SupportedOperators is null
            ? null
            : new HashSet<OnnxOperatorSupport>(options.SupportedOperators);

        if (options.Mode == OnnxCompatibilityValidationMode.TargetRuntime
            && options.MaximumTargetIrVersion is long maximumTargetIrVersion
            && model.IrVersion > maximumTargetIrVersion)
        {
            diagnostics.Add(new OnnxCompatibilityDiagnostic
            {
                Code = MaximumIrVersionCode,
                Severity = OnnxCompatibilityDiagnosticSeverity.Error,
                Message = $"Model IR version {model.IrVersion} exceeds target runtime maximum {maximumTargetIrVersion}.",
            });
        }

        foreach (var node in model.Graph.Nodes)
        {
            ValidateNodeAgainstProfile(
                node,
                modelOpsets,
                schemaRepository,
                diagnostics,
                contextLabel: "model");

            if (options.Mode != OnnxCompatibilityValidationMode.TargetRuntime)
            {
                continue;
            }

            if (targetOpsets is not null)
            {
                ValidateNodeAgainstProfile(
                    node,
                    targetOpsets,
                    schemaRepository,
                    diagnostics,
                    contextLabel: "target runtime");
            }

            if (supportedOperators is not null
                && !supportedOperators.Contains(new OnnxOperatorSupport
                {
                    Domain = node.Domain,
                    OpType = node.OpType,
                }))
            {
                diagnostics.Add(new OnnxCompatibilityDiagnostic
                {
                    Code = UnsupportedRuntimeOperatorCode,
                    Severity = OnnxCompatibilityDiagnosticSeverity.Error,
                    Message = $"Target runtime profile does not list operator '{FormatOperator(node.Domain, node.OpType)}' as supported.",
                    NodeName = node.Name,
                    Domain = node.Domain,
                    OpType = node.OpType,
                });
            }
        }

        return new OnnxCompatibilityValidationResult(new ReadOnlyCollection<OnnxCompatibilityDiagnostic>(diagnostics));
    }

    private static void ValidateNodeAgainstProfile(
        OnnxNode node,
        IReadOnlyDictionary<string, long> opsets,
        OnnxOperatorSchemaRepository schemaRepository,
        List<OnnxCompatibilityDiagnostic> diagnostics,
        string contextLabel)
    {
        if (!opsets.TryGetValue(node.Domain, out var opset))
        {
            diagnostics.Add(new OnnxCompatibilityDiagnostic
            {
                Code = MissingOpsetImportCode,
                Severity = OnnxCompatibilityDiagnosticSeverity.Error,
                Message = $"Node '{node.Name}' uses operator '{FormatOperator(node.Domain, node.OpType)}' but the {contextLabel} has no imported opset for domain '{FormatDomain(node.Domain)}'.",
                NodeName = node.Name,
                Domain = node.Domain,
                OpType = node.OpType,
            });
            return;
        }

        var resolution = schemaRepository.Resolve(node.Domain, node.OpType, opset);

        if (resolution.Status == OnnxSchemaResolutionStatus.Unsupported)
        {
            var earliestKnownVersion = resolution.EarliestKnownVersion?.ToString() ?? "?";
            diagnostics.Add(new OnnxCompatibilityDiagnostic
            {
                Code = UnsupportedOperatorCode,
                Severity = OnnxCompatibilityDiagnosticSeverity.Error,
                Message = $"Operator '{FormatOperator(node.Domain, node.OpType)}' is not available in {contextLabel} opset {opset}. Earliest bundled version is {earliestKnownVersion}.",
                NodeName = node.Name,
                Domain = node.Domain,
                OpType = node.OpType,
            });
            return;
        }

        if (resolution.Status == OnnxSchemaResolutionStatus.Unknown)
        {
            diagnostics.Add(new OnnxCompatibilityDiagnostic
            {
                Code = UnsupportedOperatorCode,
                Severity = OnnxCompatibilityDiagnosticSeverity.Error,
                Message = $"No bundled schema metadata was found for operator '{FormatOperator(node.Domain, node.OpType)}'.",
                NodeName = node.Name,
                Domain = node.Domain,
                OpType = node.OpType,
            });
            return;
        }

        if (resolution.Status == OnnxSchemaResolutionStatus.IncompleteHistory)
        {
            diagnostics.Add(new OnnxCompatibilityDiagnostic
            {
                Code = IncompleteSchemaHistoryCode,
                Severity = OnnxCompatibilityDiagnosticSeverity.Warning,
                Message = $"Bundled schema history for operator '{FormatOperator(node.Domain, node.OpType)}' in domain '{FormatDomain(node.Domain)}' starts at version {resolution.EarliestKnownVersion}. Compatibility with {contextLabel} opset {opset} could not be verified exactly.",
                NodeName = node.Name,
                Domain = node.Domain,
                OpType = node.OpType,
            });
        }

        if (resolution.Schema is null)
        {
            return;
        }

        ValidateArity(
            node,
            resolution.Schema.MinimumInputs,
            resolution.Schema.MaximumInputs,
            actualCount: node.Inputs.Count,
            code: InputArityMismatchCode,
            kind: "input",
            diagnostics);

        ValidateArity(
            node,
            resolution.Schema.MinimumOutputs,
            resolution.Schema.MaximumOutputs,
            actualCount: node.Outputs.Count,
            code: OutputArityMismatchCode,
            kind: "output",
            diagnostics);

        var actualAttributes = new HashSet<string>(node.Attributes.Select(x => x.Name), StringComparer.Ordinal);

        foreach (var requiredAttribute in resolution.Schema.RequiredAttributes)
        {
            if (actualAttributes.Contains(requiredAttribute))
            {
                continue;
            }

            diagnostics.Add(new OnnxCompatibilityDiagnostic
            {
                Code = MissingRequiredAttributeCode,
                Severity = OnnxCompatibilityDiagnosticSeverity.Error,
                Message = $"Node '{node.Name}' is missing required attribute '{requiredAttribute}' for operator '{FormatOperator(node.Domain, node.OpType)}'.",
                NodeName = node.Name,
                Domain = node.Domain,
                OpType = node.OpType,
            });
        }

        foreach (var attribute in actualAttributes)
        {
            if (resolution.Schema.KnownAttributes.Contains(attribute))
            {
                continue;
            }

            diagnostics.Add(new OnnxCompatibilityDiagnostic
            {
                Code = UnknownAttributeCode,
                Severity = OnnxCompatibilityDiagnosticSeverity.Error,
                Message = $"Node '{node.Name}' carries attribute '{attribute}', which is not declared by bundled schema metadata for operator '{FormatOperator(node.Domain, node.OpType)}'.",
                NodeName = node.Name,
                Domain = node.Domain,
                OpType = node.OpType,
            });
        }
    }

    private static void ValidateArity(
        OnnxNode node,
        int minimumCount,
        int? maximumCount,
        int actualCount,
        string code,
        string kind,
        List<OnnxCompatibilityDiagnostic> diagnostics)
    {
        if (actualCount < minimumCount || (maximumCount is not null && actualCount > maximumCount.Value))
        {
            var expected = maximumCount is null
                ? $"{minimumCount}+"
                : minimumCount == maximumCount.Value
                    ? minimumCount.ToString()
                    : $"{minimumCount}-{maximumCount.Value}";

            diagnostics.Add(new OnnxCompatibilityDiagnostic
            {
                Code = code,
                Severity = OnnxCompatibilityDiagnosticSeverity.Error,
                Message = $"Node '{node.Name}' supplies {actualCount} {kind}(s), but operator '{FormatOperator(node.Domain, node.OpType)}' expects {expected}.",
                NodeName = node.Name,
                Domain = node.Domain,
                OpType = node.OpType,
            });
        }
    }

    private static string FormatDomain(string domain)
    {
        return string.IsNullOrWhiteSpace(domain) ? "ai.onnx" : domain;
    }

    private static string FormatOperator(string domain, string opType)
    {
        return $"{FormatDomain(domain)}::{opType}";
    }
}

internal enum OnnxSchemaResolutionStatus
{
    Exact = 0,
    IncompleteHistory = 1,
    Unsupported = 2,
    Unknown = 3,
}

internal sealed class OnnxSchemaResolution
{
    public required OnnxSchemaResolutionStatus Status { get; init; }

    public OnnxBundledOperatorSchema? Schema { get; init; }

    public int? EarliestKnownVersion { get; init; }
}

internal readonly record struct OperatorKey(string Domain, string Name);

internal sealed class OnnxOperatorSchemaRepository
{
    private static readonly Lazy<OnnxOperatorSchemaRepository> _instance = new(Create);

    private readonly Dictionary<OperatorKey, SortedDictionary<int, OnnxBundledOperatorSchema>> _historicalSchemas;
    private readonly Dictionary<OperatorKey, OnnxBundledOperatorSchema> _latestSchemas;
    private readonly Dictionary<OperatorKey, int> _currentStructuralCompatibility;

    private OnnxOperatorSchemaRepository(
        Dictionary<OperatorKey, SortedDictionary<int, OnnxBundledOperatorSchema>> historicalSchemas,
        Dictionary<OperatorKey, OnnxBundledOperatorSchema> latestSchemas,
        Dictionary<OperatorKey, int> currentStructuralCompatibility)
    {
        _historicalSchemas = historicalSchemas;
        _latestSchemas = latestSchemas;
        _currentStructuralCompatibility = currentStructuralCompatibility;
    }

    public static OnnxOperatorSchemaRepository Instance => _instance.Value;

    public OnnxSchemaResolution Resolve(string domain, string opType, long opset)
    {
        var key = new OperatorKey(domain, opType);

        if (_historicalSchemas.TryGetValue(key, out var versionMap))
        {
            OnnxBundledOperatorSchema? selectedSchema = null;

            foreach (var entry in versionMap)
            {
                if (entry.Key > opset)
                {
                    break;
                }

                selectedSchema = entry.Value;
            }

            if (selectedSchema is not null)
            {
                return new OnnxSchemaResolution
                {
                    Status = OnnxSchemaResolutionStatus.Exact,
                    Schema = selectedSchema,
                    EarliestKnownVersion = versionMap.Keys.First(),
                };
            }

            return new OnnxSchemaResolution
            {
                Status = OnnxSchemaResolutionStatus.Unsupported,
                EarliestKnownVersion = versionMap.Keys.First(),
            };
        }

        if (_latestSchemas.TryGetValue(key, out var latestSchema))
        {
            if (opset < latestSchema.SinceVersion)
            {
                return new OnnxSchemaResolution
                {
                    Status = OnnxSchemaResolutionStatus.IncompleteHistory,
                    EarliestKnownVersion = latestSchema.SinceVersion,
                };
            }

            return new OnnxSchemaResolution
            {
                Status = OnnxSchemaResolutionStatus.Exact,
                Schema = latestSchema,
                EarliestKnownVersion = latestSchema.SinceVersion,
            };
        }

        return new OnnxSchemaResolution
        {
            Status = OnnxSchemaResolutionStatus.Unknown,
        };
    }

    public bool IsCurrentStructureCompatible(string domain, string opType, long opset)
    {
        return TryGetCurrentStructuralCompatibilityMinimumVersion(domain, opType, out var minimumCurrentStructureSinceVersion)
            && opset >= minimumCurrentStructureSinceVersion;
    }

    public bool TryGetCurrentStructuralCompatibilityMinimumVersion(string domain, string opType, out int minimumCurrentStructureSinceVersion)
    {
        var key = new OperatorKey(domain, opType);

        if (_currentStructuralCompatibility.TryGetValue(key, out minimumCurrentStructureSinceVersion))
        {
            return true;
        }

        if (_latestSchemas.TryGetValue(key, out var latestSchema))
        {
            minimumCurrentStructureSinceVersion = latestSchema.SinceVersion;
            return true;
        }

        minimumCurrentStructureSinceVersion = default;
        return false;
    }

    private static OnnxOperatorSchemaRepository Create()
    {
        var historicalSchemas = new Dictionary<OperatorKey, SortedDictionary<int, OnnxBundledOperatorSchema>>();
        var latestSchemas = new Dictionary<OperatorKey, OnnxBundledOperatorSchema>();
        var currentStructuralCompatibility = new Dictionary<OperatorKey, int>();

        OnnxGeneratedCompatibilityMetadata.PopulateHistoricalSchemas(historicalSchemas);
        OnnxGeneratedCompatibilityMetadata.PopulateLatestSchemas(latestSchemas);
        OnnxGeneratedCompatibilityMetadata.PopulateCurrentStructuralCompatibility(currentStructuralCompatibility);

        return new OnnxOperatorSchemaRepository(historicalSchemas, latestSchemas, currentStructuralCompatibility);
    }
}

internal static partial class OnnxGeneratedCompatibilityMetadata
{
    internal static partial void PopulateHistoricalSchemas(
        Dictionary<OperatorKey, SortedDictionary<int, OnnxBundledOperatorSchema>> historicalSchemas);

    internal static partial void PopulateLatestSchemas(
        Dictionary<OperatorKey, OnnxBundledOperatorSchema> latestSchemas);

    internal static partial void PopulateCurrentStructuralCompatibility(
        Dictionary<OperatorKey, int> currentStructuralCompatibility);
}

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
