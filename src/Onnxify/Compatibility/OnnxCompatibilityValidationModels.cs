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
