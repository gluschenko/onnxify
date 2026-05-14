using System.Collections.ObjectModel;

namespace Onnxify;

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
        _diagnostics
            .Where(x => x.Severity == OnnxCompatibilityDiagnosticSeverity.Error)
            .ToArray()
    );

    /// <summary>
    /// Gets diagnostics with warning severity.
    /// </summary>
    public IReadOnlyList<OnnxCompatibilityDiagnostic> Warnings => new ReadOnlyCollection<OnnxCompatibilityDiagnostic>(
        _diagnostics
            .Where(x => x.Severity == OnnxCompatibilityDiagnosticSeverity.Warning)
            .ToArray()
    );
}
