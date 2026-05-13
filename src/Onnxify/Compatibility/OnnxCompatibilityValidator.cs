using System.Collections.ObjectModel;

namespace Onnxify;

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
        OnnxCompatibilityValidationOptions? options
    )
    {
        ArgumentNullException.ThrowIfNull(model);

        options ??= new OnnxCompatibilityValidationOptions();

        var diagnostics = new List<OnnxCompatibilityDiagnostic>();
        var schemaRepository = OnnxOperatorSchemaResolver.Instance;

        var modelOpsets = model.OpsetImport.ToDictionary(x => x.Domain, x => x.Version, StringComparer.Ordinal);
        var targetOpsets = options.TargetOpsetImport?.ToDictionary(x => x.Domain, x => x.Version, StringComparer.Ordinal);
        var supportedOperators = options.SupportedOperators is not null
            ? new HashSet<OnnxOperatorSupport>(options.SupportedOperators)
            : null;

        if (
            options.Mode == OnnxCompatibilityValidationMode.TargetRuntime && 
            options.MaximumTargetIrVersion is long maximumTargetIrVersion && 
            model.IrVersion > maximumTargetIrVersion
        )
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
                contextLabel: "model"
            );

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
                    contextLabel: "target runtime"
                );
            }

            if (
                supportedOperators is not null && 
                !supportedOperators.Contains(new OnnxOperatorSupport
                {
                    Domain = node.Domain,
                    OpType = node.OpType,
                })
            )
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
        OnnxOperatorSchemaResolver schemaRepository,
        List<OnnxCompatibilityDiagnostic> diagnostics,
        string contextLabel
    )
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
            diagnostics
        );

        ValidateArity(
            node,
            resolution.Schema.MinimumOutputs,
            resolution.Schema.MaximumOutputs,
            actualCount: node.Outputs.Count,
            code: OutputArityMismatchCode,
            kind: "output",
            diagnostics
        );

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
        List<OnnxCompatibilityDiagnostic> diagnostics
    )
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
