namespace Onnxify;

internal sealed class OnnxOperatorSchemaResolver
{
    private static readonly Lazy<OnnxOperatorSchemaResolver> _instance = new(Create);

    private readonly Dictionary<OperatorKey, SortedDictionary<int, OnnxBundledOperatorSchema>> _historicalSchemas;
    private readonly Dictionary<OperatorKey, OnnxBundledOperatorSchema> _latestSchemas;
    private readonly Dictionary<OperatorKey, int> _currentStructuralCompatibility;

    private OnnxOperatorSchemaResolver(
        Dictionary<OperatorKey, SortedDictionary<int, OnnxBundledOperatorSchema>> historicalSchemas,
        Dictionary<OperatorKey, OnnxBundledOperatorSchema> latestSchemas,
        Dictionary<OperatorKey, int> currentStructuralCompatibility
    )
    {
        _historicalSchemas = historicalSchemas;
        _latestSchemas = latestSchemas;
        _currentStructuralCompatibility = currentStructuralCompatibility;
    }

    public static OnnxOperatorSchemaResolver Instance => _instance.Value;

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
        if (TryGetCurrentStructuralCompatibilityMinimumVersion(domain, opType, out var minimumCurrentStructureSinceVersion))
        {
            return opset >= minimumCurrentStructureSinceVersion;
        }

        return false;
    }

    public bool TryGetCurrentStructuralCompatibilityMinimumVersion(
        string domain,
        string opType,
        out int minimumCurrentStructureSinceVersion
    )
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

    private static OnnxOperatorSchemaResolver Create()
    {
        var historicalSchemas = new Dictionary<OperatorKey, SortedDictionary<int, OnnxBundledOperatorSchema>>();
        var latestSchemas = new Dictionary<OperatorKey, OnnxBundledOperatorSchema>();
        var currentStructuralCompatibility = new Dictionary<OperatorKey, int>();

        OnnxGeneratedCompatibilityMetadata.PopulateHistoricalSchemas(historicalSchemas);
        OnnxGeneratedCompatibilityMetadata.PopulateLatestSchemas(latestSchemas);
        OnnxGeneratedCompatibilityMetadata.PopulateCurrentStructuralCompatibility(currentStructuralCompatibility);

        return new OnnxOperatorSchemaResolver(historicalSchemas, latestSchemas, currentStructuralCompatibility);
    }
}
