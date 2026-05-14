namespace Onnxify;

internal static partial class OnnxGeneratedCompatibilityMetadata
{
    internal static partial void PopulateHistoricalSchemas(
        Dictionary<OperatorKey, SortedDictionary<int, OnnxBundledOperatorSchema>> historicalSchemas
    );

    internal static partial void PopulateLatestSchemas(
        Dictionary<OperatorKey, OnnxBundledOperatorSchema> latestSchemas
    );

    internal static partial void PopulateCurrentStructuralCompatibility(
        Dictionary<OperatorKey, int> currentStructuralCompatibility
    );
}
