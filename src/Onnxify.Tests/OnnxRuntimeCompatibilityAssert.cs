using Microsoft.ML.OnnxRuntime;

namespace Onnxify.Tests;

internal static class OnnxRuntimeCompatibilityAssert
{
    public static void CanCreateSession(
        OnnxModel model,
        string scenario
    )
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(scenario);

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
        try
        {
            model.Save(path, overwrite: true);
            using var _ = CreateSession(path, model, scenario);
        }
        finally
        {
            DeleteIfExists(path);
        }
    }

    public static InferenceSession CreateSession(
        string modelPath,
        OnnxModel model,
        string scenario,
        SessionOptions? sessionOptions = null
    )
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(scenario);

        try
        {
            return sessionOptions is null
                ? new InferenceSession(modelPath)
                : new InferenceSession(modelPath, sessionOptions);
        }
        catch (Exception exception) when (
            exception is OnnxRuntimeException
                or InvalidOperationException
                or NotSupportedException
        )
        {
            throw new InvalidOperationException(
                $"ONNX Runtime could not create an inference session for '{scenario}'. "
                + $"Exported operators: {DescribeOperators(model)}",
                exception
            );
        }
    }

    private static string DescribeOperators(OnnxModel model)
    {
        var operators = model.Graph.Nodes
            .GroupBy(static node => string.IsNullOrEmpty(node.Domain)
                ? node.OpType
                : $"{node.Domain}::{node.OpType}")
            .OrderBy(static group => group.Key, StringComparer.Ordinal)
            .Select(static group => $"{group.Key} x{group.Count()}")
            .ToArray();

        return operators.Length == 0
            ? "<none>"
            : string.Join(", ", operators);
    }

    private static void DeleteIfExists(string path)
    {
        if (File.Exists(path))
        {
            File.Delete(path);
        }
    }
}
