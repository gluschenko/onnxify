using System;
using System.IO;
using System.Reflection;
using Microsoft.CodeAnalysis.Text;

namespace Onnxify.ModelGenerator.Services;

internal static class GeneratedSupportSourceLoader
{
    private const string RESOURCE_PREFIX = "Onnxify.ModelGenerator.Templates.";

    internal static SourceText Load(
        string fileName,
        string namespaceName
    )
    {
        var assembly = typeof(GeneratedSupportSourceLoader).Assembly;
        var resourceName = RESOURCE_PREFIX + fileName;
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new InvalidOperationException($"Generated support source resource '{resourceName}' was not found.");
        using var reader = new StreamReader(stream);
        return SourceText.From(
            text: reader.ReadToEnd().Replace("{{OnnxifyModelNamespace}}", namespaceName),
            encoding: System.Text.Encoding.UTF8
        );
    }
}
