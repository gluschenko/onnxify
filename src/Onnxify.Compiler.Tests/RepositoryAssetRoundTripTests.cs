using Onnxify;
using Onnxify.Compiler;

namespace Onnxify.Compiler.Tests;

public sealed class RepositoryAssetRoundTripTests
{
    public static IEnumerable<object[]> RepositoryModels =>
    [
        ["mobilenet_v2_1.4_224.onnx"],
        ["yolo26s.onnx"],
    ];

    [Theory]
    [MemberData(nameof(RepositoryModels))]
    public void RepositoryAssetRoundTripsThroughCompiler(string fileName)
    {
        var sourcePath = Path.Combine(AppContext.BaseDirectory, "Assets", fileName);
        Assert.True(File.Exists(sourcePath), $"Missing compiler round-trip fixture '{sourcePath}'.");

        // Source: src/Onnxify.Tests/OnnxModelTests.cs
        // (FromFile_WithMobileNetDuplicatedLogitsValueInfo_LoadsGraph and
        // FromFile_WithYolo26sNoneDimension_LoadsExportsAndPrintsGraph).
        // No direct ONNXScript converter test exists for these repository model fixtures;
        // the compiler boundary is validated by core model reload and ONNX Runtime.
        var original = OnnxModel.FromFile(
            sourcePath,
            new OnnxModelBaseOptions
            {
                NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.PreserveUntyped,
            });

        var imported = Compiler.CreateTreeFromOnnx(sourcePath);

        Assert.True(
            imported.IsSuccess,
            FormatDiagnostics(imported.Diagnostics));
        Assert.NotNull(imported.Value);

        var emitted = Compiler.GenerateOnnx(imported.Value!);

        Assert.True(
            emitted.IsSuccess,
            FormatDiagnostics(emitted.Diagnostics));
        Assert.NotNull(emitted.Value);

        var roundTrippedPath = Path.Combine(
            Path.GetTempPath(),
            $"onnxify-compiler-asset-{Guid.NewGuid():N}.onnx");

        try
        {
            emitted.Value!.Save(roundTrippedPath, overwrite: true);
            var roundTripped = OnnxModel.FromFile(
                roundTrippedPath,
                new OnnxModelBaseOptions
                {
                    NodeTypeResolutionStrategy = NodeTypeResolutionStrategy.PreserveUntyped,
                });

            Assert.Equal(original.Graph.Name, roundTripped.Graph.Name);
            Assert.Equal(
                original.Graph.Inputs.Select(static value => value.Name),
                roundTripped.Graph.Inputs.Select(static value => value.Name));
            Assert.Equal(
                original.Graph.Outputs.Select(static value => value.Name),
                roundTripped.Graph.Outputs.Select(static value => value.Name));
            Assert.Equal(original.Graph.Nodes.Count, roundTripped.Graph.Nodes.Count);
            Assert.Equal(
                original.Graph.Nodes.Select(NodeSignature),
                roundTripped.Graph.Nodes.Select(NodeSignature));
            Assert.Equal(
                original.Graph.Initializers.Select(static initializer => initializer.Name),
                roundTripped.Graph.Initializers.Select(static initializer => initializer.Name));
            Assert.Equal(
                original.Graph.SparseInitializers.Select(static initializer => initializer.Name),
                roundTripped.Graph.SparseInitializers.Select(static initializer => initializer.Name));

            using var session = new global::Microsoft.ML.OnnxRuntime.InferenceSession(roundTrippedPath);
            Assert.NotEmpty(session.InputMetadata);
            Assert.NotEmpty(session.OutputMetadata);
        }
        finally
        {
            if (File.Exists(roundTrippedPath))
            {
                File.Delete(roundTrippedPath);
            }
        }
    }

    private static string NodeSignature(OnnxNode node)
    {
        var attributes = node.Attributes
            .OrderBy(static attribute => attribute.Name, StringComparer.Ordinal)
            .Select(static attribute => attribute.Name);

        return string.Join(
            "|",
            node.Name,
            node.Domain,
            node.OpType,
            string.Join(",", node.Inputs.Select(static input => input.Name)),
            string.Join(",", node.Outputs.Select(static output => output.Name)),
            string.Join(",", attributes));
    }

    private static string FormatDiagnostics(IReadOnlyList<CompilerDiagnostic> diagnostics)
    {
        return string.Join(
            Environment.NewLine,
            diagnostics.Select(static diagnostic =>
                $"{diagnostic.Severity} {diagnostic.Code}: {diagnostic.Message}"));
    }
}
