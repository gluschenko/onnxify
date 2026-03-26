namespace Onnxify.ProjectGenerator;

public sealed class ProjectGenerationResult
{
    public required string OutputDirectoryPath { get; init; }
    public required string ProgramFilePath { get; init; }
    public string? ProjectFilePath { get; init; }
    public required IReadOnlyList<string> TensorFilePaths { get; init; }
    public required IReadOnlyList<string> Warnings { get; init; }
}
