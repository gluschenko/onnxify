namespace Onnxify.HuggingFace;

/// <summary>
/// Describes one repository file considered by a Hugging Face download operation.
/// </summary>
public sealed class HuggingFaceDownloadedFile
{
    /// <summary>
    /// Gets the repository-relative Hugging Face path.
    /// </summary>
    public required string RepositoryPath { get; init; }

    /// <summary>
    /// Gets the absolute local path where the file is stored or would be stored.
    /// </summary>
    public required string LocalPath { get; init; }

    /// <summary>
    /// Gets the file size reported by Hugging Face, when available.
    /// </summary>
    public required long? Size { get; init; }

    /// <summary>
    /// Gets a value indicating whether the file content was downloaded during this operation.
    /// </summary>
    /// <remarks>
    /// This value is <see langword="false"/> when a local file already exists and overwrite is disabled.
    /// </remarks>
    public required bool Downloaded { get; init; }
}
