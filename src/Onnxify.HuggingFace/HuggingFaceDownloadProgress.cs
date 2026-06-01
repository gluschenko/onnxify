namespace Onnxify.HuggingFace;

/// <summary>
/// Describes progress for a Hugging Face repository file download.
/// </summary>
public sealed class HuggingFaceDownloadProgress
{
    /// <summary>
    /// Gets the Hugging Face repository identifier.
    /// </summary>
    public required string RepositoryId { get; init; }

    /// <summary>
    /// Gets the repository-relative file path being downloaded.
    /// </summary>
    public required string RepositoryPath { get; init; }

    /// <summary>
    /// Gets the one-based index of the current file within the filtered download set.
    /// </summary>
    public required int FileIndex { get; init; }

    /// <summary>
    /// Gets the number of files in the filtered download set.
    /// </summary>
    public required int FileCount { get; init; }

    /// <summary>
    /// Gets the total file size in bytes when Hugging Face reports it.
    /// </summary>
    public required long? TotalBytes { get; init; }

    /// <summary>
    /// Gets the number of bytes downloaded for the current file.
    /// </summary>
    public required long DownloadedBytes { get; init; }

    /// <summary>
    /// Gets a value indicating whether the current file is complete.
    /// </summary>
    public required bool Completed { get; init; }
}
