namespace Onnxify.HuggingFace;

/// <summary>
/// Describes the result of downloading files from a Hugging Face repository.
/// </summary>
public sealed class HuggingFaceDownloadResult
{
    /// <summary>
    /// Gets the Hugging Face repository identifier.
    /// </summary>
    public required string RepositoryId { get; init; }

    /// <summary>
    /// Gets the repository revision used for the download.
    /// </summary>
    public required string Revision { get; init; }

    /// <summary>
    /// Gets the absolute output directory path.
    /// </summary>
    public required string OutputDirectoryPath { get; init; }

    /// <summary>
    /// Gets the files considered by the filtered download operation.
    /// </summary>
    public required IReadOnlyList<HuggingFaceDownloadedFile> Files { get; init; }

    /// <summary>
    /// Gets the number of files whose content was downloaded during this operation.
    /// </summary>
    public long DownloadedFileCount => Files.Count(static x => x.Downloaded);
}
