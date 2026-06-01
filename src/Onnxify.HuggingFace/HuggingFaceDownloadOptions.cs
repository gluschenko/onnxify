namespace Onnxify.HuggingFace;

/// <summary>
/// Configures how a Hugging Face model repository is downloaded.
/// </summary>
public sealed class HuggingFaceDownloadOptions
{
    /// <summary>
    /// Gets the Hugging Face repository revision to download.
    /// </summary>
    /// <remarks>
    /// This can be a branch name, tag, or commit SHA. The default value is <c>main</c>.
    /// </remarks>
    public string Revision { get; init; } = "main";

    /// <summary>
    /// Gets the optional Hugging Face access token used for private or gated repositories.
    /// </summary>
    public string? AccessToken { get; init; }

    /// <summary>
    /// Gets an optional predicate that chooses which repository-relative paths should be downloaded.
    /// </summary>
    /// <remarks>
    /// Paths are normalized to use forward slashes. When this value is <see langword="null"/>, every repository file is included before <see cref="ExcludePath"/> is applied.
    /// </remarks>
    public Func<string, bool>? IncludePath { get; init; }

    /// <summary>
    /// Gets an optional predicate that excludes repository-relative paths from the download.
    /// </summary>
    /// <remarks>
    /// Paths are normalized to use forward slashes. This predicate is evaluated after <see cref="IncludePath"/>.
    /// </remarks>
    public Func<string, bool>? ExcludePath { get; init; }

    /// <summary>
    /// Gets an optional callback invoked as file bytes are downloaded and when each file completes.
    /// </summary>
    public Action<HuggingFaceDownloadProgress>? ProgressCallback { get; init; }

    /// <summary>
    /// Gets a value indicating whether existing local files should be replaced.
    /// </summary>
    public bool Overwrite { get; init; } = true;

    /// <summary>
    /// Gets the buffer size, in bytes, used when streaming file content to disk.
    /// </summary>
    public int BufferSize { get; init; } = 1024 * 128;
}
