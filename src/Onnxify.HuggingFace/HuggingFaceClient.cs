using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Onnxify.HuggingFace;

/// <summary>
/// Downloads model repository files from Hugging Face.
/// </summary>
public sealed class HuggingFaceClient
{
    private static readonly Uri DefaultEndpoint = new("https://huggingface.co/");
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    private readonly HttpClient _httpClient;

    /// <summary>
    /// Initializes a new instance of the <see cref="HuggingFaceClient"/> class using the public Hugging Face endpoint.
    /// </summary>
    public HuggingFaceClient()
        : this(new HttpClient { BaseAddress = DefaultEndpoint })
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="HuggingFaceClient"/> class using the specified HTTP client.
    /// </summary>
    /// <param name="httpClient">The HTTP client used to call Hugging Face APIs and download file content.</param>
    /// <remarks>
    /// If <paramref name="httpClient"/> does not have a base address, the public Hugging Face endpoint is used.
    /// </remarks>
    public HuggingFaceClient(HttpClient httpClient)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));

        if (_httpClient.BaseAddress is null)
        {
            _httpClient.BaseAddress = DefaultEndpoint;
        }
    }

    /// <summary>
    /// Downloads selected files from a Hugging Face model repository into a local directory.
    /// </summary>
    /// <param name="repositoryId">The Hugging Face repository identifier, such as <c>onnx-community/gemma-4-E2B-it-ONNX</c>.</param>
    /// <param name="targetDirectoryPath">The directory where repository files should be written.</param>
    /// <param name="options">Optional download settings such as revision, token, path filters, and overwrite behavior.</param>
    /// <param name="progress">An optional progress reporter invoked as file bytes are downloaded and when each file completes.</param>
    /// <param name="cancellationToken">A token used to cancel the operation.</param>
    /// <returns>A result describing the local output directory and the repository files considered by the filtered download.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="repositoryId"/> or <paramref name="targetDirectoryPath"/> is empty.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <see cref="HuggingFaceDownloadOptions.BufferSize"/> is less than or equal to zero.
    /// </exception>
    /// <exception cref="HttpRequestException">
    /// Thrown when Hugging Face returns a non-success HTTP response.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the Hugging Face response is malformed or a repository path would escape the target directory.
    /// </exception>
    /// <remarks>
    /// Repository paths are preserved relative to <paramref name="targetDirectoryPath"/>.
    /// Use <see cref="HuggingFaceDownloadOptions.IncludePath"/> and <see cref="HuggingFaceDownloadOptions.ExcludePath"/> to avoid downloading every file from large multi-variant repositories.
    /// </remarks>
    public async Task<HuggingFaceDownloadResult> DownloadRepositoryAsync(
        string repositoryId,
        string targetDirectoryPath,
        HuggingFaceDownloadOptions? options = null,
        IProgress<HuggingFaceDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(repositoryId))
        {
            throw new ArgumentException("Repository id is required.", nameof(repositoryId));
        }

        if (string.IsNullOrWhiteSpace(targetDirectoryPath))
        {
            throw new ArgumentException("Target directory path is required.", nameof(targetDirectoryPath));
        }

        options ??= new HuggingFaceDownloadOptions();

        if (options.BufferSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(HuggingFaceDownloadOptions.BufferSize), "Buffer size must be greater than zero.");
        }

        var outputDirectoryPath = Path.GetFullPath(targetDirectoryPath);
        Directory.CreateDirectory(outputDirectoryPath);

        var files = await GetRepositoryFilesAsync(repositoryId, options, cancellationToken);
        var downloadedFiles = new List<HuggingFaceDownloadedFile>(files.Count);

        for (var index = 0; index < files.Count; index++)
        {
            var file = files[index];
            var destinationPath = ResolveDestinationPath(outputDirectoryPath, file.Path);

            if (File.Exists(destinationPath) && !options.Overwrite)
            {
                downloadedFiles.Add(
                    new HuggingFaceDownloadedFile
                    {
                        RepositoryPath = file.Path,
                        LocalPath = destinationPath,
                        Size = file.Size,
                        Downloaded = false,
                    }
                );
                ReportProgress(
                    new HuggingFaceDownloadProgress
                    {
                        RepositoryId = repositoryId,
                        RepositoryPath = file.Path,
                        FileIndex = index + 1,
                        FileCount = files.Count,
                        TotalBytes = file.Size,
                        DownloadedBytes = 0,
                        Completed = true,
                    },
                    options,
                    progress
                );
                continue;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);

            var downloadUri = BuildResolveUri(repositoryId, options.Revision, file.Path);
            var downloadedBytes = await DownloadFileAsync(
                downloadUri,
                destinationPath,
                options,
                repositoryId,
                file,
                index,
                files.Count,
                progress,
                cancellationToken);

            downloadedFiles.Add(
                new HuggingFaceDownloadedFile
                {
                    RepositoryPath = file.Path,
                    LocalPath = destinationPath,
                    Size = file.Size,
                    Downloaded = true,
                }
            );
            ReportProgress(
                new HuggingFaceDownloadProgress
                {
                    RepositoryId = repositoryId,
                    RepositoryPath = file.Path,
                    FileIndex = index + 1,
                    FileCount = files.Count,
                    TotalBytes = file.Size,
                    DownloadedBytes = downloadedBytes,
                    Completed = true,
                },
                options,
                progress
            );
        }

        return new HuggingFaceDownloadResult
        {
            RepositoryId = repositoryId,
            Revision = options.Revision,
            OutputDirectoryPath = outputDirectoryPath,
            Files = downloadedFiles,
        };
    }

    private async Task<List<RepositoryFile>> GetRepositoryFilesAsync(
        string repositoryId,
        HuggingFaceDownloadOptions options,
        CancellationToken cancellationToken
    )
    {
        using var request = CreateRequest(HttpMethod.Get, BuildModelInfoUri(repositoryId, options.Revision), options);
        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

        await EnsureSuccessAsync(response, cancellationToken);

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        var modelInfo = await JsonSerializer.DeserializeAsync<ModelInfoResponse>(stream, JsonOptions, cancellationToken);

        if (modelInfo?.Siblings is null)
        {
            throw new InvalidOperationException($"Hugging Face response for '{repositoryId}' did not contain a file list.");
        }

        return modelInfo.Siblings
            .Where(static x => !string.IsNullOrWhiteSpace(x.RFileName))
            .Select(static x => new RepositoryFile
            {
                Path = NormalizeRepositoryPath(x.RFileName!),
                Size = x.Size,
            })
            .Where(x => ShouldDownloadPath(x.Path, options))
            .OrderBy(static x => x.Path, StringComparer.Ordinal)
            .ToList();
    }

    private static bool ShouldDownloadPath(string path, HuggingFaceDownloadOptions options)
    {
        if (options.IncludePath is not null && !options.IncludePath(path))
        {
            return false;
        }

        return options.ExcludePath is null || !options.ExcludePath(path);
    }

    private static void ReportProgress(
        HuggingFaceDownloadProgress value,
        HuggingFaceDownloadOptions options,
        IProgress<HuggingFaceDownloadProgress>? progress
    )
    {
        options.ProgressCallback?.Invoke(value);
        progress?.Report(value);
    }

    private async Task<long> DownloadFileAsync(
        Uri requestUri,
        string destinationPath,
        HuggingFaceDownloadOptions options,
        string repositoryId,
        RepositoryFile file,
        int fileIndex,
        int fileCount,
        IProgress<HuggingFaceDownloadProgress>? progress,
        CancellationToken cancellationToken
    )
    {
        using var request = CreateRequest(HttpMethod.Get, requestUri, options);
        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

        await EnsureSuccessAsync(response, cancellationToken);

        var tempPath = destinationPath + "." + Guid.NewGuid().ToString("N") + ".tmp";

        try
        {
            long downloadedBytes;

            await using (var source = await response.Content.ReadAsStreamAsync(cancellationToken))
            await using (var destination = File.Create(tempPath))
            {
                var buffer = new byte[options.BufferSize];
                downloadedBytes = 0;

                while (true)
                {
                    var read = await source.ReadAsync(buffer, cancellationToken);
                    if (read == 0)
                    {
                        break;
                    }

                    await destination.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                    downloadedBytes += read;
                    ReportProgress(
                        new HuggingFaceDownloadProgress
                        {
                            RepositoryId = repositoryId,
                            RepositoryPath = file.Path,
                            FileIndex = fileIndex + 1,
                            FileCount = fileCount,
                            TotalBytes = file.Size,
                            DownloadedBytes = downloadedBytes,
                            Completed = false,
                        },
                        options,
                        progress
                    );
                }
            }

            if (File.Exists(destinationPath))
            {
                File.Delete(destinationPath);
            }

            File.Move(tempPath, destinationPath);
            return downloadedBytes;
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    private HttpRequestMessage CreateRequest(HttpMethod method, Uri requestUri, HuggingFaceDownloadOptions options)
    {
        var request = new HttpRequestMessage(method, requestUri);
        request.Headers.UserAgent.ParseAdd("Onnxify.HuggingFace");

        if (!string.IsNullOrWhiteSpace(options.AccessToken))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", options.AccessToken);
        }

        return request;
    }

    private Uri BuildModelInfoUri(string repositoryId, string revision)
    {
        var path = "api/models/" + EscapePath(repositoryId);
        var builder = new UriBuilder(new Uri(_httpClient.BaseAddress!, path))
        {
            Query = "revision=" + Uri.EscapeDataString(revision)
        };

        return builder.Uri;
    }

    private Uri BuildResolveUri(string repositoryId, string revision, string filePath)
    {
        var path = string.Join(
            "/",
            EscapePath(repositoryId),
            "resolve",
            Uri.EscapeDataString(revision),
            EscapePath(filePath)
        );

        return new Uri(_httpClient.BaseAddress!, path);
    }

    private static string EscapePath(string value)
    {
        return string.Join(
            "/",
            value.Split('/', StringSplitOptions.RemoveEmptyEntries).Select(Uri.EscapeDataString));
    }

    private static string NormalizeRepositoryPath(string path)
    {
        return path.Replace('\\', '/');
    }

    private static string ResolveDestinationPath(string outputDirectoryPath, string repositoryPath)
    {
        var relativePath = repositoryPath.Replace('/', Path.DirectorySeparatorChar);

        if (Path.IsPathRooted(relativePath))
        {
            throw new InvalidOperationException($"Hugging Face file path '{repositoryPath}' is rooted.");
        }

        var destinationPath = Path.GetFullPath(Path.Combine(outputDirectoryPath, relativePath));
        var rootWithSeparator = outputDirectoryPath.EndsWith(Path.DirectorySeparatorChar)
            ? outputDirectoryPath
            : outputDirectoryPath + Path.DirectorySeparatorChar;

        if (!destinationPath.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"Hugging Face file path '{repositoryPath}' escapes the target directory.");
        }

        return destinationPath;
    }

    private static async Task EnsureSuccessAsync(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        throw new HttpRequestException(
            $"Hugging Face request failed with {(int)response.StatusCode} {response.ReasonPhrase}. {body}",
            null,
            response.StatusCode
        );
    }

    private sealed class ModelInfoResponse
    {
        [JsonPropertyName("siblings")]
        public IReadOnlyList<ModelSibling>? Siblings { get; init; }
    }

    private sealed class ModelSibling
    {
        [JsonPropertyName("rfilename")]
        public string? RFileName { get; init; }

        [JsonPropertyName("size")]
        public long? Size { get; init; }
    }

    private sealed class RepositoryFile
    {
        public required string Path { get; init; }

        public required long? Size { get; init; }
    }
}
