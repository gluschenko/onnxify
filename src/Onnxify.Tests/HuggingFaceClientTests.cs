using System.Net;
using System.Text;
using Onnxify.HuggingFace;

namespace Onnxify.Tests;

public sealed class HuggingFaceClientTests
{
    [Fact]
    public async Task DownloadRepositoryAsync_WritesRepositoryFiles()
    {
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            using var httpClient = new HttpClient(new FakeHuggingFaceHandler())
            {
                BaseAddress = new Uri("https://huggingface.co/")
            };

            var client = new HuggingFaceClient(httpClient);
            var progress = new CapturingProgress();

            var result = await client.DownloadRepositoryAsync(
                "sample/repo",
                outputDirectoryPath,
                new HuggingFaceDownloadOptions
                {
                    Revision = "main",
                    Overwrite = true,
                    BufferSize = 4,
                },
                progress);

            Assert.Equal("sample/repo", result.RepositoryId);
            Assert.Equal("main", result.Revision);
            Assert.Equal(2, result.DownloadedFileCount);
            Assert.Equal("model content", await File.ReadAllTextAsync(Path.Combine(outputDirectoryPath, "onnx", "model.onnx")));
            Assert.Equal("metadata content", await File.ReadAllTextAsync(Path.Combine(outputDirectoryPath, "metadata.json")));
            Assert.Contains(progress.Items, static x => x.RepositoryPath == "onnx/model.onnx" && x.Completed);
            Assert.Contains(progress.Items, static x => x.RepositoryPath == "metadata.json" && x.Completed);
        }
        finally
        {
            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    [Fact]
    public async Task DownloadRepositoryAsync_RejectsPathsEscapingTargetDirectory()
    {
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            using var httpClient = new HttpClient(new EscapingPathHandler())
            {
                BaseAddress = new Uri("https://huggingface.co/")
            };

            var client = new HuggingFaceClient(httpClient);

            await Assert.ThrowsAsync<InvalidOperationException>(() => client.DownloadRepositoryAsync("sample/repo", outputDirectoryPath));
        }
        finally
        {
            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    [Fact]
    public async Task DownloadRepositoryAsync_AppliesIncludeAndExcludePathFilters()
    {
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            using var httpClient = new HttpClient(new FilteredRepositoryHandler())
            {
                BaseAddress = new Uri("https://huggingface.co/")
            };

            var client = new HuggingFaceClient(httpClient);

            var result = await client.DownloadRepositoryAsync(
                "sample/repo",
                outputDirectoryPath,
                new HuggingFaceDownloadOptions
                {
                    IncludePath = path => path.Contains("bf16", StringComparison.OrdinalIgnoreCase) || path == "config.json",
                    ExcludePath = path => path.EndsWith(".md5", StringComparison.OrdinalIgnoreCase),
                });

            Assert.Equal(2, result.DownloadedFileCount);
            Assert.True(File.Exists(Path.Combine(outputDirectoryPath, "onnx", "decoder_model_bf16.onnx")));
            Assert.True(File.Exists(Path.Combine(outputDirectoryPath, "config.json")));
            Assert.False(File.Exists(Path.Combine(outputDirectoryPath, "onnx", "decoder_model_q4f16.onnx")));
            Assert.False(File.Exists(Path.Combine(outputDirectoryPath, "onnx", "decoder_model_bf16.onnx.md5")));
        }
        finally
        {
            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    [Fact]
    public async Task DownloadRepositoryAsync_InvokesProgressCallback()
    {
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            using var httpClient = new HttpClient(new FakeHuggingFaceHandler())
            {
                BaseAddress = new Uri("https://huggingface.co/")
            };

            var client = new HuggingFaceClient(httpClient);
            var callbackItems = new List<HuggingFaceDownloadProgress>();

            await client.DownloadRepositoryAsync(
                "sample/repo",
                outputDirectoryPath,
                new HuggingFaceDownloadOptions
                {
                    BufferSize = 4,
                    ProgressCallback = callbackItems.Add,
                });

            Assert.Contains(callbackItems, static x => x.RepositoryPath == "onnx/model.onnx" && !x.Completed && x.DownloadedBytes == 4);
            Assert.Contains(callbackItems, static x => x.RepositoryPath == "onnx/model.onnx" && x.Completed && x.DownloadedBytes == 13);
            Assert.Contains(callbackItems, static x => x.RepositoryPath == "metadata.json" && x.Completed && x.DownloadedBytes == 16);
        }
        finally
        {
            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    [Fact]
    public async Task DownloadRepositoryAsync_AllowsRepositoryFilesWithoutSize()
    {
        var outputDirectoryPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}");

        try
        {
            using var httpClient = new HttpClient(new MissingSizeHandler())
            {
                BaseAddress = new Uri("https://huggingface.co/")
            };

            var client = new HuggingFaceClient(httpClient);

            var result = await client.DownloadRepositoryAsync("sample/repo", outputDirectoryPath);
            var file = Assert.Single(result.Files);

            Assert.Null(file.Size);
            Assert.Equal("data", await File.ReadAllTextAsync(Path.Combine(outputDirectoryPath, "model.onnx")));
        }
        finally
        {
            if (Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, recursive: true);
            }
        }
    }

    private sealed class FakeHuggingFaceHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = request.RequestUri?.AbsolutePath;

            return path switch
            {
                "/api/models/sample/repo" => RespondJson(
                    """
                    {
                      "siblings": [
                        { "rfilename": "onnx/model.onnx", "size": 13 },
                        { "rfilename": "metadata.json", "size": 16 }
                      ]
                    }
                    """),
                "/sample/repo/resolve/main/onnx/model.onnx" => RespondText("model content"),
                "/sample/repo/resolve/main/metadata.json" => RespondText("metadata content"),
                _ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound)
                {
                    Content = new StringContent(path ?? string.Empty)
                }),
            };
        }

        private static Task<HttpResponseMessage> RespondJson(string json)
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            });
        }

        private static Task<HttpResponseMessage> RespondText(string text)
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(text, Encoding.UTF8, "application/octet-stream")
            });
        }
    }

    private sealed class EscapingPathHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(
                    """
                    {
                      "siblings": [
                        { "rfilename": "../escaped.txt", "size": 1 }
                      ]
                    }
                    """,
                    Encoding.UTF8,
                    "application/json")
            });
        }
    }

    private sealed class FilteredRepositoryHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = request.RequestUri?.AbsolutePath;

            if (path == "/api/models/sample/repo")
            {
                return RespondJson(
                    """
                    {
                      "siblings": [
                        { "rfilename": "onnx/decoder_model_bf16.onnx", "size": 4 },
                        { "rfilename": "onnx/decoder_model_bf16.onnx.md5", "size": 4 },
                        { "rfilename": "onnx/decoder_model_q4f16.onnx", "size": 4 },
                        { "rfilename": "config.json", "size": 4 }
                      ]
                    }
                    """);
            }

            if (path is "/sample/repo/resolve/main/onnx/decoder_model_bf16.onnx"
                or "/sample/repo/resolve/main/config.json")
            {
                return RespondText("data");
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound)
            {
                Content = new StringContent(path ?? string.Empty)
            });
        }

        private static Task<HttpResponseMessage> RespondJson(string json)
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            });
        }

        private static Task<HttpResponseMessage> RespondText(string text)
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(text, Encoding.UTF8, "application/octet-stream")
            });
        }
    }

    private sealed class MissingSizeHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = request.RequestUri?.AbsolutePath;

            return path switch
            {
                "/api/models/sample/repo" => RespondJson(
                    """
                    {
                      "siblings": [
                        { "rfilename": "model.onnx" }
                      ]
                    }
                    """),
                "/sample/repo/resolve/main/model.onnx" => RespondText("data"),
                _ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound)
                {
                    Content = new StringContent(path ?? string.Empty)
                }),
            };
        }

        private static Task<HttpResponseMessage> RespondJson(string json)
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            });
        }

        private static Task<HttpResponseMessage> RespondText(string text)
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(text, Encoding.UTF8, "application/octet-stream")
            });
        }
    }

    private sealed class CapturingProgress : IProgress<HuggingFaceDownloadProgress>
    {
        public List<HuggingFaceDownloadProgress> Items { get; } = [];

        public void Report(HuggingFaceDownloadProgress value)
        {
            Items.Add(value);
        }
    }
}
