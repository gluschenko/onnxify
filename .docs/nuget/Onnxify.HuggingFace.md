> **Warning**
> This package is currently in active development and research. Its public API is unstable and may change radically in future versions.

# Onnxify.HuggingFace

`Onnxify.HuggingFace` downloads model repository contents from Hugging Face into a local directory. It is intended for ONNX model workflows where you want to fetch only the files needed for a specific runtime variant, such as `bf16`, instead of cloning or downloading the entire repository.

## Install

```bash
dotnet add package Onnxify.HuggingFace
```

## What It Provides

- Download all files listed by a Hugging Face model repository.
- Preserve repository-relative paths in the target directory.
- Restrict downloads with include and exclude path filters.
- Track download progress through a callback or `IProgress<HuggingFaceDownloadProgress>`.
- Use a revision and optional Hugging Face access token.
- Avoid writing files outside the target directory when repository paths are malformed.

## Basic Usage

```csharp
using Onnxify.HuggingFace;

var client = new HuggingFaceClient();

var result = await client.DownloadRepositoryAsync(
    repositoryId: "onnx-community/gemma-4-E2B-it-ONNX",
    targetDirectoryPath: "models/gemma-fp16",
    options: new HuggingFaceDownloadOptions
    {
        Revision = "main",
        IncludePath = path =>
            path.Contains("fp16", StringComparison.OrdinalIgnoreCase)
            || path.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            || path.EndsWith(".model", StringComparison.OrdinalIgnoreCase),
        ExcludePath = path => path.EndsWith(".md5", StringComparison.OrdinalIgnoreCase),
        ProgressCallback = progress =>
        {
            if (progress.Completed)
            {
                Console.WriteLine($"Downloaded {progress.FileIndex}/{progress.FileCount}: {progress.RepositoryPath}");
            }
        },
        Overwrite = true,
    }
);

Console.WriteLine($"Downloaded files: {result.DownloadedFileCount}");
```

## Authentication

For private or gated repositories, pass an access token:

```csharp
var result = await client.DownloadRepositoryAsync(
    "owner/private-model",
    "models/private-model",
    new HuggingFaceDownloadOptions
    {
        AccessToken = Environment.GetEnvironmentVariable("HF_TOKEN"),
    }
);
```

## CLI

The `Onnxify.CLI` package also exposes this workflow:

```bash
onnxify hf download onnx-community/gemma-4-E2B-it-ONNX models/gemma-bf16 --variant bf16 --exclude "*.md5" --overwrite
```

## Repository

- Source: <https://github.com/gluschenko/onnxify>
