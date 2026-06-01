## 0.1.3

- Initial release.
- Added `HuggingFaceClient` for downloading Hugging Face model repository files into a local directory.
- Added include and exclude path filters for downloading only selected variants, such as `bf16`.
- Added progress reporting through `ProgressCallback` and `IProgress<HuggingFaceDownloadProgress>`.
- Added revision, access token, overwrite, and path-safety support.
