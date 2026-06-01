# Onnxify CLI

## Purpose

`src/Onnxify.CLI` is the repository command-line tool for quick model inspection and project scaffolding without first writing a temporary C# program.

Use it when you need to:

- inspect an ONNX model through the repository's own `OnnxModel.ToString()` view
- inspect a safetensors archive through `SafeTensors.ToString()`
- look only at ONNX inputs and outputs without the rest of the graph dump
- generate a standalone C# project from an existing `.onnx` file via `Onnxify.ProjectGenerator`
- download selected files from a Hugging Face model repository via `Onnxify.HuggingFace`

## Running From Repo Root

```powershell
dotnet run --project src/Onnxify.CLI -- <command>
```

## Global Tool Installation

If `Onnxify.CLI` is published as a .NET tool package, install it globally with:

```powershell
dotnet tool install Onnxify.CLI --global
```

After that, you can run commands directly as:

```powershell
onnxify <command>
```

To see whether the tool is already installed and which version you currently have:

```powershell
dotnet tool list --global
```

Look for `Onnxify.CLI` in the output.

To update an installed tool to the newest available version:

```powershell
dotnet tool update Onnxify.CLI --global
```

This is the main command to use when you want to check for a newer published version and install it if your current one is outdated.

If the tool is not installed yet, `dotnet tool update` will fail, and you should run:

```powershell
dotnet tool install Onnxify.CLI --global
```

For help:

```powershell
dotnet run --project src/Onnxify.CLI -- --help
dotnet run --project src/Onnxify.CLI -- onnx --help
dotnet run --project src/Onnxify.CLI -- safetensors --help
dotnet run --project src/Onnxify.CLI -- project --help
dotnet run --project src/Onnxify.CLI -- hf --help
```

If you installed the global tool, the equivalent help commands are:

```powershell
onnxify --help
onnxify onnx --help
onnxify safetensors --help
onnxify project --help
onnxify hf --help
```

## ONNX Inspection

Full model dump:

```powershell
dotnet run --project src/Onnxify.CLI -- onnx show path\to\model.onnx
```

This prints the `OnnxModel` string view, including:

- model metadata
- graph inputs and outputs
- initializers
- placeholder values
- nodes

Inputs and outputs only:

```powershell
dotnet run --project src/Onnxify.CLI -- onnx io path\to\model.onnx
dotnet run --project src/Onnxify.CLI -- onnx inputs-outputs path\to\model.onnx
```

Use this when you only care about model contract shape and do not want the full graph dump.

## Safetensors Inspection

```powershell
dotnet run --project src/Onnxify.CLI -- safetensors show path\to\model.safetensors
```

This is useful for checking:

- tensor names
- dtypes
- shapes
- metadata entries
- preview values

## Project Generation

Basic generation:

```powershell
dotnet run --project src/Onnxify.CLI -- project generate path\to\model.onnx path\to\output-dir --overwrite
```

Example with customization:

```powershell
dotnet run --project src/Onnxify.CLI -- project generate path\to\model.onnx path\to\output-dir --project-name MyGeneratedModel --namespace MyCompany.Models --overwrite
```

Useful options:

- `--project-name <name>`
- `--namespace <name>`
- `--package-version <version>`
- `--program-class-name <name>`
- `--factory-method-name <name>`
- `--program-file-name <name>`
- `--tensor-directory-name <name>`
- `--project-file-name <name>`
- `--no-project-file`
- `--overwrite`

The generated project recreates the source ONNX model through Onnxify APIs and writes a runnable `Program.cs`.

## Hugging Face Downloads

Download only a `bf16` variant plus support files from a Hugging Face model repository:

```powershell
dotnet run --project src/Onnxify.CLI -- hf download onnx-community/gemma-4-E2B-it-ONNX path\to\gemma-bf16 --variant bf16 --exclude "*.md5" --overwrite
```

Equivalent global-tool form:

```powershell
onnxify hf download onnx-community/gemma-4-E2B-it-ONNX path\to\gemma-bf16 --variant bf16 --exclude "*.md5" --overwrite
```

Useful options:

- `--revision <revision>`
- `--token <token>`
- `--token-env <name>` defaults to `HF_TOKEN`
- `--include <pattern>` can be repeated
- `--exclude <pattern>` can be repeated
- `--variant <name>` includes support files and paths containing the variant value, such as `bf16`
- `--overwrite`
- `--quiet`

Use explicit `--include` and `--exclude` patterns for large repositories with multiple ONNX weight variants, for example `--include "*bf16*" --include "*.json" --include "*.model" --exclude "*.md5"`.

## When To Prefer CLI Vs Library Code

Prefer the CLI when you need a quick answer from an existing file.

Prefer library code when you need to:

- mutate a graph
- inspect or transform models programmatically
- add tests around model behavior
- integrate ONNX or safetensors workflows into another .NET application
- customize Hugging Face download filtering, progress reporting, or authentication beyond the command-line options
