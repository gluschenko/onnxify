# Onnxify CLI

## Purpose

`src/Onnxify.CLI` is the repository command-line tool for quick model inspection and project scaffolding without first writing a temporary C# program.

Use it when you need to:

- inspect an ONNX model through the repository's own `OnnxModel.ToString()` view
- inspect a safetensors archive through `SafeTensors.ToString()`
- look only at ONNX inputs and outputs without the rest of the graph dump
- generate a standalone C# project from an existing `.onnx` file via `Onnxify.ProjectGenerator`

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

For help:

```powershell
dotnet run --project src/Onnxify.CLI -- --help
dotnet run --project src/Onnxify.CLI -- onnx --help
dotnet run --project src/Onnxify.CLI -- safetensors --help
dotnet run --project src/Onnxify.CLI -- project --help
```

If you installed the global tool, the equivalent help commands are:

```powershell
onnxify --help
onnxify onnx --help
onnxify safetensors --help
onnxify project --help
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

## When To Prefer CLI Vs Library Code

Prefer the CLI when you need a quick answer from an existing file.

Prefer library code when you need to:

- mutate a graph
- inspect or transform models programmatically
- add tests around model behavior
- integrate ONNX or safetensors workflows into another .NET application
