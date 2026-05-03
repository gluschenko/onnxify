# Onnxify.CLI

`Onnxify.CLI` is the command-line entry point for this repository. It is published as a `dotnet tool` and exposes the `onnxify` command.

## Install

```bash
dotnet tool install --global Onnxify.CLI
```

## What It Provides

- Inspect ONNX model structure from the terminal.
- Inspect safetensors files from the terminal.
- Generate C# project output from an ONNX model.

## Example Commands

```bash
onnxify --version
onnxify onnx show model.onnx
onnxify onnx io model.onnx
onnxify safetensors show model.safetensors
onnxify project generate model.onnx output-dir
```

## Repository

- Source: <https://github.com/gluschenko/onnxify>
