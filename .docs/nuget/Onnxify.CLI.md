> **Warning**
> This package is currently in active development and research. Its public API is unstable and may change radically in future versions.

# Onnxify.CLI

`Onnxify.CLI` is the command-line entry point for this repository. It is published as a `dotnet tool` and exposes the `onnxify` command.

## Install

```bash
dotnet tool install --global Onnxify.CLI
```

## What It Provides

- Inspect ONNX model structure from the terminal.
- Compare ONNX model structure from the terminal.
- Inspect safetensors files from the terminal.
- Generate C# project output from an ONNX model.

## Example Commands

```bash
onnxify --version
onnxify onnx show model.onnx
onnxify onnx show --inputs --outputs --nodes model.onnx
onnxify onnx diff original.onnx exported.onnx
onnxify onnx diff --nodes original.onnx exported.onnx
onnxify onnx inputs-outputs model.onnx
onnxify safetensors show model.safetensors
onnxify project generate model.onnx output-dir
```

## ONNX Commands

### Show A Model

```bash
onnxify onnx show [options] <model.onnx>
```

Without options, `onnx show` prints the default `OnnxModel` representation. With one or more section options, it prints a compact summary and only the requested sections.

Options:

- `--inputs` includes graph inputs.
- `--outputs` includes graph outputs.
- `--values` includes initializer previews and intermediate value-info entries.
- `--nodes` includes compact node signatures with inputs, outputs, and attributes.

### Compare Two Models

```bash
onnxify onnx diff [options] <left.onnx> <right.onnx>
```

`onnx diff` compares two ONNX models and prints metadata, operator counts, and ordered graph-section differences. Without section options, it includes inputs, outputs, values, and nodes. With section options, it always includes metadata and operator counts, then only the requested graph sections.

Options:

- `--inputs` includes graph input differences.
- `--outputs` includes graph output differences.
- `--values` includes initializer and intermediate value differences.
- `--nodes` includes compact node signature differences.

### Show Inputs And Outputs

```bash
onnxify onnx io <model.onnx>
onnxify onnx inputs-outputs <model.onnx>
```

Both forms print the model input and output tensors.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
