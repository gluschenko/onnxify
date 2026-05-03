# Onnxify.TorchSharp

`Onnxify.TorchSharp` is the TorchSharp-to-ONNX export layer for this repository. Use it when you want to translate TorchSharp modules and model structure into `Onnxify` graph operations.

## Install

```bash
dotnet add package Onnxify.TorchSharp
```

## What It Provides

- TorchSharp-oriented ONNX export helpers.
- A bridge between TorchSharp model structure and `Onnxify` graph construction.
- An extension point for adding `Export(...)` coverage for additional modules and operators.

## Notes

This package builds on top of `Onnxify`, so it is most useful when you are already working with the core ONNX object model from this repository.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
