# Onnxify.Compiler

`Onnxify.Compiler` is the shared compiler layer for bidirectional transformations between C# TorchSharp computation and ONNX graphs.

## Install

```bash
dotnet add package Onnxify.Compiler
```

The compiler package is also brought in transitively by the Onnxify consumer packages that integrate with the shared compiler boundary.

## Current Scope

The initial package establishes the compiler-owned source, sink, and session contracts without coupling the compiler to `Onnxify.ModelGenerator` or `Onnxify.TorchSharp`. Later compiler phases add the shared IR, ONNX and C# frontends/backends, operator mappings, diagnostics, and roundtrip validation on top of this boundary.

The dependency direction is intentionally one-way:

```text
Onnxify.Compiler -> Onnxify
Onnxify.ModelGenerator -> Onnxify.Compiler
Onnxify.TorchSharp -> Onnxify.Compiler
```

This keeps compiler semantics in one project and prevents the two consumer packages from growing parallel compiler implementations.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
