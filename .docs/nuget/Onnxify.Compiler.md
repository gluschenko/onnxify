# Onnxify.Compiler

`Onnxify.Compiler` is the shared compiler layer for bidirectional transformations between C# TorchSharp computation and ONNX graphs.

## Install

```bash
dotnet add package Onnxify.Compiler
```

The compiler package is also brought in transitively by the Onnxify consumer packages that integrate with the shared compiler boundary.

## Current Scope

The package provides the compiler-owned source, sink, and session contracts plus the immutable computation-tree IR used by both directions. The IR includes typed tensor/value metadata, dimensions, literals, state members, normalized operator attributes, module calls, reusable blocks, ordered C# syntax, capability classifications, and stage-aware diagnostics.

ONNX and C# frontends/backends, shared operator mappings, and executable roundtrip validation are implemented by later compiler phases on top of this model.

The dependency direction is intentionally one-way:

```text
Onnxify.Compiler -> Onnxify
Onnxify.ModelGenerator -> Onnxify.Compiler
Onnxify.TorchSharp -> Onnxify.Compiler
```

This keeps compiler semantics in one project and prevents the two consumer packages from growing parallel compiler implementations.

## Repository

- Source: <https://github.com/gluschenko/onnxify>
