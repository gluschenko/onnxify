# Compiler Architecture And Implementation

Use this page for OXY-019 and OXY-020—OXY-027 implementation work.

## Ownership And Dependency Direction

All new compiler functionality belongs in `Onnxify.Compiler`: the computation-tree/IR model, frontends, backends, operator mappings, diagnostics, validation primitives, and compiler orchestration APIs.

The intended dependency graph is:

```text
Onnxify.Compiler -> Onnxify
Onnxify.ModelGenerator -> Onnxify.Compiler
Onnxify.TorchSharp -> Onnxify.Compiler
Onnxify.Compiler.Tests -> Onnxify.Compiler
```

`Onnxify.Compiler` must not reference `Onnxify.ModelGenerator` or `Onnxify.TorchSharp`. Consumer projects may contain thin adapters for package-specific public/runtime types, but must not contain duplicate parsing, lowering, tree transformation, code-generation, or ONNX-emission logic.

## Compiler Surface

The compiler should provide operations equivalent to:

```csharp
var tree = Compiler.CreateTreeFromOnnx(onnxModel);
var csharp = Compiler.GenerateCSharp(tree);
var onnx = Compiler.GenerateOnnx(tree);
var treeFromTorch = Compiler.CreateTreeFromTorchSharp(torchSource);
```

Exact names may change, but tree construction and generation must return compiler-owned models and generated C# source as a `string`. Existing ModelGenerator and TorchSharp APIs remain compatibility façades over these services.

## Implementation Rules

- Keep the IR independent of raw decompiler AST and TorchSharp runtime objects.
- Use compiler-owned contracts for package-specific sources and sinks to keep the graph acyclic.
- Use deterministic, most-specific-first scanner/printer/extension dispatch.
- Preserve explicit diagnostics, source spans, normalized attributes, symbolic dimensions, metadata, scopes, and state.
- Prefer shared semantic abstractions and operator-family implementations over fixture-specific branches.
- Retain compatibility fallbacks only as explicitly marked migration surfaces with retirement conditions.

## Phase Order

1. OXY-020: create the project, test project, solution wiring, and contracts.
2. OXY-021: define IR, computation-tree nodes, metadata, attributes, and diagnostics.
3. OXY-022: implement ONNX import to tree and tree to ONNX emission.
4. OXY-023: implement C# TorchSharp scanning and C# generation.
5. OXY-024: migrate shared operator mappings and their paired verification.
6. OXY-025: migrate helper methods, module calls, recursion, and safe static control flow.
7. OXY-026: route public APIs through the compiler and add extensibility.
8. OXY-027: complete roundtrip validation, documentation, and cleanup.

Each phase must leave independent verification evidence before the next phase is considered complete. See [compiler-roundtrip-testing.md](compiler-roundtrip-testing.md) for operator verification requirements.
