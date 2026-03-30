---
name: onnxify
description: Use this skill when working with the Onnxify .NET library in this repository: reading or writing ONNX models, inspecting graphs, adding or editing nodes, tensors, attributes, or value types, extending operator wrappers, or validating serialization and project generation behavior under src/Onnxify, src/Onnxify.Tests, src/Onnxify.ConsoleTest, and related projects.
---

# Onnxify

## Overview

Use this skill for repository-specific work on Onnxify rather than generic ONNX advice. It is tuned for the object model in `src/Onnxify`, the current tests in `src/Onnxify.Tests`, and the example/playground projects in `src/Onnxify.ConsoleTest` and `src/Onnxify.Examples`.

Read [references/api-surface.md](references/api-surface.md) when you need concrete entry points, example file locations, or a quick reminder of which project to touch.

## Quick Start

When handling an Onnxify task:

1. Identify the task shape: model load/save, graph editing, operator wrapper work, project generation, or examples/tests.
2. Read the closest production type first, usually `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxTensor`, `OnnxValue`, or the relevant generated/operator wrapper class.
3. Read an existing test before changing behavior. Prefer `src/Onnxify.Tests` over guessing expected semantics.
4. Make the smallest repo-consistent change.
5. Validate with focused tests. Prefer round-trip coverage for serialization changes.

## 1. Loading And Inspecting Models

- Start with `OnnxModel.FromFile(path)` for existing models.
- Inspect through `model.Graph`, then walk `Inputs`, `Outputs`, `Initializers`, `Placeholders`, and `Nodes`.
- Prefer repository terminology: graph values may be inputs, outputs, placeholders, initializers, or loose edges.
- If you only need to inspect structure, avoid rewriting the model unless the task requires it.

## 2. Creating Or Editing Models

- Create new models with `OnnxModel.Create(new OnnxModelCreationOptions { ... })`.
- Add graph members through `AddInput`, `AddOutput`, `AddValue`, `AddTensor`, `AddEdge`, and `AddNode`.
- Respect unique-name constraints. The graph helpers throw on duplicates, so preserve stable names when patching an existing graph.
- When a task is operator-oriented, prefer existing typed helpers or wrapper classes over raw `AddNode` if the repository already exposes them.
- When no helper exists yet, implement behavior in the most local, consistent layer instead of introducing a parallel abstraction.

## 3. Serialization And Round Trips

- Save models with `model.Save(path, overwrite: ...)`.
- For any change that affects protobuf conversion, graph composition, or tensor/value metadata, add or update a round-trip test.
- Verify both structure and metadata after reload: producer info, IR/opset, graph members, attributes, tensor values, and node connections.
- Preserve `DataLocation` and path-sensitive behavior when touching tensor loading or external data logic.

## 4. Testing Strategy

- Put automated behavior checks in `src/Onnxify.Tests`.
- Use `src/Onnxify.ConsoleTest` only as a manual playground, not as proof that behavior is correct.
- Keep tests focused and deterministic. Temporary files are acceptable if cleaned up in `finally`.
- For generation features, assert on emitted file contents and output paths, as done in project generator tests.

## 5. Project Conventions

- This repo is early-stage. Favor small extensions over large framework reshapes.
- Preserve the lightweight wrapper model over protobuf-generated ONNX classes.
- Match the existing C# style: explicit names, small helper methods, and straightforward `Fact` tests.
- If a project looks like a placeholder, avoid building new functionality there unless the user asked for that exact surface.

## Common Task Mapping

- Read or rewrite `.onnx`: start with `src/Onnxify/OnnxModel.cs` and related tests.
- Add graph members or wire nodes: start with `src/Onnxify/OnnxGraph.cs`.
- Debug a node or attribute issue: inspect `OnnxNode`, `OnnxAttribute`, and a nearby round-trip test.
- Add or adjust higher-level operator APIs: inspect existing operator helper usage in tests and console examples first.
- Generate C# from a model: inspect `src/Onnxify.ProjectGenerator`.
- Need a manual repro with real assets: inspect `src/Onnxify.ConsoleTest/Assets` and `src/Onnxify.ConsoleTest/Program.cs`.

## Validation Checklist

- Did the change preserve existing naming and graph-linking behavior?
- Is there a focused automated test for the new or changed behavior?
- If serialization changed, did you verify save + load instead of only in-memory state?
- If the task touched examples or playground code only, did you avoid silently changing library semantics?
