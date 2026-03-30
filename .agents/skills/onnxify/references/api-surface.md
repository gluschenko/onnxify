# Onnxify API Surface

Use this reference when you need concrete entry points for repository work.

## Core Types

- `src/Onnxify/OnnxModel.cs`
  - `OnnxModel.Create(...)`
  - `OnnxModel.FromFile(path)`
  - `model.Save(path, overwrite)`
  - model metadata: `ProducerName`, `ProducerVersion`, `ModelVersion`, `IrVersion`, `Domain`, `Document`
  - `model.Graph`

- `src/Onnxify/OnnxGraph.cs`
  - collections: `Inputs`, `Outputs`, `Initializers`, `Placeholders`, `Nodes`
  - lookups: `GetNode(name)`, `GetValue(name)`
  - builders: `AddInput`, `AddOutput`, `AddValue`, `AddTensor`, `AddEdge`, `AddNode`

- Other common types
  - `src/Onnxify/OnnxNode.cs`
  - `src/Onnxify/OnnxTensor.cs`
  - `src/Onnxify/OnnxValue.cs`
  - `src/Onnxify/OnnxValueType.cs`
  - `src/Onnxify/OnnxAttribute.cs`

## Best Existing Examples

- `src/Onnxify.Tests/OnnxModelTests.cs`
  - creation options
  - save/load round-trip
  - tensor/value/node assertions
  - variadic input/output operator coverage

- `src/Onnxify.Tests/OnnxProjectGeneratorTests.cs`
  - generated project expectations
  - asserting emitted source and asset files

- `src/Onnxify.ConsoleTest/Program.cs`
  - playground for loading assets and composing graphs
  - examples of `Conv`, `Relu`, `MaxPool`, `Flatten`, `Gemm`

- `src/Onnxify.Examples/Program.cs`
  - end-to-end example around exporting and evaluating a model

## Useful Heuristics

- If behavior is about raw ONNX persistence, start in the library and tests, not in `ConsoleTest`.
- If the repo already has a typed operator helper, copy that pattern instead of emitting raw stringly-typed nodes.
- If adding a new helper, confirm whether the task belongs in the main library or a placeholder project.
- For regression-proofing, prefer assertions on names, shapes, attributes, and edge wiring over only checking counts.
