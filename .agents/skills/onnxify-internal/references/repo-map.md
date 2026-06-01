# Onnxify Repository Map

Use this reference when you need to decide which project owns a repository-maintenance change.

## Solution Folders

- `src/Onnxify.slnx`
  - `/generators/`: `Onnxify.AgentSkillGenerator`, `Onnxify.SourceGenerator`, `Onnxify.TorchSharp.Observer`
  - `/packages/`: `Onnxify`, `Onnxify.TorchSharp`, `Onnxify.ML`, `Onnxify.ML.TorchSharp`, `Onnxify.Safetensors`, `Onnxify.HuggingFace`, `Onnxify.ProjectGenerator`, `Onnxify.ModelGenerator`, `Onnxify.CLI`
  - `/tests/`: `Onnxify.ConsoleTest`, `Onnxify.Examples`, `Onnxify.Examples.Models`, `Onnxify.Tests`

## Project Roles

- `src/Onnxify`
  - core ONNX object model and protobuf-backed library surface

- `src/Onnxify.TorchSharp`
  - TorchSharp-to-ONNX export layer and conversion helpers

- `src/Onnxify.HuggingFace`
  - Hugging Face model repository download client with include/exclude filters, progress callbacks, revision, token, and path-safety support

- `src/Onnxify.Tests`
  - automated regression tests for library, generator, and project-generator behavior

- `src/Onnxify.ConsoleTest`
  - manual playground and ad hoc repro surface

- `src/Onnxify.Examples`
  - curated usage examples and composed export samples

- `src/Onnxify.AgentSkillGenerator`
  - generates the large operator and TorchSharp-converter reference docs used by `.agents/skills/onnxify`

- `src/Onnxify.ProjectGenerator`
  - emits C# project output from model inputs

- `src/Onnxify.SourceGenerator`
  - source-generation support code and shared models used by generation workflows

- `src/Onnxify.OperatorSchemaGenerator`
  - native/CMake-based helper for schema extraction workflows

- `src/Onnxify.TorchSharp.Observer`
  - reporting and observation utilities around TorchSharp coverage and operator support

## Docs And Skills

- `README.md`
  - public repository overview, build instructions, install notes, and quick examples

- `.agents/skills/onnxify`
  - repository-specific skill for working on the library APIs and TorchSharp export surface

- `.agents/skills/onnxify-internal`
  - repository-specific skill for maintaining docs, skills, generators, and repo structure
