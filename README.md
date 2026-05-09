# Onnxify

Onnxify is an experimental .NET library for reading, inspecting, and writing ONNX models.

Machine learning workflows are often difficult not because models are impossible to run, but because they are difficult to understand, inspect, adapt, and carry from one environment to another. A lot of useful work happens in that space between research and production, where people need clarity, control, and confidence rather than another opaque black box. Onnxify exists for that middle ground.

The idea behind this repository is simple: models should be easier to work with, easier to reason about, and easier to integrate into real development workflows. If ONNX is meant to be a common language for models, then the tools around it should help people move faster, make smaller changes safely, and build their own workflows without unnecessary friction. That is the direction Onnxify is trying to push.

## Master plan

- [x] `Onnxify`
- [x] `Onnxify.TorchSharp`
- [x] `Onnxify.ML`
- [x] `Onnxify.ML.TorchSharp`
- [x] `Onnxify.ProjectGenerator`
- [x] `Onnxify.ModelGenerator`
- [x] `Onnxify.Safetensors`
- [x] `Onnxify.CLI`

## TODO

- [ ] OnnxGraph rework
- [ ] SourceGenerator: operator type annotations
- [ ] SourceGenerator: fully-typed operator Input/Output fields (OneOf?)
- [ ] Async I/O ops
- [ ] Graph edges in a single collection (or in two for placeholders)
- [ ] Graph manipulations: add nodes, remove nodes, replace nodes
- [ ] Graph cyclicity validation
- [ ] CLI for agents and humans (to explore ONNX files)
- [x] Project generator generates operator nodes
- [x] Parse pytorch\torch\onnx\_internal\torchscript_exporter (create MD with support status)
- [x] Generate agent skills from operator-schema.json
- [x] ToString for OnnxModel, OnnxNode, OnnxxTensor, etc (recursive?)
- [x] OnnxDataProvider, SafetensorsDataProvider, BaseDataProvider...
- [x] Agent skills for Export implementation on Torch modules
- [x] Allow to add or remove OnnxModel meta (training info, imports, producer, version)

## Status

This project is in an early stage.

The core library already supports:

- loading ONNX models from disk
- creating new ONNX models in code
- inspecting graphs, nodes, values, tensors, and attributes
- saving models back to `.onnx`

Some parts of the repository are still incomplete or experimental, especially the legacy exporter path and the placeholder projects.

## Requirements

- .NET 10 SDK
- Windows development environment is the primary setup used in this repository

Optional:

- `protoc` if you want to regenerate the protobuf C# files from the ONNX `.proto3` sources

## Getting Started

Clone the repository and build the solution:

```powershell
git clone --recurse-submodules https://github.com/gluschenko/onnxify.git
cd onnxify
dotnet build src\Onnxify.slnx
```

If the generated protobuf files already exist in `src/Onnxify/Protobuf`, `protoc` is not required for a normal build. It is only needed when those files must be regenerated.

## Install the Codex Skill

If you use Codex and want repository-specific help for `Onnxify` and `Onnxify.TorchSharp`, you can install the bundled `onnxify` skill directly from GitHub.

You do not need to clone this repository just to install the skill. The installer can download the skill folder from GitHub into your local Codex skills directory. Clone the repository only if you also want the source code, examples, tests, or manual local development.

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }

py -3 "$codexHome\skills\.system\skill-installer\scripts\install-skill-from-github.py" `
  --repo gluschenko/onnxify `
  --path .agents/skills/onnxify
```

You can also install it by URL:

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }

py -3 "$codexHome\skills\.system\skill-installer\scripts\install-skill-from-github.py" `
  --url "https://github.com/gluschenko/onnxify/tree/main/.agents/skills/onnxify"
```

Restart Codex after installation so the new skill is picked up.

## License

This repository is licensed under the terms of the [LICENSE](LICENSE) file.
