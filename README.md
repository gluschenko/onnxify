# Onnxify

Onnxify is an experimental .NET library for reading, inspecting, and writing ONNX models.

## Master plan

- [x] `Onnxify`
- [x] `Onnxify.TorchSharp`
- [ ] `Onnxify.ML`
- [ ] `Onnxify.ML.TorchSharp`
- [ ] `Onnxify.ProjectGenerator`
- [ ] `Onnxify.Safetensors`
- [ ] `Onnxify.GGUF`
- [ ] `Onnxify.TFLite`
- [ ] `Onnxify.Runtime.LiteRT`

## TODO

- [ ] Async I/O ops
- [ ] Fully-typed operator Input/Output fields (OneOf?)
- [ ] Graph edges in a single collection (or in two for placeholders)
- [x] Project generator generates operator nodes
- [x] Parse pytorch\torch\onnx\_internal\torchscript_exporter (create MD with support status)
- [x] Generate agent skills from operator-schema.json
- [x] ToString for OnnxModel, OnnxNode, OnnxxTensor, etc (recursive?)
- [x] OnnxDataProvider, SafetensorsDataProvider, BaseDataProvider...
- [ ] Graph manipulations: add nodes, remove nodes, replace nodes
- [ ] Graph cyclicity validation
- [ ] CLI for agents and humans (to explore ONNX files)
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
git clone --recurse-submodules https://github.com/your-org/onnxify.git
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

## Quick Example

Create a new model:

```csharp
using Onnxify;

var model = OnnxModel.Create(new OnnxModelCreationOptions
{
    ProducerName = "demo-app",
    Opset = 13,
    IrVersion = 8,
});

model.Save("model.onnx", overwrite: true);
```

Load and inspect an existing model:

```csharp
using Onnxify;

var model = OnnxModel.FromFile("input.onnx");

Console.WriteLine(model.ProducerName);
Console.WriteLine(model.Graph.Name);

foreach (var node in model.Graph.Nodes)
{
    Console.WriteLine($"{node.Name} [{node.OpType}]");
}
```

## Project Layout

```text
src/
  Onnxify/                    Main library
  Onnxify.ConsoleTest/        Sample playground and manual tests
  Onnxify.SourceGenerator/    Placeholder for future generators
  Onnxify.TorchSharp/         Placeholder for future TorchSharp APIs
  Onnxify.OperatorSchemaGenerator/  C++ helper project
third_party/
  onnx/                       ONNX schema sources
  json/                       nlohmann/json dependency
```

## Development Notes

- The main abstraction layer is built around protobuf-generated ONNX types.
- `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxTensor`, and `OnnxAttribute` provide a more convenient .NET API over raw protobuf classes.
- `Onnxify.ConsoleTest` includes examples such as loading an ONNX model and exporting a simple AlexNet-style network from TorchSharp-style weights.

## Roadmap Ideas

- complete TorchSharp integration
- generate strongly typed operator wrappers from ONNX schemas
- expand test coverage with automated round-trip and compatibility tests
- improve support for more ONNX data types and edge cases

## License

This repository is licensed under the terms of the [LICENSE](LICENSE) file.
