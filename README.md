# Onnxify

Onnxify is an experimental .NET library for reading, inspecting, and writing ONNX models.

Machine learning workflows are often difficult not because models are impossible to run, but because they are difficult to understand, inspect, adapt, and carry from one environment to another. A lot of useful work happens in that space between research and production, where people need clarity, control, and confidence rather than another opaque black box. Onnxify exists for that middle ground.

The idea behind this repository is simple: models should be easier to work with, easier to reason about, and easier to integrate into real development workflows. If ONNX is meant to be a common language for models, then the tools around it should help people move faster, make smaller changes safely, and build their own workflows without unnecessary friction. That is the direction Onnxify is trying to push.

## NuGet Packages

The repository currently implements the following NuGet packages. Package-specific instructions live in [`.docs/nuget`](.docs/nuget).

| Package | Instructions |
| --- | --- |
| `Onnxify` | [`.docs/nuget/Onnxify.md`](.docs/nuget/Onnxify.md) |
| `Onnxify.TorchSharp` | [`.docs/nuget/Onnxify.TorchSharp.md`](.docs/nuget/Onnxify.TorchSharp.md) |
| `Onnxify.ML` | [`.docs/nuget/Onnxify.ML.md`](.docs/nuget/Onnxify.ML.md) |
| `Onnxify.ML.TorchSharp` | [`.docs/nuget/Onnxify.ML.TorchSharp.md`](.docs/nuget/Onnxify.ML.TorchSharp.md) |
| `Onnxify.ProjectGenerator` | [`.docs/nuget/Onnxify.ProjectGenerator.md`](.docs/nuget/Onnxify.ProjectGenerator.md) |
| `Onnxify.ModelGenerator` | [`.docs/nuget/Onnxify.ModelGenerator.md`](.docs/nuget/Onnxify.ModelGenerator.md) |
| `Onnxify.Safetensors` | [`.docs/nuget/Onnxify.Safetensors.md`](.docs/nuget/Onnxify.Safetensors.md) |
| `Onnxify.CLI` | [`.docs/nuget/Onnxify.CLI.md`](.docs/nuget/Onnxify.CLI.md) |

## TorchSharp Operator Porting

`Onnxify.TorchSharp` ports TorchSharp operators by translating Torch modules and tensor-style ops into explicit ONNX graph construction with `Onnxify`, including nodes, attributes, weights, and constants, instead of relying on runtime tracing.

To inspect the current coverage and the highest-value gaps, see [`src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`](src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md).

## Requirements

- .NET 8 SDK + .NET 10 SDK
- Windows 11 or Linux-based system
- NuGet packages are cross-platform for consumer projects

## Getting Started

Clone the repository and build the solution.

Windows:

```powershell
git clone --recurse-submodules https://github.com/gluschenko/onnxify.git
cd onnxify
dotnet build src\Onnxify.slnx
```

Linux:

```bash
git clone --recurse-submodules https://github.com/gluschenko/onnxify.git
cd onnxify
dotnet build src/Onnxify.slnx
```

To pack and install the local `Onnxify.CLI` tool from this repository:

Windows:

```powershell
.\install-onnxify-cli.ps1
```

Linux:

```bash
chmod +x ./install-onnxify-cli.sh
./install-onnxify-cli.sh
```

To install or refresh the bundled Codex skills from this repository:

Windows:

```powershell
.\install-onnxify-skills.ps1
```

Linux:

```bash
chmod +x ./install-onnxify-skills.sh
./install-onnxify-skills.sh
```

Both install scripts support help output:

Windows:

```powershell
.\install-onnxify-cli.ps1 -Help
.\install-onnxify-skills.ps1 -Help
```

Linux:

```bash
./install-onnxify-cli.sh --help
./install-onnxify-skills.sh --help
```

## Install the Codex Skill

This section is optional. If you only want to consume the NuGet packages in your own .NET project, you do not need the Codex skill.

If you use Codex and want repository-specific help for `Onnxify`, `Onnxify.TorchSharp`, and the related package family, you can install the bundled `onnxify` skill directly from GitHub without cloning the repository.

Windows:

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }

py -3 "$codexHome\skills\.system\skill-installer\scripts\install-skill-from-github.py" `
  --repo gluschenko/onnxify `
  --path .agents/skills/onnxify
```

Linux:

```bash
codex_home="${CODEX_HOME:-$HOME/.codex}"

python3 "$codex_home/skills/.system/skill-installer/scripts/install-skill-from-github.py" \
  --repo gluschenko/onnxify \
  --path .agents/skills/onnxify
```

You can also install it by URL instead of `--repo` and `--path`.

Windows:

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }

py -3 "$codexHome\skills\.system\skill-installer\scripts\install-skill-from-github.py" `
  --url "https://github.com/gluschenko/onnxify/tree/main/.agents/skills/onnxify"
```

Linux:

```bash
codex_home="${CODEX_HOME:-$HOME/.codex}"

python3 "$codex_home/skills/.system/skill-installer/scripts/install-skill-from-github.py" \
  --url "https://github.com/gluschenko/onnxify/tree/main/.agents/skills/onnxify"
```

If you already cloned this repository and want to install both bundled skills from the local checkout, use the `install-onnxify-skills.ps1` or `install-onnxify-skills.sh` scripts shown in [Getting Started](#getting-started).

Restart Codex after installation so it picks up the new or refreshed skill files.

## TODO

- [x] OnnxGraph rework
- [x] SourceGenerator: operator type annotations
- [ ] SourceGenerator: fully-typed operator Input/Output fields (OneOf?)
- [ ] Async I/O ops
- [ ] Graph edges in a single collection (or in two for placeholders)
- [ ] Graph manipulations: add nodes, remove nodes, replace nodes
- [ ] Graph cyclicity validation
- [x] CLI for agents and humans (to explore ONNX files)
- [x] Project generator generates operator nodes
- [x] Parse pytorch\torch\onnx\_internal\torchscript_exporter (create MD with support status)
- [x] Generate agent skills from operator-schema.json
- [x] ToString for OnnxModel, OnnxNode, OnnxxTensor, etc (recursive?)
- [x] OnnxDataProvider, SafetensorsDataProvider, BaseDataProvider...
- [x] Agent skills for Export implementation on Torch modules
- [x] Allow to add or remove OnnxModel meta (training info, imports, producer, version)

## License

This repository is licensed under the terms of the [LICENSE](LICENSE) file.
