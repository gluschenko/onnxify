---
name: onnxify-internal
description: "Use this skill when maintaining the Onnxify repository, package docs, and generators rather than consuming the public APIs directly. It covers repository-facing work for the Onnxify package family, including Onnxify, Onnxify.TorchSharp, Onnxify.Safetensors, Onnxify.HuggingFace, Onnxify.ProjectGenerator, Onnxify.ModelGenerator, Onnxify.ML, Onnxify.ML.TorchSharp, and Onnxify.CLI: updating README, NuGet docs, and skill content; editing solution or project wiring under src/; keeping generated skill artifacts in sync; and modifying repository generators such as Onnxify.AgentSkillGenerator, Onnxify.ProjectGenerator, and Onnxify.SourceGenerator."
---

# Onnxify Internal

## Overview

Use this skill for repository maintenance, contributor-facing docs, build/test wiring, and generated-artifact workflows in this repo.

Read [references/repo-map.md](references/repo-map.md) when you need a quick reminder of which project owns which responsibility in `src/`.
Read [references/generated-artifacts.md](references/generated-artifacts.md) when the task touches generated files, skill docs, protobuf outputs, or build artifacts.
Read [references/finding-torchsharp-porting-candidates.md](references/finding-torchsharp-porting-candidates.md) when you need to identify the highest-value missing `Onnxify.TorchSharp` operators before starting a port.
Read [references/porting-onnxscript-converters.md](references/porting-onnxscript-converters.md) when you need to port a Python-side ONNXScript Torch conversion into `Onnxify.TorchSharp`.
Read [references/validating-existing-torchsharp-converters-against-onnxscript.md](references/validating-existing-torchsharp-converters-against-onnxscript.md) only when the user explicitly asks you to validate an already covered `Onnxify.TorchSharp` exporter against the original Python exporter in `third_party/onnxscript`.
Read [references/torchsharp-operator-verification-log.md](references/torchsharp-operator-verification-log.md) when the user wants the already completed parity-validation batches for covered tensor operators or needs to continue the same running validation log.
Read [references/porting-safetensors.md](references/porting-safetensors.md) when you need to port `third_party/safetensors` into `Onnxify.Safetensors`.

For requests like "port more operators", "add 25 unsupported operators", or "what should we port next", do not jump straight into implementation from the observer table alone.
First read [references/finding-torchsharp-porting-candidates.md](references/finding-torchsharp-porting-candidates.md) to rank candidates by model impact, example pain, and available ONNX primitives.
Only after that shortlist exists should you read [references/porting-onnxscript-converters.md](references/porting-onnxscript-converters.md) and implement the selected operators.

## Quick Start

When handling an internal Onnxify maintenance task:

1. Identify the task shape: repo docs, skill/docs generation, solution or project wiring, project generator work, source-generator work, test infrastructure, or manual playground cleanup.
2. Edit the smallest owning surface first. Change generator code before changing generator output by hand.
3. Prefer `src/Onnxify.Tests` for automated validation, `src/Onnxify.Examples` for curated usage samples, and `src/Onnxify.ConsoleTest` only for manual repros.
4. Keep repository docs and skill instructions aligned when you change user-facing workflows.
5. If a change affects generated artifacts, update the generator, update or add tests, and then refresh generated outputs.

When the task is specifically about TorchSharp operator porting:

1. Start with `references/finding-torchsharp-porting-candidates.md` to choose candidates.
2. Build a shortlist from `Found = yes` and `Coverage = no` rows in `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`.
3. Prefer candidates that remove manual ONNX graph work in `src/Onnxify.Examples` or unblock existing model families in the repo.
4. Then use `references/porting-onnxscript-converters.md` to mirror the chosen ONNXScript converters into `Onnxify.TorchSharp`.
5. Add focused smoke tests in `src/Onnxify.Tests` before moving to the next batch.

When the task is specifically about validating an existing TorchSharp exporter against ONNXScript:

This workflow is opt-in.
Do not start it just because the task mentions Torch ops, coverage, porting, or ONNXScript.
Use it only when the user explicitly asks to validate, compare, audit, or check parity of already existing `Onnxify.TorchSharp` operators.

1. Start with `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md` to get the exact Torch op spelling.
2. Confirm support in `.agents/skills/onnxify/references/torchsharp-converters/index.md`.
3. Open the linked generated converter page to get the exact C# signature and source file.
4. Trace the real exporter implementation in `src/Onnxify.TorchSharp`, including semantic helper methods it calls.
5. Then use `references/validating-existing-torchsharp-converters-against-onnxscript.md` to compare that behavior against the Python exporter in `third_party/onnxscript`.
6. If the task is about continuing or reviewing the existing documented validation batches, open `references/torchsharp-operator-verification-log.md` before starting the next wave and keep extending its combined operator table.

## Core Principles

- Treat repository maintenance as a separate concern from library feature work. Use `$onnxify` for API semantics and `$onnxify-internal` for repo structure, docs, skills, and generators.
- Change source-of-truth files instead of patching generated outputs directly whenever a generator already owns that output.
- Keep skill install instructions, repo layout descriptions, and actual folder structure synchronized.
- Keep package-facing docs under `.docs` organized by purpose, and keep the documented layout aligned with the actual NuGet packaging flow.
- Keep placeholder or future-facing projects small unless the user explicitly asks to expand them.
- Prefer focused, repo-consistent edits over broad reorganizations. This repo is still evolving.

## Ownership Rules

- Put library semantics in `src/Onnxify` and `src/Onnxify.TorchSharp`, not in docs-only or sample-only projects.
- Keep core ONNX model I/O centered in `src/Onnxify/OnnxModel.cs`. File APIs should remain consistent with the stream APIs: `FromFile(...)` / `FromFileAsync(...)` configure path-relative external data, while `FromStream(...)` / `FromStreamAsync(...)` rely on caller-provided `OnnxModelBaseOptions.DataLocation` when external tensor data needs a base path.
- Put repository-facing generation logic in `src/Onnxify.AgentSkillGenerator`, `src/Onnxify.ProjectGenerator`, or `src/Onnxify.SourceGenerator`, depending on ownership.
- Put durable automated checks in `src/Onnxify.Tests`.
- Put walkthroughs, install notes, and contributor-facing explanations in `README.md` or skill docs under `.agents/skills`.
- Put NuGet package readmes in `.docs/nuget/<PackageName>.md` and NuGet release notes in `.docs/nuget/release-notes/<PackageName>.md`.
- Treat `src/out` as build output, not as an editing target, unless the task is explicitly about the native build pipeline.

## Docs Layout

- Treat `.docs/nuget` as the source of truth for package-facing Markdown that ships with or describes NuGet packages.
- Keep one package README per published package at `.docs/nuget/<PackageName>.md`.
- Keep one release-notes file per published package at `.docs/nuget/release-notes/<PackageName>.md`.
- Treat `.docs/nuget/release-notes` as the home for NuGet package release notes in this repository, and keep each Markdown filename aligned with the corresponding package name.
- When changing package metadata conventions, update `src/Directory.Build.props` and `.docs` together so packaging behavior and repository docs do not drift apart.
- If a `.docs` subfolder becomes obsolete after a metadata-layout change, remove it or document why it still exists.

## Code Style

- Prefer explicit multi-line wrapping for long or intentionally expanded method declarations and method calls.
- In this repo, when a method declaration or invocation is split across multiple lines, put the closing parenthesis on its own line instead of attaching it to the last argument line.
- Apply the same rule to constructors and other argument lists when the call or declaration is formatted vertically.
- Name constants in C++-style uppercase snake case rather than PascalCase.

Preferred declaration style:

```csharp
public void SomeMethod(
    int a,
    int b,
    int c
)
{
    // ...
}
```

Preferred invocation style:

```csharp
SomeMethod(
    a: 1,
    b: 2,
    c: 4
);
```

Avoid this style when using vertical argument layout:

```csharp
public void SomeMethod(
    int a,
    int b,
    int c)
{
    // ...
}

SomeMethod(
    a,
    b,
    c);
```

Preferred constant naming:

```csharp
private const long INLINE_TENSOR_ELEMENT_THRESHOLD = 20L;
```

Avoid this constant naming style:

```csharp
private const long InlineTensorElementThreshold = 20L;
```

## Common Task Mapping

- Update the public repo description or install instructions: start with `README.md`.
- Add or refine Codex guidance for library users: start with `.agents/skills/onnxify`.
- Add or refine Codex guidance for repo maintainers: start with `.agents/skills/onnxify-internal`.
- Find the next best TorchSharp operator to port: start with `references/finding-torchsharp-porting-candidates.md`, then inspect `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`.
- Port an ONNXScript Torch conversion into `Onnxify.TorchSharp`: if the user did not name the operator explicitly, first use `references/finding-torchsharp-porting-candidates.md`; then use `references/porting-onnxscript-converters.md`.
- Validate an existing `Onnxify.TorchSharp` exporter against ONNXScript: only do this when the user explicitly asks for validation/parity checking of already supported operators; then start with `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md` and use `references/validating-existing-torchsharp-converters-against-onnxscript.md`.
- Need the already written parity-validation batch reports for covered tensor operators: open `references/torchsharp-operator-verification-log.md`.
- For requests like "port 10/25/50 more operators", treat candidate selection as a required first phase, not an optional nicety.
- Port `third_party/safetensors` into `Onnxify.Safetensors`: start with `third_party/safetensors/safetensors/src`, then use `references/porting-safetensors.md`.
- Fix generated operator or TorchSharp converter docs: start with `src/Onnxify.AgentSkillGenerator`, then refresh `.agents/skills/onnxify/references`.
- Adjust the generated-project output or C# scaffolding shape: start with `src/Onnxify.ProjectGenerator` and `src/Onnxify.Tests/OnnxProjectGeneratorTests.cs`.
- Adjust source-generation inputs or linked schema models: inspect `src/Onnxify.SourceGenerator` and any linked model files before changing consuming projects.
- Reorganize solution structure or project references: inspect `src/Onnxify.slnx` and the relevant `*.csproj` files together.
- Need a manual repro or scratch integration: use `src/Onnxify.ConsoleTest`, but do not treat it as the final validation surface.
- Need a polished sample or end-to-end usage example: use `src/Onnxify.Examples`.
- Need to touch the native ONNX schema extraction pipeline: inspect `src/Onnxify.OperatorSchemaGenerator` and related CMake files.

## Generated Artifact Workflow

- Do not hand-edit `.agents/skills/onnxify/references/operators/*` or `.agents/skills/onnxify/references/torchsharp-converters/*` if the change should come from `src/Onnxify.AgentSkillGenerator`.
- If you change generator behavior, add or update tests in `src/Onnxify.Tests/AgentSkillGeneratorTests.cs`.
- If you change project generation behavior, add or update tests in `src/Onnxify.Tests/OnnxProjectGeneratorTests.cs`.
- If the task is only about static prose in `SKILL.md`, `README.md`, or `agents/openai.yaml`, edit those files directly.
- If a new user-visible feature changes what a NuGet package can do, update the matching `.docs/nuget/release-notes/<PackageName>.md` file in the same task instead of leaving release notes for later.
- If one feature affects multiple published packages, update every affected package's release-notes file rather than recording the change in only one place.

## Validation Checklist

- Did the edit touch the true owning project rather than only a downstream generated file?
- Did docs, skill instructions, and actual commands still agree after the change?
- If package behavior or package-facing metadata changed, did you update the relevant `.docs/nuget` README and `.docs/nuget/release-notes` files?
- If generated output changed, did you update generator code and the closest generator tests?
- If solution or project wiring changed, did you inspect both `src/Onnxify.slnx` and the affected `*.csproj` files?
- If model I/O changed, did you cover both file and stream surfaces where relevant, including async overloads and `DataLocation` behavior for external data?
- If the task changed only playground or sample code, did you avoid implying that production behavior changed too?
- If the task introduced a new internal workflow, did you document it in the most discoverable place?
