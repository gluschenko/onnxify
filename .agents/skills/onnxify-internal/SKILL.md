---
name: onnxify-internal
description: "Use this skill when maintaining the Onnxify repository itself rather than implementing library API behavior: updating README or contributor docs, editing solution or project wiring under src/, changing .agents/skills content, keeping generated skill artifacts in sync, or modifying repository-facing generators such as Onnxify.AgentSkillGenerator, Onnxify.ProjectGenerator, Onnxify.SourceGenerator, and related test or example infrastructure."
---

# Onnxify Internal

## Overview

Use this skill for repository maintenance, contributor-facing docs, build/test wiring, and generated-artifact workflows in this repo.

Read [references/repo-map.md](references/repo-map.md) when you need a quick reminder of which project owns which responsibility in `src/`.
Read [references/generated-artifacts.md](references/generated-artifacts.md) when the task touches generated files, skill docs, protobuf outputs, or build artifacts.

## Quick Start

When handling an internal Onnxify maintenance task:

1. Identify the task shape: repo docs, skill/docs generation, solution or project wiring, project generator work, source-generator work, test infrastructure, or manual playground cleanup.
2. Edit the smallest owning surface first. Change generator code before changing generator output by hand.
3. Prefer `src/Onnxify.Tests` for automated validation, `src/Onnxify.Examples` for curated usage samples, and `src/Onnxify.ConsoleTest` only for manual repros.
4. Keep repository docs and skill instructions aligned when you change user-facing workflows.
5. If a change affects generated artifacts, update the generator, update or add tests, and then refresh generated outputs.

## Core Principles

- Treat repository maintenance as a separate concern from library feature work. Use `$onnxify` for API semantics and `$onnxify-internal` for repo structure, docs, skills, and generators.
- Change source-of-truth files instead of patching generated outputs directly whenever a generator already owns that output.
- Keep skill install instructions, repo layout descriptions, and actual folder structure synchronized.
- Keep placeholder or future-facing projects small unless the user explicitly asks to expand them.
- Prefer focused, repo-consistent edits over broad reorganizations. This repo is still evolving.

## Ownership Rules

- Put library semantics in `src/Onnxify` and `src/Onnxify.TorchSharp`, not in docs-only or sample-only projects.
- Put repository-facing generation logic in `src/Onnxify.AgentSkillGenerator`, `src/Onnxify.ProjectGenerator`, or `src/Onnxify.SourceGenerator`, depending on ownership.
- Put durable automated checks in `src/Onnxify.Tests`.
- Put walkthroughs, install notes, and contributor-facing explanations in `README.md` or skill docs under `.agents/skills`.
- Treat `src/out` as build output, not as an editing target, unless the task is explicitly about the native build pipeline.

## Common Task Mapping

- Update the public repo description or install instructions: start with `README.md`.
- Add or refine Codex guidance for library users: start with `.agents/skills/onnxify`.
- Add or refine Codex guidance for repo maintainers: start with `.agents/skills/onnxify-internal`.
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

## Validation Checklist

- Did the edit touch the true owning project rather than only a downstream generated file?
- Did docs, skill instructions, and actual commands still agree after the change?
- If generated output changed, did you update generator code and the closest generator tests?
- If solution or project wiring changed, did you inspect both `src/Onnxify.slnx` and the affected `*.csproj` files?
- If the task changed only playground or sample code, did you avoid implying that production behavior changed too?
- If the task introduced a new internal workflow, did you document it in the most discoverable place?
