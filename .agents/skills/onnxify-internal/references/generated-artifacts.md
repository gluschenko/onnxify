# Onnxify Generated Artifacts

Use this reference when deciding whether to edit a file directly or change the generator that owns it.

## Generated Skill Docs

- Generator source: `src/Onnxify.AgentSkillGenerator`
- Generated outputs:
  - `.agents/skills/onnxify/references/operators/**`
  - `.agents/skills/onnxify/references/torchsharp-converters/**`

When these outputs are wrong, change the generator first, then refresh the generated markdown.

## Generated Protobuf Surface

- ONNX schema assets live under `third_party/onnx`
- Generated protobuf C# files live under `src/Onnxify/Protobuf`

Do not hand-edit protobuf output if the task is really a regeneration workflow.

## Build Artifacts

- Native and CMake build output appears under `src/out`
- `bin/` and `obj/` folders under projects are build output, not source-of-truth content

Avoid editing these directories unless the task is explicitly about the build pipeline or build output inspection.

## Skill Metadata

- Skill trigger metadata lives in each skill's `SKILL.md` frontmatter
- UI metadata lives in `agents/openai.yaml`

If a skill should trigger in new situations, update `description` in `SKILL.md`.
If the change is only about UI-facing naming or starter text, update `agents/openai.yaml`.
