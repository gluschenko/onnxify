# Porting Torch Converters From ONNXScript

Use this workflow when you want to port a Python-side ONNXScript conversion into `Onnxify.TorchSharp`.

## Goal

Start from the ONNXScript Torch registry in `third_party/onnxscript`, find an operator that TorchSharp exposes but `Onnxify.TorchSharp` does not yet cover, and implement an equivalent export path in C#.

## 1. Refresh The Coverage Report

Run the observer project first so you are working from a fresh snapshot:

- `dotnet run --project src/Onnxify.TorchSharp.Observer`

By default it rewrites:

- `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`

## 2. Choose A Good Porting Candidate

Open `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md` and inspect the table.

- `Found` means the observer found a corresponding TorchSharp API or module surface.
- `Coverage` means `Onnxify.TorchSharp` already declares support for that ONNXScript operator through `[TorchOp(...)]`.
- The best porting targets are usually rows where `Found` is checked and `Coverage` is empty.

That combination usually means TorchSharp already exposes the operator, but `Onnxify.TorchSharp` still lacks a converter.

## 3. Find The Python Reference In ONNXScript

The ONNXScript Torch registry lives under:

- `third_party/onnxscript/onnxscript/function_libs/torch_lib/ops`

Search for the exact operator name from the report, for example:

- `rg -n '@torch_op\\(\"aten::gelu\"' third_party/onnxscript/onnxscript/function_libs/torch_lib/ops`

The decorator arguments on `@torch_op(...)` are the quickest way to find the owning Python conversion. The implementation often lives in one of these files:

- `core.py`
- `nn.py`
- `linalg.py`
- `fft.py`
- `special.py`
- `vision.py`

Read the Python converter closely and identify:

- which ONNX operators it emits
- which attributes or constants it normalizes
- whether it handles multiple overloads for the same Torch op
- whether it depends on inference-only assumptions or unsupported edge cases

## 4. Mirror The Conversion In `Onnxify.TorchSharp`

Most direct TorchSharp exports live in:

- `src/Onnxify.TorchSharp/TorchModuleExtensions.cs`

When possible, follow an existing nearby exporter with similar semantics instead of inventing a new pattern.

Typical implementation work:

- add or extend the relevant `Export(...)` overload
- normalize TorchSharp options into ONNX attributes
- materialize weights, biases, and constants through existing helpers
- keep unsupported semantics explicit with `NotSupportedException` or `NotImplementedException`

If you are porting a module-backed operator, also make sure the public dispatch path still reaches your new exporter consistently.

## 5. Decorate The Converter With `TorchOp`

The observer and generated converter docs discover coverage from `TorchOpAttribute`.

Always decorate the conversion method with the matching ONNXScript operator name, for example:

```csharp
[TorchOp("aten::gelu")]
```

The attribute type lives in:

- `src/Onnxify.TorchSharp/TorchOpAttribute.cs`

If one exporter covers several ONNXScript spellings or overload names, add multiple `[TorchOp(...)]` attributes.

## 6. Add Smoke Tests In `Onnxify.Tests`

Prefer adding focused smoke coverage in:

- `src/Onnxify.Tests`

Good tests for ported converters usually assert:

- the exporter does not throw for a representative module/input
- the expected ONNX node type is emitted
- important attributes, tensor names, or graph wiring are preserved

Keep these tests lightweight. A small structural graph assertion is usually better than a large end-to-end fixture.

## 7. Refresh The Generated Skill Docs

After changing TorchSharp converter coverage, regenerate the skill artifacts:

- `dotnet run --project src/Onnxify.AgentSkillGenerator`

This refreshes the generated TorchSharp converter references under:

- `.agents/skills/onnxify/references/torchsharp-converters`

## 8. Sanity-Check The Workflow

Before finishing:

- rerun `src/Onnxify.TorchSharp.Observer` if you want to confirm the new operator now shows coverage
- run the focused `Onnxify.Tests` coverage you added
- inspect the regenerated converter docs if the new exporter should now appear there

## Heuristics

- Prefer candidates with `Found = yes` and `Coverage = no` before adding brand-new TorchSharp surface area.
- Match the ONNXScript behavior semantically, not line-for-line. Reuse existing C# helpers where the repo already has them.
- Keep the `TorchOp` names aligned with ONNXScript operator names, including overload suffixes when relevant.
- If the Python version handles more cases than the current C# export layer can support safely, implement the safe subset first and fail clearly for the rest.
