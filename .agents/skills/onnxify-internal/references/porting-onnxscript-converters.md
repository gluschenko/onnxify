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
- `Onnxify.TorchSharp coverage` means `Onnxify.TorchSharp` already declares support for that ONNXScript operator through `[TorchOp(...)]`.
- `Onnxify.ModelGenerator coverage` means the reverse ONNX-to-TorchSharp ModelGenerator path declares support through `[TorchSharpOp(...)]`; it is useful context, but it does not mean the TorchSharp-to-ONNX exporter exists.
- The best porting targets are usually rows where `Found` is checked and `Onnxify.TorchSharp coverage` is missing.

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

Do not stop at the exporter implementation alone.
You also need to understand how ONNXScript expects that exporter to behave from the Python tests that cover it.

## 3A. Find The Python Tests For The Operator

After locating the Python exporter, search `third_party/onnxscript` for tests that cover the same Torch op or nearby conversion path.

Practical searches usually start with:

```powershell
rg -n "aten::addmm|addmm" third_party\onnxscript
```

Use those tests to infer:

- the behavior the exporter is expected to preserve
- the kinds of inputs the operator accepts
- the shapes and value patterns the operator returns
- important overload, dtype, broadcasting, rank, or tuple-output expectations
- edge cases that the exporter is supposed to reject or normalize

Treat the Python tests as behavior evidence, not as C# code templates.
The goal is to understand the operator contract, not to mechanically translate Python tests line-for-line.

## 3B. Convert That Understanding Into C# Test Intent

Before adjusting `Onnxify.TorchSharp`, decide what behavior must be protected in:

- `src/Onnxify.Tests`

From the ONNXScript exporter plus its tests, extract the smallest high-value C# test cases that cover:

- representative successful inputs
- important output shape and output value expectations
- overload-specific semantics
- special branches such as scalar-vs-tensor, `keepdim`, `beta == 0`, or tuple outputs
- explicit unsupported cases when the C# exporter intentionally supports a narrower surface

These C# tests should encode the same semantic expectations, but they must remain idiomatic for this repo.
Do not mirror Python-specific helper structure, naming style, or language-specific mechanics when those clash with the C# test style or the local export architecture.

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

In practice, the implementation order should usually be:

1. understand the Python exporter
2. understand the Python tests for that exporter
3. implement focused C# tests in `Onnxify.Tests` that express the same behavioral contract
4. bring the `Onnxify.TorchSharp` exporter into alignment with those tests

Prefer this test-first or test-guided flow over writing the C# exporter in isolation and only validating it afterward.

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
The tests should reflect operator semantics learned from ONNXScript and its tests, but they should not be a line-by-line translation of Python tests.
Avoid dragging Python-specific conventions into C# when they conflict with local helper patterns, naming, or runtime assumptions.

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

- Prefer candidates with `Found = yes` and `Onnxify.TorchSharp coverage = no` before adding brand-new TorchSharp surface area.
- Match the ONNXScript behavior semantically, not line-for-line. Reuse existing C# helpers where the repo already has them.
- Keep the `TorchOp` names aligned with ONNXScript operator names, including overload suffixes when relevant.
- If the Python version handles more cases than the current C# export layer can support safely, implement the safe subset first and fail clearly for the rest.
- Use ONNXScript tests to learn the operator contract, then restate that contract as idiomatic `Onnxify.Tests` coverage instead of transliterating Python test code.
