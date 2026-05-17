# Validating Existing TorchSharp Converters Against ONNXScript

Use this workflow when you need to validate an existing `Onnxify.TorchSharp` exporter against the original Python-side exporter in `third_party/onnxscript`.

## Goal

Start from the repo's generated coverage and converter references, identify the exact `Onnxify.TorchSharp` export surface for a Torch op, trace the real C# implementation including helper calls, then compare it against the ONNXScript implementation for semantic parity.

This workflow is for validation of already covered operators, not for choosing missing operators to port.
If the operator is not already exposed by `Onnxify.TorchSharp`, treat that as unsupported and stop the validation there.

## 1. Start With The Observer Report

Open:

- `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`

Use it to confirm the exact Torch op spelling and the discovered TorchSharp surface.

Read the row this way:

- `ONNXScript operator`: the exact Torch op name to validate, such as `aten::addmm` or `aten::layer_norm`
- `TorchSharp module`: the TorchSharp API or module surface the observer matched
- `Coverage`: whether `Onnxify.TorchSharp` currently claims support through `[TorchOp(...)]`

This step matters because validation should always use the exact operator spelling from the report, including overload suffixes such as:

- `aten::sum.dim_IntList`
- `aten::where.ScalarOther`
- `aten::lerp.Tensor`

## 2. Confirm Whether `Onnxify.TorchSharp` Actually Supports It

Open:

- `.agents/skills/onnxify/references/torchsharp-converters/index.md`

Use the exact operator name from the observer report.

Search for that spelling in the `Torch ops` column or with `rg`, for example:

```powershell
rg -n "aten::addmm" .agents\skills\onnxify\references\torchsharp-converters\index.md
```

Interpret the result strictly:

- if the operator is absent from `index.md`, treat it as not supported by `Onnxify.TorchSharp`
- if the operator is present, open the linked converter page for the exact signature

Do not infer support from a nearby alias or a related overload unless the exact op spelling appears in the generated converter references.

## 3. Read The Generated Converter Signature

Open the linked converter page under:

- `.agents/skills/onnxify/references/torchsharp-converters/torch-ops`
- or `.agents/skills/onnxify/references/torchsharp-converters/composites` for composite cases

These generated pages tell you:

- the C# method name, such as `ExportAddMM`
- the declaring type, such as `TorchTensorOperatorExtensions` or `TorchModuleExtensions`
- the full signature
- the source file
- all Torch ops declared on that method

For example, a typical page gives you a signature like:

- `Onnxify.TorchSharp.TorchTensorOperatorExtensions.ExportAddMM(this OnnxGraph graph, IOnnxGraphEdge input, IOnnxGraphEdge mat1, IOnnxGraphEdge mat2, float beta, float alpha) -> IOnnxGraphEdge`

Treat this generated page as the bridge between the observer row and the real source code.

## 4. Locate The Real C# Exporter In Source

Go to the source file named on the generated converter page, usually one of:

- `src/Onnxify.TorchSharp/TorchTensorOperatorExtensions.cs`
- `src/Onnxify.TorchSharp/TorchModuleExtensions.cs`

Search by method name first, not just by Torch op string. For example:

```powershell
rg -n "ExportAddMM|\[TorchOp\(\" src\Onnxify.TorchSharp
```

Then read:

- the exporter method itself
- every helper that materially affects semantics
- any shared helper that builds constants, casts, reshapes, reductions, tuple outputs, or fallback branches

For validation purposes, do not stop at the top-level method if it immediately delegates into helpers such as:

- `ExportClampCore`
- `ExportTruthReduction`
- `ScaleLikeIfNeeded`
- `ExportReduceNode`
- `ExportExtremumByDim`

You are validating emitted behavior, not just the public signature.

## 5. Trace Helper Methods Transitively

When the exporter calls helpers, follow them until you understand the real ONNX graph behavior.

Typical things hidden in helpers:

- scalar constant typing
- `Cast` insertion
- shape normalization
- `keepdim` handling
- rank checks
- dtype restrictions
- special-case branches for scalar tensors
- explicit unsupported cases via `NotSupportedException` or `NotImplementedException`

A good practical pattern is:

1. read the main exporter
2. list every helper that changes semantics
3. open each helper immediately
4. stop only when the remaining calls are plain `graph.Add`, `graph.Mul`, `graph.Gemm`, `graph.ReduceSum`, and similar direct node emissions

## 6. Find The Original ONNXScript Exporter

The ONNXScript Torch registry is in:

- `third_party/onnxscript/onnxscript/function_libs/torch_lib/ops`

Search by exact Torch op name from the observer report, for example:

```powershell
rg -n '@torch_op\("aten::addmm"|@torch_op\(\("aten::addmm"' third_party\onnxscript\onnxscript\function_libs\torch_lib\ops
```

The main files are usually:

- `core.py`
- `nn.py`
- `linalg.py`
- `fft.py`
- `special.py`
- `vision.py`

Read the Python function decorated with the exact op name. If the decorator covers several ops together, validate only the branch or semantics relevant to your chosen op.

## 7. Compare Semantics, Not Just Node Names

The comparison should be semantic and structured.

Check at least these dimensions:

- input and output arity
- overload shape
- default parameter values
- optional parameter behavior
- scalar-vs-tensor overload differences
- dtype assumptions and casts
- shape reshaping, unsqueeze, squeeze, transpose, and broadcast behavior
- reduction axes and `keepdim` behavior
- branch conditions such as `beta == 0`, `alpha == 1`, scalar-rank special cases, or empty-tensor handling
- whether unsupported cases are rejected explicitly in C#
- whether ONNXScript handles more cases than `Onnxify.TorchSharp`

Focus on the emitted graph contract, for example:

- does `addmm` use `Gemm` with the same `alpha` and `beta` semantics
- does `addr` preserve the `beta == 0` behavior that avoids propagating `self`
- does `lerp` follow the same two-branch numerically stable formulation
- does `max.dim` or `topk` return both values and indices in the same shape conventions

Do not mark two exporters equivalent just because both eventually use the same ONNX primitive.

## 8. Record The Validation Outcome Clearly

A useful validation result usually fits one of these buckets:

- equivalent: C# behavior appears semantically aligned with ONNXScript for the covered surface
- equivalent with explicit constraints: C# matches, but only for a narrower safe subset and rejects the rest clearly
- partial mismatch: the main path matches, but one or more branches, defaults, or shape rules differ
- unsupported despite nearby coverage: the generated docs may show a related exporter, but the exact op or overload is not actually covered

If you find a mismatch, write down:

- the exact Torch op name
- the exact C# signature validated
- the Python function and file in `third_party/onnxscript`
- the concrete semantic difference
- whether the difference is acceptable repo policy, an intentional subset, or a bug

## 9. Suggested Search Pattern

For a single operator, a practical terminal workflow is:

```powershell
rg -n "aten::addmm" src\Onnxify.TorchSharp.Observer\torchsharp-operator-report.md
rg -n "aten::addmm" .agents\skills\onnxify\references\torchsharp-converters\index.md
rg -n "ExportAddMM|\[TorchOp\(\" src\Onnxify.TorchSharp
rg -n '@torch_op\("aten::addmm"|@torch_op\(\("aten::addmm"' third_party\onnxscript\onnxscript\function_libs\torch_lib\ops
```

Then open:

- the generated converter page
- the C# exporter source
- every semantic helper it calls
- the ONNXScript Python implementation

## 10. Common Pitfalls

- Do not validate against a guessed Torch op spelling. Use the exact observer row.
- Do not treat `Coverage = yes` in the observer report as proof of semantic parity. It only proves `[TorchOp(...)]` coverage exists.
- Do not stop at the generated converter page. It gives the signature, not the full behavior.
- Do not ignore helper methods that change dtype, rank, branching, or tuple outputs.
- Do not compare only the first ONNX node emitted. Many converters differ in surrounding casts, reshapes, or reduction behavior.
- Do not treat missing presence in `index.md` as a docs bug by default. For this workflow, absence means unsupported until proven otherwise.

## Heuristics

- Prefer validating the exact overload named in the observer report before collapsing several overloads together.
- Prefer method-name search from the generated converter page over broad repo-wide Torch op search once you know the signature.
- Treat helper tracing as required whenever the exporter is more than a one-node wrapper.
- When ONNXScript supports a wider surface than C#, first decide whether the C# subset is intentionally narrower and explicit or silently incomplete.
- If the validation uncovers a real mismatch, add or update focused tests in `src/Onnxify.Tests` close to the exporter you inspected.
