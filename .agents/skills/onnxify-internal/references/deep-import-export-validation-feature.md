# Validating Existing Deep Import/Export Operators Against ONNXScript

Use this workflow when you need to validate an existing `Onnxify.TorchSharp` exporter or `Onnxify.ModelGenerator` TorchModule importer against the original Python-side exporter in `third_party/onnxscript`.
This workflow validates already-added operator coverage in the two complementary feature paths:

- `deep export`: TorchSharp execution or module structure exported into an ONNX graph by `Onnxify.TorchSharp`.
- `deep import`: ONNX graph structure imported into generated TorchSharp module code by `Onnxify.ModelGenerator`.

Use it only when the user explicitly asks for validation, parity checking, comparison, or audit of already existing `deep export` or `deep import` operators.
Do not trigger this workflow automatically for generic operator-porting, coverage, or ONNXScript tasks.

## Goal

Start from the repo's generated coverage and converter references, identify the exact `Onnxify.TorchSharp` export surface or `Onnxify.ModelGenerator` import surface for a Torch op, trace the real C# implementation including helper calls, then compare it against the ONNXScript implementation for semantic parity.

This workflow is for validation of already covered operators, not for choosing missing operators to port.
If the operator is not already exposed by the feature path being validated, treat that side as unsupported and stop validation for that side.
This workflow also includes reviewing the existing C# unit tests that cover the exporter or importer, because semantic parity is not fully validated unless the current test surface is checked too.
When the operator participates in both paths, validation should include or recommend `deep roundtrip tests` so `deep export` and `deep import` stay maximally complementary rather than independently correct in isolation.

## 1. Start With The Observer Report

Open:

- `TORCH_OPERATOR_COVERAGE.md`

Use it to confirm the exact Torch op spelling and the discovered TorchSharp surface.

Read the row this way:

- `ONNXScript operator`: the exact Torch op name to validate, such as `aten::addmm` or `aten::layer_norm`
- `TorchSharp module`: the TorchSharp API or module surface the observer matched
- `Onnxify.TorchSharp coverage`: whether `Onnxify.TorchSharp` currently claims support through `[TorchOp(...)]`
- `Onnxify.ModelGenerator coverage`: whether `Onnxify.ModelGenerator` currently claims reverse TorchModule support through `[TorchSharpOp(...)]`

This step matters because validation should always use the exact operator spelling from the report, including overload suffixes such as:

- `aten::sum.dim_IntList`
- `aten::where.ScalarOther`
- `aten::lerp.Tensor`

## 2. Confirm Which Feature Path Supports It

For `deep export`, open:

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

For `deep import`, use the observer report's `Onnxify.ModelGenerator coverage` column and the generated operator references under:

- `.agents/skills/onnxify/references/operators`

Then inspect the actual ModelGenerator converter source before treating coverage as semantically valid.
Report-visible support means a `[TorchSharpOp(...)]` mapping exists; it does not prove the ONNX-to-TorchSharp import semantics are equivalent.

## 3. Read The Generated Converter Or Operator Signature

For `deep export`, open the linked converter page under:

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

For `deep import`, use the generated operator references and source search to find the owning ModelGenerator converter:

- `src/Onnxify.ModelGenerator/Services/TorchModuleInlineOperators`
- `src/Onnxify.ModelGenerator/Services/TorchModuleOperators`

The important bridge is the `[TorchSharpOp(...)]` declaration plus the converter class that emits the generated TorchSharp code for the ONNX operator or graph pattern.

## 4. Locate The Real C# Implementation In Source

For `deep export`, go to the source file named on the generated converter page, usually one of:

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

For `deep import`, locate the ModelGenerator converter class and read the generated C# expression or module emission path:

- inline converter `Emit(...)` implementations
- module converter pattern matching and load statements
- `TorchModulePrinter` helper calls that affect source layout, rank, dtype, padding, pooling, or initializer handling

You are validating the generated TorchSharp behavior, not just the presence of an ONNX op name in a registry.

## 5. Trace Helper Methods Transitively

When the exporter or importer calls helpers, follow them until you understand the real emitted behavior.

Typical things hidden in helpers:

- scalar constant typing
- `Cast` insertion
- shape normalization
- `keepdim` handling
- rank checks
- dtype restrictions
- special-case branches for scalar tensors
- explicit unsupported cases via `NotSupportedException` or `NotImplementedException`
- generated state loading and buffer/parameter decisions for `deep import`

A good practical pattern is:

1. read the main exporter or importer
2. list every helper that changes semantics
3. open each helper immediately
4. stop only when the remaining calls are plain direct node emissions, simple generated-code formatting, or direct TorchSharp calls whose semantics are already clear

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
For `deep import`, read the same function in reverse: it explains which ONNX operator or pattern normally represents the Torch semantics you are reconstructing.

## 6A. Find The Python Tests For That Exporter

Do not validate against the Python exporter implementation alone.
You also need to inspect the ONNXScript tests that cover the same operator or conversion path.

Start with a broad search in `third_party/onnxscript`, for example:

```powershell
rg -n "aten::addmm|addmm" third_party\onnxscript
```

Use the tests to infer the practical behavior contract of the operator, including:

- what kinds of inputs the exporter is expected to accept
- what kinds of outputs it returns
- what shapes, dtypes, ranks, tuple members, or broadcasting rules matter
- which branches are important enough to be locked down in tests
- which edge cases are intentionally unsupported or normalized

The Python tests are especially useful when the exporter implementation is compact but hides important behavior in the surrounding test matrix.

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
- whether `Onnxify.ModelGenerator` reconstructs a TorchSharp expression/module with matching runtime behavior for the ONNX pattern

Focus on the emitted graph contract, for example:

- does `addmm` use `Gemm` with the same `alpha` and `beta` semantics
- does `addr` preserve the `beta == 0` behavior that avoids propagating `self`
- does `lerp` follow the same two-branch numerically stable formulation
- does `max.dim` or `topk` return both values and indices in the same shape conventions

Do not mark two exporters equivalent just because both eventually use the same ONNX primitive.
Also compare the C# behavior against the intent visible in the Python tests, not just the raw Python exporter body.
For `deep import`, do not mark a converter equivalent just because generated code compiles; compare the generated TorchSharp runtime behavior against the ONNX pattern produced by ONNXScript.

## 8. Review The Existing C# Unit Tests

After tracing the exporter or importer and comparing it against ONNXScript, inspect the current tests that cover the behavior in:

- `src/Onnxify.Tests`

Search by:

- exporter method name
- ModelGenerator converter class name
- Torch op spelling
- nearby helper name when the main implementation is thin

For example:

```powershell
rg -n "ExportAddMM|aten::addmm|addmm" src\Onnxify.Tests
rg -n "Conv2dTorchModuleOperator|Conv|DeepImportExportParity" src\Onnxify.Tests
```

Review whether the current tests actually cover the semantic points that matter for parity with ONNXScript, such as:

- the expected ONNX node kind
- shape and tuple-output conventions
- default parameter behavior
- overload-specific behavior
- scalar-versus-tensor cases
- dtype-sensitive branches
- `keepdim`, axis normalization, reshape, squeeze, or broadcast behavior
- special branches like `beta == 0`, `alpha == 1`, or scalar-rank handling
- explicit unsupported paths

Do not assume that existing tests are sufficient just because some exporter test already exists.
Part of validation is deciding whether the current tests really protect the C# behavior you just compared against ONNXScript.
Use the Python tests as a guide for which semantic branches deserve coverage in `Onnxify.Tests`, but do not transliterate Python test code directly into C#.
The C# tests should express the same behavioral contract in repo-idiomatic form, without dragging Python-specific language patterns or helpers into the C# suite.

If tests are missing an important semantic branch, record that gap explicitly and, when the task requires code changes, add or update focused tests in `src/Onnxify.Tests`.

For operators that can be represented by both `deep import` and `deep export`, prefer covering the validated behavior with `deep roundtrip tests` in `src/Onnxify.Tests/DeepImportExportParityTests.cs`.
Those tests should construct or load a small ONNX graph, deep import it into a generated TorchSharp module, deep export it back to ONNX, and compare runtime outputs.
This is the strongest evidence that the two feature paths are complementary: the import side can reconstruct the operator, and the export side can emit an ONNX graph that preserves the same behavior.
Deep roundtrip tests must also verify ONNX Runtime session creation for the original or round-tripped model, preferably through `OnnxRuntimeCompatibilityAssert`.
Do not treat a graph-shape assertion, serialization round trip, generated-code compilation, or TorchSharp eager comparison as enough by itself; operators can be valid ONNX and still fail in `Microsoft.ML.OnnxRuntime` because a kernel is unavailable or unsupported for the emitted op version/type combination.
When a roundtrip or deep-export test writes a temporary model and creates an `InferenceSession`, route that session creation through `OnnxRuntimeCompatibilityAssert.CreateSession(...)` so failures include the exported operator summary.
Keep focused exporter or importer unit tests for local ONNXScript parity branches that cannot be exercised clearly through a roundtrip.

## 9. Record The Validation Outcome Clearly

A useful validation result usually fits one of these buckets:

- equivalent: C# behavior appears semantically aligned with ONNXScript for the covered surface
- equivalent with explicit constraints: C# matches, but only for a narrower safe subset and rejects the rest clearly
- partial mismatch: the main path matches, but one or more branches, defaults, or shape rules differ
- unsupported despite nearby coverage: the generated docs may show a related exporter, but the exact op or overload is not actually covered
- import/export drift: one feature path matches the expected semantics, but the complementary path cannot roundtrip the same operator or graph pattern

If you find a mismatch, write down:

- the exact Torch op name
- the exact C# signature or ModelGenerator converter validated
- the Python function and file in `third_party/onnxscript`
- the concrete semantic difference
- the current state of C# test coverage for that behavior
- whether `deep roundtrip tests` cover the behavior when the operator belongs to both `deep export` and `deep import`
- whether the difference is acceptable repo policy, an intentional subset, or a bug

## 10. Suggested Search Pattern

For a single operator, a practical terminal workflow is:

```powershell
rg -n "aten::addmm" TORCH_OPERATOR_COVERAGE.md
rg -n "aten::addmm" .agents\skills\onnxify\references\torchsharp-converters\index.md
rg -n "ExportAddMM|\[TorchOp\(\" src\Onnxify.TorchSharp
rg -n "TorchSharpOp|AddMM|Gemm|Conv" src\Onnxify.ModelGenerator\Services\TorchModuleInlineOperators src\Onnxify.ModelGenerator\Services\TorchModuleOperators
rg -n '@torch_op\("aten::addmm"|@torch_op\(\("aten::addmm"' third_party\onnxscript\onnxscript\function_libs\torch_lib\ops
rg -n "ExportAddMM|aten::addmm|addmm" src\Onnxify.Tests
rg -n "DeepImportExportParity|addmm|Gemm|Conv" src\Onnxify.Tests
```

Then open:

- the generated converter page
- the C# exporter source
- the ModelGenerator TorchModule importer source when validating `deep import`
- every semantic helper it calls
- the ONNXScript Python implementation
- the ONNXScript Python tests for that operator
- the existing C# tests for that exporter, importer, and deep roundtrip path

## 11. Common Pitfalls

- Do not validate against a guessed Torch op spelling. Use the exact observer row.
- Do not treat `Onnxify.TorchSharp coverage = yes` in the observer report as proof of semantic parity. It only proves `[TorchOp(...)]` coverage exists.
- Do not treat `Onnxify.ModelGenerator coverage = yes` in the observer report as proof of semantic parity. It only proves `[TorchSharpOp(...)]` coverage exists.
- Do not stop at the generated converter page. It gives the signature, not the full behavior.
- Do not ignore helper methods that change dtype, rank, branching, or tuple outputs.
- Do not compare only the first ONNX node emitted. Many converters differ in surrounding casts, reshapes, or reduction behavior.
- Do not treat missing presence in `index.md` as a docs bug by default. For this workflow, absence means unsupported until proven otherwise.
- Do not treat the presence of a smoke test as proof that all parity-relevant branches are covered.
- Do not treat generated-code compilation as proof that `deep import` preserves runtime behavior.
- Do not copy Python tests line-for-line into C# when the language model, test helpers, or runtime expectations differ.

## Heuristics

- Prefer validating the exact overload named in the observer report before collapsing several overloads together.
- Prefer method-name search from the generated converter page over broad repo-wide Torch op search once you know the signature.
- Treat helper tracing as required whenever the exporter or importer is more than a one-node or one-expression wrapper.
- When ONNXScript supports a wider surface than C#, first decide whether the C# subset is intentionally narrower and explicit or silently incomplete.
- Treat test review as part of validation, not as an optional follow-up.
- If the validation uncovers a real mismatch or an uncovered semantic branch, add or update focused tests in `src/Onnxify.Tests` close to the exporter you inspected.
- Use the ONNXScript tests to understand the operator's contract, then restate that contract as idiomatic C# tests before or alongside exporter changes.
- Prefer `deep roundtrip tests` whenever the validated operator should work through both `deep import` and `deep export`; that keeps one feature path from drifting away from the other.
