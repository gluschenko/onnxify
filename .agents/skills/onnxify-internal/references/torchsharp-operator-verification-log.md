# TorchSharp Operator Verification Log

## Purpose

This page is the living parity-validation log for already covered `Onnxify.TorchSharp` tensor operators in `src/Onnxify.TorchSharp/TorchTensorOperatorExtensions.cs`.

Use it as the single place to record completed validation waves against the original Python exporter in `third_party/onnxscript`.

## How To Extend This Log

When verifying the next batch of already supported operators:

- keep appending the newly validated Torch op spellings to the combined operator table on this page instead of creating a separate wave-only report;
- add one more wave entry to the change log below with the date-independent summary of fixes, scope, and focused test result;
- preserve the exact Torch op spellings from `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`;
- keep the status vocabulary consistent with the legend on this page;
- update the notes column when a later wave narrows, fixes, or further constrains a previously logged operator.

This page is intentionally both the instruction anchor and the historical log.

## Validation Inputs

Validation in the currently recorded waves was performed against:

- `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`
- `.agents/skills/onnxify/references/torchsharp-converters/index.md`
- `third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py`
- `third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/prims.py`
- `third_party/onnxscript/tests/function_libs/torch_lib/ops_test.py`
- `third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py`
- `src/Onnxify.Tests/TorchTensorOperatorExtensionsTests.cs`

Focused validation command:

```powershell
dotnet test src\Onnxify.Tests\Onnxify.Tests.csproj --filter TorchTensorOperatorExtensionsTests
```

## Status Legend

- `equivalent`: exporter behavior matches the ONNXScript exporter contract covered in Python code/tests.
- `equivalent with constraints`: parity matches ONNXScript, but the shared contract is intentionally constrained, known-incomplete, or the current C# surface is deliberately narrower.
- `partial mismatch fixed`: a real C# exporter mismatch was found during validation and fixed in the logged wave.

## Change Log

### Wave 1

Scope: first explicit parity-validation wave for 25 already covered tensor operators.

Focused result after this wave: `47/47` passing on `net8.0` and `net10.0`.

Fix summary:

- `aten::topk`: scalar inputs are now rejected explicitly, matching ONNXScript's unsupported path.
- `aten::sort`: scalar inputs now export as `Identity(self)` plus scalar index `0`, matching ONNXScript.
- `aten::gather`: scalar-input and scalar-index branches were aligned with ONNXScript, and the scalar-index path was adjusted to survive post-cast rank loss in C# graph building.
- `src/Onnxify.Tests/TorchTensorOperatorExtensionsTests.cs` was expanded with runtime and branch-sensitive coverage for ordering, gather/index/select, where overloads, slice/broadcast/expand/unsqueeze, scalar extrema, `logit`, `logsumexp`, `remainder.Scalar_Tensor`, and `addr(beta != 0)`.

### Wave 2

Scope: second explicit parity-validation wave for 25 more already covered tensor operators.

Focused result after this wave: `53/53` passing on `net8.0` and `net10.0`.

Fix summary:

- `aten::view` was separated from `aten::reshape` / `prims::reshape` so the `view` path now emits `Reshape(..., allowzero=1)`, matching ONNXScript.
- `aten::permute` now normalizes negative axes and treats an empty permutation the same way ONNXScript does.
- `aten::transpose.int` now returns `Identity(self)` for scalar inputs instead of failing.
- `prims::transpose` now has its own full-permutation exporter instead of incorrectly reusing the two-axis swap path from `aten::transpose.int`.
- `aten::t` now returns `Identity(self)` for non-matrix inputs, matching ONNXScript's trace-only contract.
- `src/Onnxify.Tests/TorchTensorOperatorExtensionsTests.cs` was expanded with branch-sensitive coverage for reshape-vs-view, permute normalization, scalar transpose, prims transpose, rank-1 `t`, and runtime checks for view/reshape, transpose variants, concat, and split.

## Combined Operator Log

Continue appending newly verified operators to this table during future waves. Do not start a fresh standalone table elsewhere unless there is a strong reason to change the workflow.

| # | Wave | Torch op | Status | Notes |
| --- | --- | --- | --- | --- |
| 1 | `wave1` | `aten::topk` | `partial mismatch fixed` | ONNXScript does not support scalar inputs; C# now rejects them explicitly and has runtime coverage for normal ordered outputs. |
| 2 | `wave1` | `aten::sort` | `partial mismatch fixed` | Scalar branch now matches ONNXScript's `Identity + 0` behavior; runtime tests cover values and indices. |
| 3 | `wave1` | `aten::max.dim` | `equivalent` | Covered for values, indices, `keepdim=false`, and scalar branch returning index `0`. |
| 4 | `wave1` | `aten::min.dim` | `equivalent` | Covered for values, indices, `keepdim=false`, and scalar branch returning index `0`. |
| 5 | `wave1` | `aten::logsumexp` | `equivalent` | Covered for reduction path and scalar identity path. |
| 6 | `wave1` | `aten::logit` | `equivalent` | Covered for `eps != null` clamped path and `eps == null` direct-logit path. |
| 7 | `wave1` | `aten::isclose` | `equivalent with constraints` | Matches ONNXScript, including the current shared omission around `equal_nan` noted by ONNXScript's `FIXME`. |
| 8 | `wave1` | `aten::allclose` | `equivalent with constraints` | Same `equal_nan` limitation as ONNXScript; otherwise parity is aligned. |
| 9 | `wave1` | `aten::floor_divide` | `equivalent` | Existing runtime coverage already validates signed integer behavior. |
| 10 | `wave1` | `aten::remainder.Scalar_Tensor` | `equivalent` | Runtime coverage now includes scalar-tensor remainder semantics. |
| 11 | `wave1` | `aten::where.self` | `equivalent` | Runtime coverage added through the tensor/tensor/tensor branch. |
| 12 | `wave1` | `aten::where.ScalarOther` | `equivalent` | Runtime coverage added. |
| 13 | `wave1` | `aten::where.ScalarSelf` | `equivalent` | Runtime coverage added. |
| 14 | `wave1` | `aten::gather` | `partial mismatch fixed` | Scalar input expansion and scalar index handling were aligned with ONNXScript; runtime coverage now exercises valid scalar branches. |
| 15 | `wave1` | `aten::index_select` | `equivalent` | Runtime coverage now includes scalar-input behavior after reshape/gather/squeeze. |
| 16 | `wave1` | `aten::slice.Tensor` | `equivalent` | Runtime coverage added for forward slicing with explicit `start/end/step`. |
| 17 | `wave1` | `aten::broadcast_to` | `equivalent` | Runtime coverage confirms alias behavior through `ExportExpand`. |
| 18 | `wave1` | `aten::expand_as` | `equivalent` | Runtime coverage added for `Shape(other)`-driven expansion. |
| 19 | `wave1` | `aten::unsqueeze` | `equivalent` | Runtime coverage added for insertion at a negative axis. |
| 20 | `wave1` | `aten::addmm` | `equivalent` | Existing runtime coverage validates `alpha` and `beta` fused GEMM semantics. |
| 21 | `wave1` | `aten::addmv` | `equivalent` | Existing runtime coverage validates scaled matrix-vector accumulation. |
| 22 | `wave1` | `aten::addbmm` | `equivalent` | Existing runtime coverage validates batch reduction plus scaled accumulation. |
| 23 | `wave1` | `aten::baddbmm` | `equivalent` | Existing runtime coverage validates batch matmul with `beta/alpha` scaling. |
| 24 | `wave1` | `aten::addr` | `equivalent` | Runtime coverage now includes both `beta == 0` and `beta != 0` branches. |
| 25 | `wave1` | `aten::mv` | `equivalent` | Existing runtime coverage validates matrix-vector output semantics. |
| 26 | `wave2` | `aten::reshape` | `equivalent` | Matches ONNXScript's plain `Reshape` path without `allowzero`. |
| 27 | `wave2` | `aten::view` | `partial mismatch fixed` | C# previously shared reshape semantics; `view` now emits `allowzero=1` like ONNXScript. |
| 28 | `wave2` | `prims::reshape` | `equivalent` | Matches the prims-side `Reshape` contract. |
| 29 | `wave2` | `aten::permute` | `partial mismatch fixed` | Negative-axis normalization and empty-permutation handling were aligned with ONNXScript. |
| 30 | `wave2` | `aten::transpose.int` | `partial mismatch fixed` | Scalar inputs now return identity instead of throwing; swap semantics remain covered. |
| 31 | `wave2` | `prims::transpose` | `partial mismatch fixed` | C# now exports the exact full permutation instead of treating the op like a two-axis swap. |
| 32 | `wave2` | `aten::t` | `partial mismatch fixed` | Non-rank-2 inputs now return identity, matching ONNXScript's contract. |
| 33 | `wave2` | `aten::expand` | `equivalent with constraints` | Normal expand semantics match; `-1` placeholders still require statically known input dimensions in C#. |
| 34 | `wave2` | `aten::view_as` | `equivalent` | Uses `Shape(other)` plus `Reshape(..., allowzero=1)` like ONNXScript. |
| 35 | `wave2` | `aten::alias` | `equivalent` | Identity semantics match ONNXScript. |
| 36 | `wave2` | `aten::_conj` | `equivalent with constraints` | Matches the non-complex identity contract; this wave did not extend C# to ONNXScript's separate complex branch. |
| 37 | `wave2` | `aten::clone` | `equivalent` | Identity semantics match ONNXScript's covered contract. |
| 38 | `wave2` | `aten::conj` | `equivalent with constraints` | Matches the non-complex identity contract; complex-specific parity remains outside this wave. |
| 39 | `wave2` | `aten::contiguous` | `equivalent` | Identity semantics match ONNXScript. |
| 40 | `wave2` | `aten::detach` | `equivalent` | Identity semantics match ONNXScript. |
| 41 | `wave2` | `aten::resolve_conj` | `equivalent` | Identity semantics match ONNXScript. |
| 42 | `wave2` | `aten::resolve_neg` | `equivalent` | Identity semantics match ONNXScript. |
| 43 | `wave2` | `aten::narrow` | `equivalent` | Existing slice-based lowering matches ONNXScript's narrow contract. |
| 44 | `wave2` | `aten::arange` | `equivalent with constraints` | Existing runtime coverage validates the integer `long` surface that `Onnxify.TorchSharp` currently exposes. |
| 45 | `wave2` | `aten::arange.start` | `equivalent with constraints` | Existing runtime coverage validates the current integer two-bound surface. |
| 46 | `wave2` | `aten::arange.start_step` | `equivalent with constraints` | Existing runtime coverage validates non-zero integer step behavior on the current C# surface. |
| 47 | `wave2` | `aten::cat` | `equivalent with constraints` | Concat semantics match ONNXScript for normal tensor inputs; ONNXScript's own test matrix still skips known ORT zero-dim edge cases. |
| 48 | `wave2` | `aten::concat` | `equivalent with constraints` | Alias of `aten::cat`; validated on the same shared exporter contract. |
| 49 | `wave2` | `aten::concatenate` | `equivalent with constraints` | Alias of `aten::cat`; validated on the same shared exporter contract. |
| 50 | `wave2` | `aten::split` | `equivalent` | Runtime coverage now validates split-size semantics, including a trailing smaller chunk. |

## Notes For Future Waves

- This log validates parity against the ONNXScript exporter and its tests, not against raw PyTorch behavior independently of ONNXScript.
- `aten::isclose` and `aten::allclose` should stay marked as constrained until ONNXScript itself resolves the `equal_nan` gap.
- `aten::_conj` and `aten::conj` should stay marked as constrained until `Onnxify.TorchSharp` grows an explicit complex-tensor story comparable to ONNXScript's dedicated complex branches.
- `aten::expand` and the `aten::arange*` family are aligned on the currently exposed C# surface, but they remain narrower than ONNXScript's broader dtype- and dynamic-shape-aware Python contract.
- `aten::topk` remains intentionally constrained the same way ONNXScript is constrained: scalar inputs are unsupported.
