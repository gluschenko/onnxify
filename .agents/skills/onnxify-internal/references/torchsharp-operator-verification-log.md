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

### Wave 3

Scope: third explicit parity-validation wave for the next 25 already covered tensor operators after the previous logged batch.

Focused result after this wave: `57/57` passing on `net8.0` and `net10.0`.

Fix summary:

- `aten::split.Tensor` was remapped onto the scalar split-size exporter, and the list-based overload was corrected to `aten::split_with_sizes`, matching the ONNXScript operator split between `split` and `split_with_sizes`.
- `prims::squeeze` now has its own explicit-axes exporter instead of incorrectly sharing `aten::squeeze`'s no-axes behavior.
- `aten::squeeze.dim` now returns `Identity(self)` for scalar inputs, matching ONNXScript.
- `aten::masked_fill.Tensor` now casts the replacement value like `self` before `Where`, matching ONNXScript's `CastLike` behavior.
- `aten::all.dims` and `aten::any.dims` now treat an empty `dims` list as reduce-all-dimensions instead of passing an empty axes tensor through to ONNX.
- `aten::clamp` now returns `Identity(self)` when both bounds are omitted, matching ONNXScript's null-bounds branch.
- `src/Onnxify.Tests/TorchTensorOperatorExtensionsTests.cs` was expanded with branch-sensitive and runtime coverage for split-with-sizes, scalar `squeeze.dim`, `prims::squeeze`, tensor `masked_fill`, empty-dims truth reductions, scalar `where`, `select`, `chunk`, and the no-bounds `clamp` path.

### Wave 4

Scope: fourth explicit parity-validation wave for the 50 newly covered tensor operators introduced by the latest Torch ops batch.

Focused result after this wave: `70/70` passing on `net8.0` and `net10.0`.

Fix summary:

- `aten::bitwise_not`, `aten::bitwise_and.*`, `aten::bitwise_or.*`, and `aten::bitwise_xor.*`: bool tensor inputs now emit ONNX logical `Not` / `And` / `Or` / `Xor`, matching ONNXScript's bool branch instead of incorrectly using integer-only `Bitwise*` nodes.
- The integer bitwise-shift exporters were rechecked against ONNXScript's unsigned-cast plus arithmetic-right-shift mask lowering.
- The `atleast_*`, `empty*`, `new_*`, window, copy/cast, `xlogy`, `logcumsumexp`, determinant, `heaviside`, and vector-norm exporters were rechecked against the ONNXScript implementations and tests; narrower C# surfaces are recorded as explicit constraints where dtype, device/layout, sequence, dynamic-rank, or multi-axis behavior is intentionally not exposed.
- `src/Onnxify.Tests/TorchTensorOperatorExtensionsTests.cs` was expanded with bool bitwise branch coverage.

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
| 51 | `wave3` | `aten::split.Tensor` | `partial mismatch fixed` | C# had mapped this spelling onto split-with-sizes semantics; it now shares the scalar split-size path like ONNXScript, while the list-based overload is exposed as `aten::split_with_sizes`. |
| 52 | `wave3` | `aten::chunk` | `equivalent with constraints` | Normal chunk semantics match and now have runtime coverage, but C# still relies on a statically known axis size to determine the emitted output count. |
| 53 | `wave3` | `aten::select.int` | `equivalent` | Gather-based select semantics match ONNXScript and now have runtime coverage. |
| 54 | `wave3` | `aten::squeeze` | `equivalent` | No-axes squeeze semantics match ONNXScript. |
| 55 | `wave3` | `prims::squeeze` | `partial mismatch fixed` | C# now exports explicit axes for prims squeeze instead of reusing `aten::squeeze`'s remove-all-size-1-dims behavior. |
| 56 | `wave3` | `aten::squeeze.dim` | `partial mismatch fixed` | Scalar inputs now return identity instead of failing on rank-0 axis normalization. |
| 57 | `wave3` | `prims::where` | `equivalent` | Tensor/tensor/tensor `Where` semantics match ONNXScript. |
| 58 | `wave3` | `aten::where.Scalar` | `equivalent with constraints` | The scalar/scalar branch is aligned for the current float-literal C# surface and now has runtime coverage, but it remains narrower than ONNXScript's broader promotion-aware Python contract. |
| 59 | `wave3` | `aten::masked_fill.Scalar` | `equivalent` | Scalar masked-fill semantics remain aligned through the shared `Where` lowering. |
| 60 | `wave3` | `aten::masked_fill.Tensor` | `partial mismatch fixed` | Replacement tensors are now cast like `self` before `Where`, matching ONNXScript. |
| 61 | `wave3` | `aten::all` | `equivalent` | Reduce-all truth semantics match ONNXScript, including the scalar identity branch. |
| 62 | `wave3` | `aten::all.dim` | `equivalent` | Single-dimension truth reduction semantics match ONNXScript. |
| 63 | `wave3` | `aten::all.dims` | `partial mismatch fixed` | Empty `dims` now means reduce all dimensions, matching ONNXScript's dedicated no-dim branch. |
| 64 | `wave3` | `aten::any` | `equivalent` | Reduce-any truth semantics match ONNXScript, including the scalar identity branch. |
| 65 | `wave3` | `aten::any.dim` | `equivalent` | Single-dimension truth reduction semantics match ONNXScript. |
| 66 | `wave3` | `aten::any.dims` | `partial mismatch fixed` | Empty `dims` now means reduce all dimensions, matching ONNXScript's dedicated no-dim branch. |
| 67 | `wave3` | `aten::nonzero` | `equivalent` | `NonZero` plus transpose semantics match ONNXScript and retain runtime coverage. |
| 68 | `wave3` | `aten::cumsum` | `equivalent with constraints` | Main cumsum semantics match and scalar inputs return identity, but the currently exposed C# surface does not add ONNXScript's optional dtype override and inherits the same practical dtype caveats around non-default accumulation types. |
| 69 | `wave3` | `aten::clamp` | `partial mismatch fixed` | The no-bounds branch now returns identity instead of throwing, matching ONNXScript. |
| 70 | `wave3` | `aten::full` | `equivalent with constraints` | Expand-based creation semantics match for the current float-producing C# surface, but the broader ONNXScript dtype/device parameters remain outside this exporter signature. |
| 71 | `wave3` | `aten::full_like` | `equivalent with constraints` | Shape-driven fill semantics match, with the current C# surface intentionally narrower than ONNXScript's explicit dtype override options. |
| 72 | `wave3` | `aten::ones` | `equivalent with constraints` | Current C# surface matches the float creator path but is narrower than ONNXScript's broader dtype-aware signature. |
| 73 | `wave3` | `aten::ones_like` | `equivalent with constraints` | Shape-driven ones creation matches for the current C# surface, with narrower dtype configurability than ONNXScript. |
| 74 | `wave3` | `aten::zeros` | `equivalent with constraints` | Current C# surface matches the float creator path but is narrower than ONNXScript's broader dtype-aware signature. |
| 75 | `wave3` | `aten::zeros_like` | `equivalent with constraints` | Shape-driven zeros creation matches for the current C# surface, with narrower dtype configurability than ONNXScript. |
| 76 | `wave4` | `aten::div.Tensor_mode` | `equivalent with constraints` | Rounding modes `null`, `trunc`, and `floor` match ONNXScript; integer-result parity relies on known input element type so C# can cast rounded results back like ONNXScript. |
| 77 | `wave4` | `aten::div.Scalar_mode` | `equivalent with constraints` | Same `rounding_mode` contract as `aten::div.Tensor_mode`, with the scalar lowered through a typed scalar initializer. |
| 78 | `wave4` | `aten::angle` | `equivalent with constraints` | Real-valued branch matches ONNXScript; the separate complex-tensor branch remains outside the current C# surface. |
| 79 | `wave4` | `aten::bitwise_not` | `partial mismatch fixed` | Bool inputs now emit logical `Not`, while integer inputs continue to emit `BitwiseNot`, matching ONNXScript's dtype split. |
| 80 | `wave4` | `aten::bitwise_and.Tensor` | `partial mismatch fixed` | Bool inputs now emit logical `And`; integer inputs continue to emit `BitwiseAnd`. |
| 81 | `wave4` | `aten::bitwise_and.Scalar` | `partial mismatch fixed` | Bool scalar branch now lowers through logical `And`; integer scalar branch remains `BitwiseAnd`. |
| 82 | `wave4` | `aten::bitwise_and.Scalar_Tensor` | `partial mismatch fixed` | Bool scalar-tensor branch now lowers through logical `And`; integer branch remains `BitwiseAnd`. |
| 83 | `wave4` | `aten::bitwise_or.Tensor` | `partial mismatch fixed` | Bool inputs now emit logical `Or`; integer inputs continue to emit `BitwiseOr`. |
| 84 | `wave4` | `aten::bitwise_or.Scalar` | `partial mismatch fixed` | Bool scalar branch now lowers through logical `Or`; integer scalar branch remains `BitwiseOr`. |
| 85 | `wave4` | `aten::bitwise_or.Scalar_Tensor` | `partial mismatch fixed` | Bool scalar-tensor branch now lowers through logical `Or`; integer branch remains `BitwiseOr`. |
| 86 | `wave4` | `aten::bitwise_xor.Tensor` | `partial mismatch fixed` | Bool inputs now emit logical `Xor`; integer inputs continue to emit `BitwiseXor`. |
| 87 | `wave4` | `aten::bitwise_xor.Scalar` | `partial mismatch fixed` | Bool scalar branch now lowers through logical `Xor`; integer scalar branch remains `BitwiseXor`. |
| 88 | `wave4` | `aten::bitwise_xor.Scalar_Tensor` | `partial mismatch fixed` | Bool scalar-tensor branch now lowers through logical `Xor`; integer branch remains `BitwiseXor`. |
| 89 | `wave4` | `aten::bitwise_left_shift.Tensor` | `equivalent with constraints` | Integer shift matches ONNXScript's unsigned-cast BitShift lowering; bool and floating inputs are explicitly rejected. |
| 90 | `wave4` | `aten::bitwise_left_shift.Tensor_Scalar` | `equivalent with constraints` | Scalar shift count is typed like the input and shares the integer-only BitShift lowering. |
| 91 | `wave4` | `aten::bitwise_left_shift.Scalar_Tensor` | `equivalent with constraints` | Scalar-left branch is typed like the tensor operand and shares the integer-only BitShift lowering. |
| 92 | `wave4` | `aten::bitwise_right_shift.Tensor` | `equivalent with constraints` | Integer arithmetic right shift matches ONNXScript's unsigned mask reconstruction for signed inputs; bool and floating inputs are rejected. |
| 93 | `wave4` | `aten::bitwise_right_shift.Tensor_Scalar` | `equivalent with constraints` | Scalar shift count shares the signed arithmetic-right-shift mask lowering. |
| 94 | `wave4` | `aten::bitwise_right_shift.Scalar_Tensor` | `equivalent with constraints` | Scalar-left branch shares the signed arithmetic-right-shift mask lowering. |
| 95 | `wave4` | `aten::dot` | `equivalent` | 1-D dot surface lowers to `MatMul`, matching ONNXScript. |
| 96 | `wave4` | `aten::xlogy.Tensor` | `equivalent` | `IsNaN(other)`, `self == 0`, and `self * Log(other)` branches match ONNXScript. |
| 97 | `wave4` | `aten::xlogy.Scalar_Self` | `equivalent with constraints` | Scalar-self branch matches ONNXScript after typing the scalar like `other`; the C# scalar surface is narrower than Python promotion behavior. |
| 98 | `wave4` | `aten::xlogy.Scalar_Other` | `equivalent with constraints` | Scalar-other branch matches ONNXScript after typing the scalar like `self`; the C# scalar surface is narrower than Python promotion behavior. |
| 99 | `wave4` | `aten::_linalg_det` | `equivalent` | Direct `Det` lowering matches the shared ONNXScript determinant exporter. |
| 100 | `wave4` | `aten::linalg_det` | `equivalent` | Alias of the determinant exporter. |
| 101 | `wave4` | `aten::det` | `equivalent` | Alias of the determinant exporter. |
| 102 | `wave4` | `aten::heaviside` | `equivalent` | `self < 0`, `self == 0`, and positive branches match ONNXScript's nested `Where` behavior. |
| 103 | `wave4` | `aten::atleast_1d` | `equivalent with constraints` | Static-rank scalar reshape and non-scalar identity branches match ONNXScript; C# requires known rank. |
| 104 | `wave4` | `aten::atleast_1d.Sequence` | `equivalent with constraints` | Per-input behavior matches ONNXScript's sequence map, represented as a C# list surface with known ranks. |
| 105 | `wave4` | `aten::atleast_2d` | `equivalent with constraints` | Rank 0/1 reshape to `[1, -1]` and higher-rank identity match ONNXScript; C# requires known rank. |
| 106 | `wave4` | `aten::atleast_2d.Sequence` | `equivalent with constraints` | Per-input behavior matches ONNXScript's sequence map, represented as a C# list surface with known ranks. |
| 107 | `wave4` | `aten::atleast_3d` | `equivalent with constraints` | Rank 0/1 reshape, rank-2 unsqueeze, and higher-rank identity match ONNXScript; C# requires known rank. |
| 108 | `wave4` | `aten::atleast_3d.Sequence` | `equivalent with constraints` | Per-input behavior matches ONNXScript's sequence map, represented as a C# list surface with known ranks. |
| 109 | `wave4` | `aten::copy` | `equivalent` | `CastLike(src, self)` behavior matches ONNXScript. |
| 110 | `wave4` | `aten::_to_copy` | `equivalent with constraints` | `dtype == -1` identity and explicit dtype cast match ONNXScript; layout/device/memory-format parameters remain ignored like the Python exporter. |
| 111 | `wave4` | `aten::empty.memory_format` | `equivalent with constraints` | Zero-filled simulation of nondeterministic empty tensors matches ONNXScript's export strategy; layout/device/memory-format are not represented in ONNX. |
| 112 | `wave4` | `aten::empty_strided` | `equivalent with constraints` | Shares the same zero-filled empty simulation and intentionally ignores stride/layout/device metadata. |
| 113 | `wave4` | `aten::blackman_window` | `equivalent with constraints` | Emits ONNX `BlackmanWindow` with dtype defaulting to float, matching ONNXScript for the covered dtype surface. |
| 114 | `wave4` | `aten::hamming_window` | `equivalent with constraints` | Emits ONNX `HammingWindow` with dtype defaulting to float, matching ONNXScript for the covered dtype surface. |
| 115 | `wave4` | `aten::hann_window` | `equivalent with constraints` | Emits ONNX `HannWindow` with dtype defaulting to float, matching ONNXScript for the covered dtype surface. |
| 116 | `wave4` | `aten::empty_like` | `equivalent with constraints` | Shape-driven zero-filled simulation plus optional dtype cast matches ONNXScript's empty-like export strategy. |
| 117 | `wave4` | `aten::fill.Scalar` | `equivalent` | Scalar value is cast like `self` and expanded to `Shape(self)`, matching ONNXScript. |
| 118 | `wave4` | `aten::fill.Tensor` | `equivalent` | Tensor value is cast like `self` and expanded to `Shape(self)`, matching ONNXScript. |
| 119 | `wave4` | `aten::new_empty` | `equivalent with constraints` | Zero-filled empty simulation plus `CastLike`/dtype cast matches ONNXScript's export strategy. |
| 120 | `wave4` | `aten::new_full` | `equivalent with constraints` | Fill value is cast to requested dtype or like `self` and expanded to requested size, matching ONNXScript for the current scalar C# surface. |
| 121 | `wave4` | `aten::new_ones` | `equivalent with constraints` | Ones creation delegates through the same `new_full` contract and matches ONNXScript for the current scalar C# surface. |
| 122 | `wave4` | `aten::new_zeros` | `equivalent with constraints` | Zero creation delegates through the same zero-filled creator contract and matches ONNXScript's `ConstantOfShape` intent. |
| 123 | `wave4` | `aten::logcumsumexp` | `equivalent` | Scalar identity and max-trick `Log(CumSum(Exp(self - max))) + max` behavior match ONNXScript. |
| 124 | `wave4` | `aten::logdet` | `equivalent` | `Log(Det(self))` lowering matches ONNXScript. |
| 125 | `wave4` | `aten::linalg_vector_norm` | `equivalent with constraints` | `ord` branches, optional dtype cast, flatten-when-dim-null, and keepdim behavior match ONNXScript for the current single-axis C# surface. |

## Notes For Future Waves

- This log validates parity against the ONNXScript exporter and its tests, not against raw PyTorch behavior independently of ONNXScript.
- `aten::isclose` and `aten::allclose` should stay marked as constrained until ONNXScript itself resolves the `equal_nan` gap.
- `aten::_conj` and `aten::conj` should stay marked as constrained until `Onnxify.TorchSharp` grows an explicit complex-tensor story comparable to ONNXScript's dedicated complex branches.
- `aten::expand` and the `aten::arange*` family are aligned on the currently exposed C# surface, but they remain narrower than ONNXScript's broader dtype- and dynamic-shape-aware Python contract.
- `aten::chunk`, `aten::full*`, `aten::ones*`, `aten::zeros*`, and `aten::where.Scalar` should stay marked as constrained until `Onnxify.TorchSharp` grows the same dynamic-shape and dtype-override surface area that ONNXScript already exposes in Python.
- `aten::topk` remains intentionally constrained the same way ONNXScript is constrained: scalar inputs are unsupported.
- `aten::angle` should stay marked as constrained until `Onnxify.TorchSharp` has a complex-tensor representation comparable to ONNXScript's complex branch.
- The `aten::atleast_*` sequence entries are validated against the C# list-returning export surface, not a first-class ONNX `SequenceMap` API surface.
- The `aten::empty*` and `aten::new_*` entries intentionally validate the deterministic zero-filled ONNX export strategy used by ONNXScript, not PyTorch's runtime nondeterministic memory contents.
