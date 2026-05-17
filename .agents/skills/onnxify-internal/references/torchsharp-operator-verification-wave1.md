# Verification Wave 1: 25 Covered `Onnxify.TorchSharp` Tensor Operators

## Scope

This report records the first explicit parity-validation wave for 25 already covered tensor operators in `src/Onnxify.TorchSharp/TorchTensorOperatorExtensions.cs`.

Validation was performed against:

- `src/Onnxify.TorchSharp.Observer/torchsharp-operator-report.md`
- `.agents/skills/onnxify/references/torchsharp-converters/index.md`
- `third_party/onnxscript/onnxscript/function_libs/torch_lib/ops/core.py`
- `third_party/onnxscript/tests/function_libs/torch_lib/ops_test.py`
- `third_party/onnxscript/tests/function_libs/torch_lib/ops_test_data.py`
- `src/Onnxify.Tests/TorchTensorOperatorExtensionsTests.cs`

Focused validation command:

```powershell
dotnet test src\Onnxify.Tests\Onnxify.Tests.csproj --filter TorchTensorOperatorExtensionsTests
```

Result after this wave: `47/47` passing on `net8.0` and `net10.0`.

## Status Legend

- `equivalent`: exporter behavior matches the ONNXScript exporter contract covered in Python code/tests.
- `equivalent with constraints`: parity matches ONNXScript, but the shared contract is itself intentionally constrained or known-incomplete.
- `partial mismatch fixed`: a real C# exporter mismatch was found during validation and fixed in this wave.

## Summary Of Fixes Applied In This Wave

- `aten::topk`: scalar inputs are now rejected explicitly, matching ONNXScript's unsupported path.
- `aten::sort`: scalar inputs now export as `Identity(self)` plus scalar index `0`, matching ONNXScript.
- `aten::gather`: scalar-input and scalar-index branches were aligned with ONNXScript, and the scalar-index path was adjusted to survive post-cast rank loss in C# graph building.
- `src/Onnxify.Tests/TorchTensorOperatorExtensionsTests.cs` was expanded with runtime and branch-sensitive coverage for ordering, gather/index/select, where overloads, slice/broadcast/expand/unsqueeze, scalar extrema, `logit`, `logsumexp`, `remainder.Scalar_Tensor`, and `addr(beta != 0)`.

## Operator Matrix

| # | Torch op | Status | Notes |
| --- | --- | --- | --- |
| 1 | `aten::topk` | `partial mismatch fixed` | ONNXScript does not support scalar inputs; C# now rejects them explicitly and has runtime coverage for normal ordered outputs. |
| 2 | `aten::sort` | `partial mismatch fixed` | Scalar branch now matches ONNXScript's `Identity + 0` behavior; runtime tests cover values and indices. |
| 3 | `aten::max.dim` | `equivalent` | Covered for values, indices, `keepdim=false`, and scalar branch returning index `0`. |
| 4 | `aten::min.dim` | `equivalent` | Covered for values, indices, `keepdim=false`, and scalar branch returning index `0`. |
| 5 | `aten::logsumexp` | `equivalent` | Covered for reduction path and scalar identity path. |
| 6 | `aten::logit` | `equivalent` | Covered for `eps != null` clamped path and `eps == null` direct-logit path. |
| 7 | `aten::isclose` | `equivalent with constraints` | Matches ONNXScript, including the current shared omission around `equal_nan` noted by ONNXScript's `FIXME`. |
| 8 | `aten::allclose` | `equivalent with constraints` | Same `equal_nan` limitation as ONNXScript; otherwise parity is aligned. |
| 9 | `aten::floor_divide` | `equivalent` | Existing runtime coverage already validates signed integer behavior. |
| 10 | `aten::remainder.Scalar_Tensor` | `equivalent` | Runtime coverage now includes scalar-tensor remainder semantics. |
| 11 | `aten::where.self` | `equivalent` | Runtime coverage added through the tensor/tensor/tensor branch. |
| 12 | `aten::where.ScalarOther` | `equivalent` | Runtime coverage added. |
| 13 | `aten::where.ScalarSelf` | `equivalent` | Runtime coverage added. |
| 14 | `aten::gather` | `partial mismatch fixed` | Scalar input expansion and scalar index handling were aligned with ONNXScript; runtime coverage now exercises valid scalar branches. |
| 15 | `aten::index_select` | `equivalent` | Runtime coverage now includes scalar-input behavior after reshape/gather/squeeze. |
| 16 | `aten::slice.Tensor` | `equivalent` | Runtime coverage added for forward slicing with explicit `start/end/step`. |
| 17 | `aten::broadcast_to` | `equivalent` | Runtime coverage confirms alias behavior through `ExportExpand`. |
| 18 | `aten::expand_as` | `equivalent` | Runtime coverage added for `Shape(other)`-driven expansion. |
| 19 | `aten::unsqueeze` | `equivalent` | Runtime coverage added for insertion at a negative axis. |
| 20 | `aten::addmm` | `equivalent` | Existing runtime coverage validates `alpha` and `beta` fused GEMM semantics. |
| 21 | `aten::addmv` | `equivalent` | Existing runtime coverage validates scaled matrix-vector accumulation. |
| 22 | `aten::addbmm` | `equivalent` | Existing runtime coverage validates batch reduction plus scaled accumulation. |
| 23 | `aten::baddbmm` | `equivalent` | Existing runtime coverage validates batch matmul with `beta/alpha` scaling. |
| 24 | `aten::addr` | `equivalent` | Runtime coverage now includes both `beta == 0` and `beta != 0` branches. |
| 25 | `aten::mv` | `equivalent` | Existing runtime coverage validates matrix-vector output semantics. |

## Notes For Future Waves

- This wave validated parity against the ONNXScript exporter and its tests, not against raw PyTorch behavior independently of ONNXScript.
- `aten::isclose` and `aten::allclose` should stay marked as constrained until ONNXScript itself resolves the `equal_nan` gap.
- `aten::topk` remains intentionally constrained the same way ONNXScript is constrained: scalar inputs are unsupported.
