## 0.3.9

- Aligned the package version with the 0.3.9 Onnxify package family release.

## 0.3.8

- Added `Evaluate(...)` for comparing a TorchSharp module against an exported `OnnxModel` with ONNX Runtime on caller-provided representative samples.
- Added evaluation result types with aggregate loss, per-sample/per-output details, pass/fail status, and diagnostics for runtime or comparison failures.
- Added configurable tolerances, input/output name mapping, temporary model serialization, ONNX Runtime session options, and custom output comparison callbacks.
- Added `Microsoft.ML.OnnxRuntime.Managed` `1.27.0` as an `Onnxify.TorchSharp` dependency so export evaluation stays out of the core `Onnxify` package.

## 0.3.7

- Fixed `nn.Linear` export for rank-3 and higher inputs by explicitly flattening leading dimensions before `MatMul` and restoring them afterwards, avoiding ONNX Runtime extended-optimizer `Gemm` fusion failures on transformer activations such as `[batch, sequence, hidden]`.
- Fixed deep export of `torch.matmul(...)` calls whose operands include imported TorchSharp `Parameter` fields, materializing those parameters as ONNX initializers so deep-imported `MatMul` graphs can round-trip back through ONNX Runtime.
- Fixed signed integer `bitwise_right_shift` export so the arithmetic-shift sign mask uses unsigned `BitShift` inputs accepted by ONNX Runtime.
- Added ONNX Runtime regression coverage for TorchSharp bitwise and bit-shift exporters across integer, boolean, tensor/scalar, and signed arithmetic-shift cases.

## 0.3.6

- Fixed deep export scalar comparison typing for runtime tensor masks, preserving propagated tensor element types through factory outputs such as `torch.ones(..., dtype: ScalarType.Int32)` before lowering `eq(...)` / `ne(...)`.
- Added regression coverage for an `Int32` triangular runtime mask equality pattern, ensuring scalar comparison initializers are emitted with `int32` rather than widening to `int64`.
- Aligned the package version with the 0.3.6 Onnxify package family release.

## 0.3.5

- Aligned the package version with the 0.3.5 Onnxify package family release.

## 0.3.4

- Added `ModuleInputAttribute`, `ModuleOutputAttribute`, and `TensorDimension` so TorchSharp modules can declare tensor contracts with `ScalarType` and mixed fixed/symbolic dimensions, then export through attribute-driven `ExportOnnxModel(options)`.
- Added deep export support for static tensor factories such as `torch.ones(...)`, `torch.zeros(...)`, `torch.empty(...)`, `torch.rand(...)`, `torch.randn(...)`, `torch.randint(...)`, and `torch.normal(...)`, including named `dtype`/`device` metadata and compiler-generated inline-array shapes.
- Added deep export support for dynamic-shape tensor factories and reshapes that use `x.shape[i]` values, lowering them through ONNX `Shape`, `Gather`, `Concat`, and `ConstantOfShape` instead of treating unresolved dimensions as zero-sized initializers.
- Added deep export support for tensor `tril(...)`, `triu(...)`, `eq(...)`, `torch.where(...)`, `torch.full_like(...)`, `torch.ones_like(...)`, `torch.zeros_like(...)`, `torch.empty_like(...)`, `torch.nn.functional.softmax(...)`, and `torch.nn.functional.gelu(...)`.
- Added deep export lowering for nearest-neighbor 2D upsampling helper patterns, emitting ONNX `Resize` with spatial scale factors.
- Added deep export support for bare, `torch.*`, and tensor instance `matmul(...)`, `mm(...)`, and `bmm(...)` calls, all lowered through ONNX `MatMul`.
- Added deep export normalization for local C# array expressions, including sized initializers, implicit arrays, assigned arrays, empty arrays, and tensor arrays used as shape, permutation, scalar, or tensor-list arguments.
- Fixed deep export of compound assignments such as `scores += mask` so they preserve the previous graph value and emit the corresponding arithmetic node before rebinding the local variable.
- Fixed deep export helper-method body handling so local helper returns stop evaluation and preserve the actual returned graph value.
- Expanded deep export coverage for multi-input tuple-output modules, tensor `gather(...)`, attention-style tensor programs, and tuple-output safetensors modules.

## 0.3.3

- Added dictionary-based deep `ExportOnnxModel(...)` overloads for TorchSharp modules with multiple tensor inputs and multiple tensor outputs, including tuple-output forwards.
- Added deep export support on the base `torch.nn.Module` type and generic `Module<Tensor, ..., TOutput>` shapes so generated modules such as `Module<Tensor, (Tensor, Tensor)>` can be exported.
- Added deep export support for tensor instance `sigmoid()`, `softmax(...)`, `repeat(...)`, and `gather(...)`, plus generated TorchModule helper patterns for `Slice`, `Split`, `Resize`, `Squeeze`, reductions, and `TopK`.
- Fixed deep export of generated `ResizeTensor(...)` calls so ONNX `Resize` emits either `scales` or `sizes`, never both, producing models accepted by ONNX Runtime.
- Improved deep export materialization of runtime Torch tensor constants and `.data<T>().ToArray()` expressions used by generated TorchModule initializers.
- Generalized TorchModule safetensors extension methods to all `torch.nn.Module` instances, including modules whose `forward` returns tuples.
- Added deep roundtrip coverage and focused tests for multi-input tuple-output export, tensor `gather(...)`, and tuple-output safetensors modules.

## 0.3.2

- Added deep TorchSharp export support for tensor instance `flatten(...)` calls, including full flatten, `flatten(1, -1)`, and statically shaped partial flatten ranges.
- Kept `flatten(1, -1)` lowered to ONNX `Flatten(axis: 1)` for common classifier heads and used ONNX `Reshape` for PyTorch flatten forms whose result is not rank 2.
- Added deep TorchSharp export support for `torch.nn.functional.pad(...)`, including rank tracking for intermediate graph edges so generated TorchModule imports can round-trip asymmetric convolution padding back to ONNX.
- Improved deep import plus deep export parity for stride-2 convolutions whose ONNX padding is asymmetric, avoiding spatial computation drift after round-tripping.
- Added deep roundtrip tests that import ONNX graphs as TorchSharp modules, export them back to ONNX, and compare runtime outputs for classifier-head, convolution, and residual patterns.
- Aligned the package version with the 0.3.2 Onnxify package family release.

## 0.3.0

- Aligned the package version with the 0.3.0 Onnxify package family release.
- Updated the TorchSharp dependency baseline to `0.107.0`; matching TorchSharp runtime packages such as `TorchSharp-cpu` and `TorchSharp-cuda-windows` should use the same version.
- Updated generated TorchSharp converter documentation and coverage reports for the latest ModelGenerator TorchModule reconstruction coverage.

## 0.2.0

- Added `TorchTensorDataType` for public exporter `dtype` parameters, replacing numeric magic values such as `-1`, `1`, `7`, and `9` with readable enum values.
- Added TorchSharp tensor exporters for random and sampling operators: `bernoulli`, `rand`, `rand_like`, `randn`, `randn_like`, `randint`, `randint.low`, `randint_like`, `randint_like.low_dtype`, `normal.*`, and `multinomial`.
- Added TorchSharp tensor exporters for shape and rearrangement operators including `flip`, `roll`, `unbind.int`, `diagonal`, `repeat_interleave.self_int`, `repeat_interleave.Tensor`, and statically representable `as_strided`.
- Added TorchSharp tensor exporters for loss, padding, pooling, and miscellaneous operators including `mse_loss`, `nll_loss`, `cross_entropy_loss`, `pad`, `max_pool2d_with_indices`, `cross`, `is_nonzero`, `tensor.bool`, `tensor.float`, `tensor.int`, and `prims::var`.
- Added ONNXScript-style operator alias coverage for bitwise, shift, modulo, and true-division tensor operators, and lowered boolean bitwise operations to ONNX logical nodes.
- Improved deep TorchSharp module export for functional activation calls, `torch.sigmoid`, `torch.exp`, tuple expressions, tuple-returning helper deconstruction, and explicit array-shaped `view(...)` / `reshape(...)` patterns.
- Updated generated TorchSharp converter documentation and coverage reports for the newly covered operators.

## 0.1.2

- Added async TorchSharp safetensors checkpoint helpers with `SaveStateAsSafetensorsAsync(...)` and `LoadStateFromSafetensorsAsync(...)`.
- Fixed deep export of `torch.cat([left, right], dim: ...)` when C# collection expressions decompile into compiler-generated inline-array spans.
- Added an `ExportOnnxModel(...)` overload that lets callers provide the exported input and output name prefixes.
- Promoted `ExportOnnxModel(...)` by removing its experimental obsolete marker.
- Added TorchSharp tensor exporters for bitwise unary and binary operators, bit shifts, vector dot products, `xlogy`, real-valued `angle`, `atleast_1d` / `atleast_2d` / `atleast_3d`, `div(..., rounding_mode: ...)`, determinant aliases, and tensor `copy`.
- Fixed generated TorchSharp operator coverage docs to count `[TorchOp]` `OnnxGraph` extension exporters, including `aten::addmm` coverage for ONNX `Gemm`.
- Added TorchSharp tensor exporters for `_to_copy`, `empty`, `empty_like`, scalar and tensor `fill`, `heaviside`, and `new_empty` / `new_full` / `new_ones` / `new_zeros`.
- Added TorchSharp tensor exporters for sequence `atleast_1d` / `atleast_2d` / `atleast_3d`, `empty_strided`, window creators, `logcumsumexp`, `logdet`, and `linalg_vector_norm`.

## 0.1.1

- Added deep export for single-input TorchSharp modules via `TorchModule.ExportOnnxModel(...)`, which decompiles supported `forward(Tensor)` methods and lowers the resulting data flow into ONNX.
- Added recursive export support for user-defined child modules when a built-in module exporter is not available.
- Added lowering for common forward-body patterns, including local helper calls, tuple deconstruction, validation guards, shape reads, tensor indexing, scalar arithmetic, `using var` temporaries, `torch.arange`, `torch.full`, `torch.triu`, `torch.matmul`, `torch.softmax`, and tensor reshape/transpose/slice/expand methods.
- Changed `nn.Linear` export from `Gemm` to `MatMul` plus optional `Add`, preserving leading batch dimensions for higher-rank inputs.
- Added deep-export smoke and unit coverage for recurrent, convolutional, attention-like, and classifier-style TorchSharp module patterns.

## 0.0.0.15

- Improved ONNXScript parity for the next wave of already covered TorchSharp tensor operators.
- Fixed `aten::split.Tensor` to use split-size semantics and mapped the list-sized overload to `aten::split_with_sizes`.
- Added a dedicated exporter for `prims::squeeze` and fixed `aten::squeeze.dim` on scalar inputs.
- Fixed `aten::masked_fill.Tensor` to cast replacement values like the input tensor before `Where`.
- Fixed `aten::all.dims` and `aten::any.dims` so an empty dims list reduces across all dimensions.
- Fixed `aten::clamp` so omitting both bounds now returns `Identity(self)`, matching ONNXScript.
- Expanded TorchSharp exporter test coverage for split/chunk/select, squeeze variants, masked fill, truth reductions, scalar `where`, and creator-style tensor ops.

## 0.0.0.1

- Initial release
