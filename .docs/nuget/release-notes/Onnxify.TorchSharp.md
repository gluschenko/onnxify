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
- Added lowering for common forward-body patterns used by LSTM, CNN, and MiniGPT-style models, including local helper calls, tuple deconstruction, validation guards, shape reads, tensor indexing, scalar arithmetic, `using var` temporaries, `torch.arange`, `torch.full`, `torch.triu`, `torch.matmul`, `torch.softmax`, and tensor reshape/transpose/slice/expand methods.
- Changed `nn.Linear` export from `Gemm` to `MatMul` plus optional `Add`, preserving leading batch dimensions for higher-rank inputs.
- Added deep-export smoke and unit coverage for LSTM-style, MiniGPT-style, AlexNet-like, and MobileNet-like TorchSharp modules.

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
