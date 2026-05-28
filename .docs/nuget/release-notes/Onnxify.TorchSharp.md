## 0.1.2

- Added async TorchSharp safetensors checkpoint helpers with `SaveStateAsSafetensorsAsync(...)` and `LoadStateFromSafetensorsAsync(...)`.
- Added an `ExportOnnxModel(...)` overload that lets callers provide the exported input and output name prefixes.
- Promoted `ExportOnnxModel(...)` by removing its experimental obsolete marker.

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
