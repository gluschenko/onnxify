## 0.3.0

- Expanded TorchModule import support for ONNX graphs with broader runtime tensor data types that map to TorchSharp `ScalarType`, including non-float inputs and outputs.
- Added TorchModule import support for `GRU` and multi-output operators such as `Split` and `TopK`.
- Added TorchModule inline support for additional ONNX operators including `ArgMax`, `ArgMin`, `Celu`, `CumSum`, `DepthToSpace`, `Dropout`, `Expand`, `GatherElements`, `Gelu`, `GroupNormalization`, `HardSwish`, `InstanceNormalization`, `LayerNormalization`, `LogSoftmax`, `Mish`, `PRelu`, `Pad`, `ReduceMax`, `ReduceMin`, `ReduceProd`, `Resize`, `Selu`, `Slice`, `Softplus`, `SpaceToDepth`, `Tile`, and `Trilu`.
- Removed MVP wording from TorchModule backend diagnostics.

## 0.2.0

- Aligned the package version with the 0.2.0 Onnxify package family release.
- Added `OnnxifyModelImportType`, with `OnnxRuntimeInference` as the default and an opt-in `TorchModule` mode that emits a graph-shaped TorchSharp module for supported single-input/single-output ONNX graphs.
- Added TorchModule import support for ONNX `Acos`, `Acosh`, `Asin`, `Asinh`, `Atan`, `Atanh`, `Round`, `Sign`, `GreaterOrEqual`, and `LessOrEqual` operators.
- Added TorchModule import support for runtime input and output tensor data types that map to TorchSharp `ScalarType`, plus ONNX `LSTM`, `Not`, `Max`, and `Min`, including LSTM gate-order conversion when loading ONNX weights into TorchSharp.

## 0.1.2

- Aligned the package version with the 0.1.2 Onnxify package family release.

## 0.1.1

- Aligned the package version with the 0.1.1 Onnxify package family release.

## 0.1.0

- Fixed `OnnxModel` metadata overrides flowing from MSBuild into the source generator, so `OnnxifyModelNamespace` and `OnnxifyModelClassName` now work in real consumer projects.
- Expanded the package documentation with working runtime snippets that show `SessionOptions`, `RunOptions`, and correct disposal of generated model and output wrappers.

## 0.0.0.14

- Added `Microsoft.ML.OnnxRuntime.Float16` support for generated wrappers over ONNX `float16` tensor inputs and outputs.
- Added `Microsoft.ML.OnnxRuntime.BFloat16` support for generated wrappers over ONNX `bfloat16` tensor inputs and outputs.

## 0.0.0.8

- Initial release
