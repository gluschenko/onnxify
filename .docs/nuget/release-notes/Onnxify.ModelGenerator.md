## 0.3.8

- Hoisted repeated generated ONNX Runtime and TorchModule runtime plumbing into shared generated base classes emitted once per consuming compilation and only for the import modes in use.
- Added editable generator template sources for the shared generated support classes so common generated helpers can be maintained as normal C#.

## 0.3.6

- Added `System.Collections.Immutable` to the analyzer runtime dependency payload so generated model wrappers can initialize the source generator reliably in consumer builds.
- Aligned the package version with the 0.3.6 Onnxify package family release.

## 0.3.5

- Aligned Roslyn references to `Microsoft.CodeAnalysis.CSharp` `4.11.0` for source-generator compatibility with the solution analyzer/runtime setup.
- Aligned the package version with the 0.3.5 Onnxify package family release.

## 0.3.3

- Changed the default `OnnxRuntimeInference` generator path to read ONNX metadata through generated protobuf classes instead of loading models through the public Onnxify object model, improving wrapper generation stability when core graph APIs have unrelated import bugs.
- Preserved unknown `None` tensor dimensions in generated runtime metadata by emitting `OnnxDimensionNone`.
- Preserved empty optional ONNX input slots in TorchModule generation so positional optional inputs such as `Resize` `scales` and `sizes` remain distinguishable.

## 0.3.2

- Aligned the package version with the 0.3.2 Onnxify package family release.
- Picked up the core graph-loading fix for ONNX models whose `value_info` repeats graph input or output names when generated TorchModule loader code reads weights through `OnnxModel.FromFile(...)`.
- Kept the source generator strictly on `netstandard2.0` while referencing the core `Onnxify` package for shared graph loading and `OnnxGraph.SortTopologically()`.
- Removed the duplicated raw-protobuf topological sort from TorchModule analysis; TorchModule generation now walks the shared `OnnxGraph` wrapper in `PreserveUntyped` mode to retain raw ONNX node inputs and attributes.
- Fixed TorchModule deep import for ONNX `Conv` nodes with asymmetric `pads` by emitting an explicit `torch.nn.functional.pad(...)` before `conv2d(...)` instead of dropping the ending spatial padding values.
- Added deep roundtrip tests that import ONNX graphs as TorchSharp modules, export them back to ONNX, and compare runtime outputs for classifier-head, convolution, and residual patterns.

## 0.3.1

- Added TorchModule initializer import support for ONNX `float16` and `bfloat16` tensors, using TorchSharp `BFloat16` support from TorchSharp `0.107.0` for generated bfloat16 loader paths.
- Expanded TorchModule generation to support ONNX graphs with multiple non-initializer inputs and multiple graph outputs.
- Added TorchModule reconstruction support for `ScatterND`, `ConvTranspose`, `SimplifiedLayerNormalization`, `GRU` initial hidden state inputs, and asymmetric convolution fallback through functional `conv2d`.

## 0.3.0

- Expanded TorchModule import support for ONNX graphs with broader runtime tensor data types that map to TorchSharp `ScalarType`, including non-float inputs and outputs.
- Added TorchModule import support for `GRU` and multi-output operators such as `Split` and `TopK`.
- Added TorchModule inline support for additional ONNX operators including `ArgMax`, `ArgMin`, `Celu`, `CumSum`, `DepthToSpace`, `Dropout`, `Expand`, `GatherElements`, `Gelu`, `GroupNormalization`, `HardSwish`, `InstanceNormalization`, `LayerNormalization`, `LogSoftmax`, `Mish`, `PRelu`, `Pad`, `ReduceMax`, `ReduceMin`, `ReduceProd`, `Resize`, `Selu`, `Slice`, `Softplus`, `SpaceToDepth`, `Tile`, and `Trilu`.
- TorchModule `LoadWeightsFromOnnx(...)` now canonicalizes the source graph and can fall back from initializer names to deterministic canonical initializer indexes when loading weights from structurally identical models with different value names.
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
