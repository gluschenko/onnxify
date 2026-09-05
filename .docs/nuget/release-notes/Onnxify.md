## 0.3.11

- Added `OnnxGraph.Clean(...)` with byte-backed `OnnxGraphCleanupFlags` and `OnnxGraphCleanupReport` for deterministic, idempotent liveness cleanup.
- Cleanup reports the names and categories of removed nodes, values, initializers, inputs, outputs, and annotations.
- Added sparse initializer round-tripping and nested graph cleanup support.

## 0.3.9

- Aligned the package version with the 0.3.9 Onnxify package family release.

## 0.3.6

- Updated the `netstandard2.0` compatibility dependency on `System.Text.Json` to `10.0.9`.
- Aligned the package version with the 0.3.6 Onnxify package family release.

## 0.3.5

- Aligned the package version with the 0.3.5 Onnxify package family release.

## 0.3.4

- Fixed ONNX attribute type detection for `OnnxTensor` subclasses so tensor-valued attributes, including `ConstantOfShape.value`, serialize correctly.

## 0.3.3

- Added `OnnxDimensionNone` and ONNX shape parsing support for dimensions whose protobuf value case is `None`.
- Preserved empty optional node input and output slots when loading, editing, and saving ONNX graphs, fixing operators such as `Resize` that rely on positional optional inputs.
- Improved graph string output and project-generation rendering for unknown `None` dimensions by displaying them as `[none]`.
- Added regression coverage for loading `OnnxModel`, saving it back to ONNX, and rendering `OnnxGraph.ToString()` with unknown dimensions and positional optional node inputs.

## 0.3.2

- Fixed ONNX graph loading for models whose `graph.value_info` repeats a graph input or output name, preserving the first loaded value metadata while marking the value as an input or output.
- Added regression coverage for loading an ONNX graph whose output is also present in `value_info`.
- Added a `netstandard2.0` target for core graph/model APIs so analyzer packages can share the same `OnnxModel` and `OnnxGraph` implementation.
- Added `NodeTypeResolutionStrategy.PreserveUntyped` for callers that need to inspect loaded graphs without projecting nodes into generated typed wrappers.
- Aligned the package version with the 0.3.2 Onnxify package family release.

## 0.3.0

- Aligned the package version with the 0.3.0 Onnxify package family release.
- Kept core ONNX model APIs aligned with the expanded TorchModule generation and package documentation updates in the 0.3.0 family.
- Added `OnnxGraph.SortTopologically()` to deterministically reorder graph nodes, initializers, loose edges, and value-info entries for structurally comparable ONNX graphs.

## 0.2.0

- Raised the default `OnnxModel.Create()` profile to standard ONNX opset 25 and IR version 11.
- Added graph editing APIs for marking existing `OnnxValue` instances as inputs or outputs, replacing and removing nodes and values, and removing tensors or loose edges.
- Removal helpers now clear matching node input/output references and prune unused loose edges to avoid dangling graph pieces after edits.

## 0.1.2

- Added async ONNX model file I/O with `OnnxModel.FromFileAsync(...)` and `SaveAsync(...)`.
- Added stream-based ONNX model I/O with `OnnxModel.FromStream(...)`, `FromStreamAsync(...)`, `Save(Stream)`, and `SaveAsync(Stream, ...)`.
- Aligned the package version with the 0.1.2 Onnxify package family release.

## 0.1.1

- Aligned the package version with the 0.1.1 Onnxify package family release.

## 0.0.0.1

- Initial release
