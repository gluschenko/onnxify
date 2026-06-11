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
