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
