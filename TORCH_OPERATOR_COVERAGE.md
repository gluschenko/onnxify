# TorchSharp operator coverage

* Found: 90.16% (449/498)
* Onnxify.TorchSharp coverage: 83.53% (416/498)
* Onnxify.ModelGenerator coverage: 56.22% (280/498)
* Deep export support: 83.53% (416/498)
* Deep import support: 56.02% (279/498)

## Coverage Columns

* `Found` means the observer found a likely matching public TorchSharp API or module for the ONNXScript Torch operator name. This is a discovery signal, not an Onnxify implementation guarantee.
* `Onnxify.TorchSharp coverage` means `Onnxify.TorchSharp` declares exporter support for that Torch operator through `[TorchOp(...)]`, so TorchSharp code can be exported to ONNX through that converter path.
* `Onnxify.ModelGenerator coverage` means `Onnxify.ModelGenerator` declares reverse TorchModule reconstruction support through `[TorchSharpOp(...)]` or the shared canonical ONNX mapping resolves to actual deep-import registry support for that operator family.
* `Deep export support` means the exact ONNXScript Torch operator is registered in the actual `Onnxify.TorchSharp` deep-export coverage set through `[TorchOp(...)]`.
* `Deep import support` means the observer can map the ONNXScript Torch operator to expected ONNX `OpType` nodes and every mapped `OpType` is registered in the actual `Onnxify.ModelGenerator` TorchModule deep-import registries.
* `Onnxify.Tests tests` is the number of `[Fact]` / `[Theory]` test methods in `src/Onnxify.Tests` whose name or body mentions the ONNXScript operator, normalized TorchSharp API name, or a known operator alias.
* `✅` means the category is covered/found. `❌` means it is not covered/found.

| ONNXScript operator | TorchSharp module | Found | Onnxify.TorchSharp coverage | Onnxify.ModelGenerator coverage | Deep export support | Deep import support | Onnxify.Tests tests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `_operator::__lshift__` | `TorchSharp.torch+Tensor.bitwise_left_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `_operator::__rshift__` | `TorchSharp.torch+Tensor.bitwise_right_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `_operator::abs` | `TorchSharp.torch+Tensor.abs` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `_operator::add` | `TorchSharp.torch+Tensor.add` | ✅ | ✅ | ✅ | ✅ | ✅ | 13 |
| `_operator::and_` | `TorchSharp.torch+Tensor.bitwise_and` | ✅ | ✅ | ✅ | ✅ | ✅ | 85 |
| `_operator::eq` | `TorchSharp.torch+Tensor.eq` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `_operator::floordiv` | `TorchSharp.torch+Tensor.floor_divide` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `_operator::ge` | `TorchSharp.torch+Tensor.ge` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `_operator::getitem` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `_operator::gt` | `TorchSharp.torch+Tensor.gt` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `_operator::le` | `TorchSharp.torch+Tensor.le` | ✅ | ✅ | ✅ | ✅ | ✅ | 11 |
| `_operator::lt` | `TorchSharp.torch+Tensor.lt` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `_operator::mod` | `TorchSharp.torch+Tensor.remainder` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `_operator::mul` | `TorchSharp.torch+Tensor.mul` | ✅ | ✅ | ✅ | ✅ | ✅ | 16 |
| `_operator::ne` | `TorchSharp.torch+Tensor.ne` | ✅ | ✅ | ✅ | ✅ | ✅ | 32 |
| `_operator::neg` | `TorchSharp.torch+Tensor.neg` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `_operator::or_` | `TorchSharp.torch+Tensor.bitwise_or` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `_operator::pow` | `TorchSharp.torch+Tensor.pow` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `_operator::sub` | `TorchSharp.torch+Tensor.sub` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `_operator::truediv` | `TorchSharp.torch+Tensor.true_divide` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `aten::__lshift__.Scalar` | `TorchSharp.torch+Tensor.bitwise_left_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::__rshift__.Scalar` | `TorchSharp.torch+Tensor.bitwise_right_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::_conj` | `TorchSharp.torch+Tensor.conj` | ✅ | ✅ | ❌ | ✅ | ❌ | 1 |
| `aten::_embedding_bag` | `TorchSharp.Modules.EmbeddingBag` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::_embedding_bag_forward_only` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 6 |
| `aten::_fft_c2c` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::_fft_c2r` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::_fft_r2c` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::_linalg_det` | `TorchSharp.torch+Tensor.det` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::_local_scalar_dense` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 7 |
| `aten::_log_softmax` | `TorchSharp.Modules.LogSoftmax` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `aten::_native_batch_norm_legit` | `TorchSharp.Modules.BatchNorm` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `aten::_native_batch_norm_legit.no_stats` | `TorchSharp.Modules.BatchNorm` | ✅ | ✅ | ✅ | ✅ | ✅ | 12 |
| `aten::_native_batch_norm_legit_functional` | `TorchSharp.Modules.BatchNorm` | ✅ | ✅ | ✅ | ✅ | ✅ | 10 |
| `aten::_native_batch_norm_legit_no_training` | `TorchSharp.Modules.BatchNorm` | ✅ | ✅ | ✅ | ✅ | ✅ | 13 |
| `aten::_prelu_kernel` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::_scaled_dot_product_efficient_attention` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 2 |
| `aten::_scaled_dot_product_flash_attention` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 2 |
| `aten::_scaled_dot_product_flash_attention_for_cpu` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 86 |
| `aten::_softmax` | `TorchSharp.Modules.Softmax` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::_to_copy` | `TorchSharp.ModuleExtensionMethods.to` | ✅ | ✅ | ✅ | ✅ | ✅ | 62 |
| `aten::_unique` | `TorchSharp.torch+Tensor.unique` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::_unique2` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::_unsafe_index.Tensor` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 6 |
| `aten::_unsafe_index_put` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 6 |
| `aten::_unsafe_view` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 12 |
| `aten::_upsample_bicubic2d_aa` | `TorchSharp.Modules.Upsample` | ✅ | ❌ | ✅ | ❌ | ✅ | 2 |
| `aten::_upsample_bilinear2d_aa` | `TorchSharp.Modules.Upsample` | ✅ | ❌ | ✅ | ❌ | ✅ | 2 |
| `aten::abs` | `TorchSharp.torch+Tensor.abs` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::acos` | `TorchSharp.torch+Tensor.acos` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::acosh` | `TorchSharp.torch+Tensor.acosh` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::add.Scalar` | `TorchSharp.torch+Tensor.add` | ✅ | ✅ | ✅ | ✅ | ✅ | 10 |
| `aten::add.Tensor` | `TorchSharp.torch+Tensor.add` | ✅ | ✅ | ✅ | ✅ | ✅ | 11 |
| `aten::addbmm` | `TorchSharp.torch+Tensor.addbmm` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::addcdiv` | `TorchSharp.torch+Tensor.addcdiv` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::addcmul` | `TorchSharp.torch+Tensor.addcmul` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::addmm` | `TorchSharp.torch+Tensor.addmm` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::addmv` | `TorchSharp.torch+Tensor.addmv` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::addr` | `TorchSharp.torch+Tensor.addr` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::alias` | `TorchSharp.torch+Tensor.alias` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::all` | `TorchSharp.torch+Tensor.all` | ✅ | ✅ | ❌ | ✅ | ❌ | 10 |
| `aten::all.dim` | `TorchSharp.torch+Tensor.all` | ✅ | ✅ | ❌ | ✅ | ❌ | 10 |
| `aten::all.dims` | `TorchSharp.torch+Tensor.all` | ✅ | ✅ | ❌ | ✅ | ❌ | 10 |
| `aten::allclose` | `TorchSharp.torch+Tensor.allclose` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::amax` | `TorchSharp.torch+Tensor.amax` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::amin` | `TorchSharp.torch+Tensor.amin` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::angle` | `TorchSharp.torch+Tensor.angle` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::any` | `TorchSharp.torch+Tensor.any` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::any.dim` | `TorchSharp.torch+Tensor.any` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::any.dims` | `TorchSharp.torch+Tensor.any` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::arange` | `TorchSharp.torch.arange` | ✅ | ✅ | ❌ | ✅ | ❌ | 4 |
| `aten::arange.start` | `TorchSharp.torch.arange` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::arange.start_step` | `TorchSharp.torch.arange` | ✅ | ✅ | ❌ | ✅ | ❌ | 6 |
| `aten::argmax` | `TorchSharp.torch+Tensor.argmax` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::argmin` | `TorchSharp.torch+Tensor.argmin` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::as_strided` | `TorchSharp.torch+Tensor.as_strided` | ✅ | ✅ | ❌ | ✅ | ❌ | 10 |
| `aten::asin` | `TorchSharp.torch+Tensor.asin` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::asinh` | `TorchSharp.torch+Tensor.asinh` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::atan` | `TorchSharp.torch+Tensor.atan` | ✅ | ✅ | ✅ | ✅ | ✅ | 12 |
| `aten::atan2` | `TorchSharp.torch+Tensor.atan2` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::atanh` | `TorchSharp.torch+Tensor.atanh` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::atleast_1d` | `TorchSharp.torch+Tensor.atleast_1d` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::atleast_1d.Sequence` | `TorchSharp.torch+Tensor.atleast_1d` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::atleast_2d` | `TorchSharp.torch+Tensor.atleast_2d` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::atleast_2d.Sequence` | `TorchSharp.torch+Tensor.atleast_2d` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::atleast_3d` | `TorchSharp.torch+Tensor.atleast_3d` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::atleast_3d.Sequence` | `TorchSharp.torch+Tensor.atleast_3d` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::avg_pool1d` | `TorchSharp.Modules.AvgPool1d` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::avg_pool2d` | `TorchSharp.Modules.AvgPool2d` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::avg_pool3d` | `TorchSharp.Modules.AvgPool3d` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::baddbmm` | `TorchSharp.torch+Tensor.baddbmm` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::bernoulli` | `TorchSharp.torch+Tensor.bernoulli` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::bernoulli.p` | `TorchSharp.torch+Tensor.bernoulli` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::bilinear` | `TorchSharp.Modules.Bilinear` | ✅ | ❌ | ❌ | ❌ | ❌ | 1 |
| `aten::bitwise_and.Scalar` | `TorchSharp.torch+Tensor.bitwise_and` | ✅ | ✅ | ✅ | ✅ | ✅ | 84 |
| `aten::bitwise_and.Scalar_Tensor` | `TorchSharp.torch+Tensor.bitwise_and` | ✅ | ✅ | ✅ | ✅ | ✅ | 84 |
| `aten::bitwise_and.Tensor` | `TorchSharp.torch+Tensor.bitwise_and` | ✅ | ✅ | ✅ | ✅ | ✅ | 85 |
| `aten::bitwise_left_shift.Scalar_Tensor` | `TorchSharp.torch+Tensor.bitwise_left_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::bitwise_left_shift.Tensor` | `TorchSharp.torch+Tensor.bitwise_left_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::bitwise_left_shift.Tensor_Scalar` | `TorchSharp.torch+Tensor.bitwise_left_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::bitwise_not` | `TorchSharp.torch+Tensor.bitwise_not` | ✅ | ✅ | ✅ | ✅ | ✅ | 16 |
| `aten::bitwise_or.Scalar` | `TorchSharp.torch+Tensor.bitwise_or` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::bitwise_or.Scalar_Tensor` | `TorchSharp.torch+Tensor.bitwise_or` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::bitwise_or.Tensor` | `TorchSharp.torch+Tensor.bitwise_or` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::bitwise_right_shift.Scalar_Tensor` | `TorchSharp.torch+Tensor.bitwise_right_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::bitwise_right_shift.Tensor` | `TorchSharp.torch+Tensor.bitwise_right_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::bitwise_right_shift.Tensor_Scalar` | `TorchSharp.torch+Tensor.bitwise_right_shift` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::bitwise_xor.Scalar` | `TorchSharp.torch+Tensor.bitwise_xor` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::bitwise_xor.Scalar_Tensor` | `TorchSharp.torch+Tensor.bitwise_xor` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::bitwise_xor.Tensor` | `TorchSharp.torch+Tensor.bitwise_xor` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::blackman_window` | `TorchSharp.torch.blackman_window` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::bmm` | `TorchSharp.torch+Tensor.bmm` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::broadcast_to` | `TorchSharp.torch+Tensor.broadcast_to` | ✅ | ✅ | ✅ | ✅ | ✅ | 15 |
| `aten::cat` | `TorchSharp.torch+distributions+constraints.cat` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::ceil` | `TorchSharp.torch+Tensor.ceil` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::celu` | `TorchSharp.Modules.CELU` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::chunk` | `TorchSharp.torch+Tensor.chunk` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::clamp` | `TorchSharp.torch+Tensor.clamp` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `aten::clamp.Tensor` | `TorchSharp.torch+Tensor.clamp` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `aten::clamp_max` | `TorchSharp.torch+Tensor.clamp_max` | ✅ | ✅ | ✅ | ✅ | ✅ | 12 |
| `aten::clamp_max.Tensor` | `TorchSharp.torch+Tensor.clamp_max` | ✅ | ✅ | ✅ | ✅ | ✅ | 12 |
| `aten::clamp_min` | `TorchSharp.torch+Tensor.clamp_min` | ✅ | ✅ | ✅ | ✅ | ✅ | 10 |
| `aten::clamp_min.Tensor` | `TorchSharp.torch+Tensor.clamp_min` | ✅ | ✅ | ✅ | ✅ | ✅ | 10 |
| `aten::clone` | `TorchSharp.torch+Tensor.clone` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::col2im` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::complex` | `TorchSharp.torch.complex` | ✅ | ❌ | ❌ | ❌ | ❌ | 1 |
| `aten::concat` | `TorchSharp.torch.concat` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::concatenate` | `TorchSharp.torch.concatenate` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::conj` | `TorchSharp.torch+Tensor.conj` | ✅ | ✅ | ❌ | ✅ | ❌ | 1 |
| `aten::constant_pad_nd` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 4 |
| `aten::contiguous` | `TorchSharp.torch+Tensor.contiguous` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::conv1d` | `TorchSharp.Modules.Conv1d` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::conv2d` | `TorchSharp.Modules.Conv2d` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `aten::conv3d` | `TorchSharp.Modules.Conv3d` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::convolution` | `TorchSharp.Modules.Convolution` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::copy` | `TorchSharp.torch+Storage`1.copy_` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::cos` | `TorchSharp.torch+Tensor.cos` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::cosh` | `TorchSharp.torch+Tensor.cosh` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::cross` | `TorchSharp.torch+Tensor.cross` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::cross_entropy_loss` | `TorchSharp.Modules.CrossEntropyLoss` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::cumsum` | `TorchSharp.torch+Tensor.cumsum` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::deg2rad` | `TorchSharp.torch+Tensor.deg2rad` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::det` | `TorchSharp.torch+Tensor.det` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::detach` | `TorchSharp.torch+Tensor.detach` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::diagonal` | `TorchSharp.torch+Tensor.diagonal` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::diagonal_copy` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 3 |
| `aten::div.Scalar` | `TorchSharp.torch+Tensor.div` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `aten::div.Scalar_mode` | `TorchSharp.torch+Tensor.div` | ✅ | ✅ | ✅ | ✅ | ✅ | 23 |
| `aten::div.Tensor` | `TorchSharp.torch+Tensor.div` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `aten::div.Tensor_mode` | `TorchSharp.torch+Tensor.div` | ✅ | ✅ | ✅ | ✅ | ✅ | 23 |
| `aten::divide.Scalar` | `TorchSharp.torch+Tensor.divide` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::divide.Tensor` | `TorchSharp.torch+Tensor.divide` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::dot` | `TorchSharp.torch+Tensor.dot` | ✅ | ✅ | ✅ | ✅ | ✅ | 9 |
| `aten::dropout` | `TorchSharp.Modules.Dropout` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::einsum` | `TorchSharp.torch.einsum` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::elu` | `TorchSharp.Modules.ELU` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::embedding` | `TorchSharp.Modules.Embedding` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::embedding_bag` | `TorchSharp.Modules.EmbeddingBag` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::embedding_bag.padding_idx` | `TorchSharp.Modules.EmbeddingBag` | ✅ | ❌ | ❌ | ❌ | ❌ | 2 |
| `aten::embedding_renorm` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::empty.memory_format` | `TorchSharp.torch+Tensor.empty` | ✅ | ✅ | ❌ | ✅ | ❌ | 20 |
| `aten::empty_like` | `TorchSharp.torch+Tensor.empty_like` | ✅ | ✅ | ❌ | ✅ | ❌ | 11 |
| `aten::empty_strided` | `TorchSharp.torch.empty_strided` | ✅ | ✅ | ❌ | ✅ | ❌ | 9 |
| `aten::eq` | `TorchSharp.torch+Tensor.eq` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::eq.Scalar` | `TorchSharp.torch+Tensor.eq` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::eq.Tensor` | `TorchSharp.torch+Tensor.eq` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::equal` | `TorchSharp.torch+Tensor.equal` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::erf` | `TorchSharp.torch+Tensor.erf` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `aten::erfc` | `TorchSharp.torch+Tensor.erfc` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::exp` | `TorchSharp.torch+Tensor.exp` | ✅ | ✅ | ✅ | ✅ | ✅ | 10 |
| `aten::exp2` | `TorchSharp.torch+Tensor.exp2` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::expand` | `TorchSharp.torch+Tensor.expand` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::expand_as` | `TorchSharp.torch+Tensor.expand_as` | ✅ | ✅ | ✅ | ✅ | ✅ | 15 |
| `aten::expm1` | `TorchSharp.torch+Tensor.expm1` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::fake_quantize_per_channel_affine` | `TorchSharp.torch.fake_quantize_per_channel_affine` | ✅ | ❌ | ❌ | ❌ | ❌ | 5 |
| `aten::fake_quantize_per_tensor_affine` | `TorchSharp.torch.fake_quantize_per_tensor_affine` | ✅ | ❌ | ❌ | ❌ | ❌ | 5 |
| `aten::fake_quantize_per_tensor_affine.tensor_qparams` | `TorchSharp.torch.fake_quantize_per_tensor_affine` | ✅ | ❌ | ❌ | ❌ | ❌ | 5 |
| `aten::fill.Scalar` | `TorchSharp.torch+Storage`1.fill_` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::fill.Tensor` | `TorchSharp.torch+Storage`1.fill_` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::flatten.using_ints` | `TorchSharp.Modules.Flatten` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::flip` | `TorchSharp.torch+Tensor.flip` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::floor` | `TorchSharp.torch+Tensor.floor` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::floor_divide` | `TorchSharp.torch+Tensor.floor_divide` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::fmod.Scalar` | `TorchSharp.torch+Tensor.fmod` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::fmod.Tensor` | `TorchSharp.torch+Tensor.fmod` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::frac` | `TorchSharp.torch+Tensor.frac` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::full` | `TorchSharp.torch+Tensor.full` | ✅ | ✅ | ❌ | ✅ | ❌ | 4 |
| `aten::full_like` | `TorchSharp.torch+Tensor.full_like` | ✅ | ✅ | ❌ | ✅ | ❌ | 6 |
| `aten::gather` | `TorchSharp.torch+Tensor.gather` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::ge.Scalar` | `TorchSharp.torch+Tensor.ge` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::ge.Tensor` | `TorchSharp.torch+Tensor.ge` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::gelu` | `TorchSharp.Modules.GELU` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::getitem` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::glu` | `TorchSharp.Modules.GLU` | ✅ | ✅ | ❌ | ✅ | ❌ | 1 |
| `aten::greater.Tensor` | `TorchSharp.torch+Tensor.greater` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::greater_equal.Tensor` | `TorchSharp.torch+Tensor.greater_equal` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::grid_sampler` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::grid_sampler_2d` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::group_norm` | `TorchSharp.Modules.GroupNorm` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::gru.input` | `TorchSharp.Modules.GRU` | ✅ | ✅ | ✅ | ✅ | ✅ | 23 |
| `aten::gt.Scalar` | `TorchSharp.torch+Tensor.gt` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::gt.Tensor` | `TorchSharp.torch+Tensor.gt` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::hamming_window` | `TorchSharp.torch.hamming_window` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::hann_window` | `TorchSharp.torch.hann_window` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::hardsigmoid` | `TorchSharp.Modules.Hardsigmoid` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::hardswish` | `TorchSharp.Modules.Hardswish` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::hardtanh` | `TorchSharp.Modules.Hardtanh` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::hardtanh_backward` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::heaviside` | `TorchSharp.torch+Tensor.heaviside` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::histc` | `TorchSharp.torch+Tensor.histc` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::im2col` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::index.Tensor` | `TorchSharp.torch+Tensor.index` | ✅ | ❌ | ❌ | ❌ | ❌ | 5 |
| `aten::index_put` | `TorchSharp.torch+Tensor.index_put_` | ✅ | ❌ | ❌ | ❌ | ❌ | 5 |
| `aten::index_select` | `TorchSharp.torch+Tensor.index_select` | ✅ | ✅ | ❌ | ✅ | ❌ | 14 |
| `aten::instance_norm` | `TorchSharp.Modules.InstanceNorm` | ✅ | ✅ | ❌ | ✅ | ❌ | 9 |
| `aten::is_nonzero` | `TorchSharp.torch+Tensor.is_nonzero` | ✅ | ✅ | ❌ | ✅ | ❌ | 8 |
| `aten::isclose` | `TorchSharp.torch+Tensor.isclose` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::isfinite` | `TorchSharp.BFloat16.IsFinite` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::isinf` | `TorchSharp.torch+Tensor.isinf` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::isnan` | `TorchSharp.BFloat16.IsNaN` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::isneginf` | `TorchSharp.torch+Tensor.isneginf` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::isposinf` | `TorchSharp.torch+Tensor.isposinf` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::layer_norm` | `TorchSharp.Modules.LayerNorm` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::le.Scalar` | `TorchSharp.torch+Tensor.le` | ✅ | ✅ | ✅ | ✅ | ✅ | 10 |
| `aten::le.Tensor` | `TorchSharp.torch+Tensor.le` | ✅ | ✅ | ✅ | ✅ | ✅ | 11 |
| `aten::leaky_relu` | `TorchSharp.Modules.LeakyReLU` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::lerp.Scalar` | `TorchSharp.torch+Tensor.lerp` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::lerp.Tensor` | `TorchSharp.torch+Tensor.lerp` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::less.Tensor` | `TorchSharp.torch+Tensor.less` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::less_equal.Tensor` | `TorchSharp.torch+Tensor.less_equal` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `aten::lift_fresh_copy` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 2 |
| `aten::linalg_cross` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 2 |
| `aten::linalg_det` | `TorchSharp.torch+Tensor.det` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::linalg_vector_norm` | `TorchSharp.torch+linalg.vector_norm` | ✅ | ✅ | ❌ | ✅ | ❌ | 9 |
| `aten::linear` | `TorchSharp.Modules.Linear` | ✅ | ✅ | ✅ | ✅ | ✅ | 10 |
| `aten::linspace` | `TorchSharp.torch.linspace` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::log` | `TorchSharp.torch+Tensor.log` | ✅ | ✅ | ✅ | ✅ | ✅ | 9 |
| `aten::log10` | `TorchSharp.torch+Tensor.log10` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::log1p` | `TorchSharp.torch+Tensor.log1p` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::log2` | `TorchSharp.torch+Tensor.log2` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::log_sigmoid` | `TorchSharp.Modules.LogSigmoid` | ✅ | ✅ | ❌ | ✅ | ❌ | 9 |
| `aten::log_softmax.int` | `TorchSharp.Modules.LogSoftmax` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `aten::logaddexp` | `TorchSharp.torch+Tensor.logaddexp` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::logaddexp2` | `TorchSharp.torch+Tensor.logaddexp2` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::logcumsumexp` | `TorchSharp.torch+Tensor.logcumsumexp` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::logdet` | `TorchSharp.torch+Tensor.logdet` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::logical_and` | `TorchSharp.torch+Tensor.logical_and` | ✅ | ✅ | ✅ | ✅ | ✅ | 80 |
| `aten::logical_not` | `TorchSharp.torch+Tensor.logical_not` | ✅ | ✅ | ✅ | ✅ | ✅ | 14 |
| `aten::logical_or` | `TorchSharp.torch+Tensor.logical_or` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `aten::logical_xor` | `TorchSharp.torch+Tensor.logical_xor` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::logit` | `TorchSharp.torch+Tensor.logit` | ✅ | ✅ | ❌ | ✅ | ❌ | 4 |
| `aten::logsumexp` | `TorchSharp.torch+Tensor.logsumexp` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::lstm.input` | `TorchSharp.Modules.LSTM` | ✅ | ✅ | ✅ | ✅ | ✅ | 26 |
| `aten::lt.Scalar` | `TorchSharp.torch+Tensor.lt` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::lt.Tensor` | `TorchSharp.torch+Tensor.lt` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::mH` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::mT` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::masked_fill.Scalar` | `TorchSharp.torch+Tensor.masked_fill` | ✅ | ✅ | ❌ | ✅ | ❌ | 4 |
| `aten::masked_fill.Tensor` | `TorchSharp.torch+Tensor.masked_fill` | ✅ | ✅ | ❌ | ✅ | ❌ | 4 |
| `aten::masked_scatter` | `TorchSharp.torch+Tensor.masked_scatter` | ✅ | ❌ | ❌ | ❌ | ❌ | 2 |
| `aten::matmul` | `TorchSharp.torch+Tensor.matmul` | ✅ | ✅ | ✅ | ✅ | ✅ | 11 |
| `aten::max` | `TorchSharp.torch+Tensor.max` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `aten::max.dim` | `TorchSharp.torch+Tensor.max` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `aten::max_pool1d` | `TorchSharp.Modules.MaxPool1d` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::max_pool1d_with_indices` | `TorchSharp.torch+nn+functional.max_pool1d_with_indices` | ✅ | ❌ | ❌ | ❌ | ❌ | 66 |
| `aten::max_pool2d` | `TorchSharp.Modules.MaxPool2d` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `aten::max_pool2d_with_indices` | `TorchSharp.torch+nn+functional.max_pool2d_with_indices` | ✅ | ✅ | ❌ | ✅ | ❌ | 67 |
| `aten::max_pool3d` | `TorchSharp.Modules.MaxPool3d` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::max_pool3d_with_indices` | `TorchSharp.torch+nn+functional.max_pool3d_with_indices` | ✅ | ❌ | ❌ | ❌ | ❌ | 66 |
| `aten::maximum` | `TorchSharp.torch+Tensor.maximum` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::mean` | `TorchSharp.torch+Tensor.mean` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::mean.dim` | `TorchSharp.torch+Tensor.mean` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::min` | `TorchSharp.torch+Tensor.min` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::min.dim` | `TorchSharp.torch+Tensor.min` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::minimum` | `TorchSharp.torch+Tensor.minimum` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::mish` | `TorchSharp.Modules.Mish` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::mm` | `TorchSharp.torch+Tensor.mm` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::mse_loss` | `TorchSharp.Modules.MSELoss` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::mul` | `TorchSharp.torch+Tensor.mul` | ✅ | ✅ | ✅ | ✅ | ✅ | 16 |
| `aten::mul.Tensor` | `TorchSharp.torch+Tensor.mul` | ✅ | ✅ | ✅ | ✅ | ✅ | 15 |
| `aten::multinomial` | `TorchSharp.torch+Tensor.multinomial` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::multiply.Tensor` | `TorchSharp.torch+Tensor.multiply` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::mv` | `TorchSharp.torch+Tensor.mv` | ✅ | ✅ | ✅ | ✅ | ✅ | 9 |
| `aten::narrow` | `TorchSharp.torch+Tensor.narrow` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::native_batch_norm` | `TorchSharp.Modules.BatchNorm` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `aten::native_dropout` | `TorchSharp.Modules.Dropout` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::native_group_norm` | `TorchSharp.Modules.GroupNorm` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `aten::native_layer_norm` | `TorchSharp.Modules.LayerNorm` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::ne` | `TorchSharp.torch+Tensor.ne` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::ne.Scalar` | `TorchSharp.torch+Tensor.ne` | ✅ | ✅ | ✅ | ✅ | ✅ | 32 |
| `aten::ne.Tensor` | `TorchSharp.torch+Tensor.ne` | ✅ | ✅ | ✅ | ✅ | ✅ | 32 |
| `aten::neg` | `TorchSharp.torch+Tensor.neg` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::new_empty` | `TorchSharp.torch+Tensor.new_empty` | ✅ | ✅ | ❌ | ✅ | ❌ | 18 |
| `aten::new_empty_strided` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 18 |
| `aten::new_full` | `TorchSharp.torch+Tensor.new_full` | ✅ | ✅ | ❌ | ✅ | ❌ | 12 |
| `aten::new_ones` | `TorchSharp.torch+Tensor.new_ones` | ✅ | ✅ | ❌ | ✅ | ❌ | 13 |
| `aten::new_zeros` | `TorchSharp.torch+Tensor.new_zeros` | ✅ | ✅ | ❌ | ✅ | ❌ | 13 |
| `aten::nll_loss` | `TorchSharp.Modules.NLLLoss` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::nll_loss_forward` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 5 |
| `aten::nonzero` | `TorchSharp.torch+Tensor.nonzero` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::normal.Tensor_Tensor` | `TorchSharp.torch+Tensor.normal_` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::normal.Tensor_float` | `TorchSharp.torch+Tensor.normal_` | ✅ | ✅ | ❌ | ✅ | ❌ | 13 |
| `aten::normal.float_Tensor` | `TorchSharp.torch+Tensor.normal_` | ✅ | ✅ | ❌ | ✅ | ❌ | 13 |
| `aten::normal.float_float` | `TorchSharp.torch+Tensor.normal_` | ✅ | ✅ | ❌ | ✅ | ❌ | 13 |
| `aten::normal_functional` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 6 |
| `aten::ones` | `TorchSharp.torch+Tensor.ones` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::ones_like` | `TorchSharp.torch+Tensor.ones_like` | ✅ | ✅ | ❌ | ✅ | ❌ | 7 |
| `aten::pad` | `TorchSharp.torch+nn+functional.pad` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::permute` | `TorchSharp.torch+Tensor.permute` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::pixel_shuffle` | `TorchSharp.Modules.PixelShuffle` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::pixel_unshuffle` | `TorchSharp.Modules.PixelUnshuffle` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::polar` | `TorchSharp.torch.polar` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::pow.Scalar` | `TorchSharp.torch+Tensor.pow` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::pow.Tensor_Scalar` | `TorchSharp.torch+Tensor.pow` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::pow.Tensor_Tensor` | `TorchSharp.torch+Tensor.pow` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::prelu` | `TorchSharp.Modules.PReLU` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::prod` | `TorchSharp.torch+Tensor.prod` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::prod.dim_int` | `TorchSharp.torch+Tensor.prod` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::rad2deg` | `TorchSharp.torch+Tensor.rad2deg` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::rand` | `TorchSharp.torch.rand` | ✅ | ✅ | ❌ | ✅ | ❌ | 12 |
| `aten::rand_like` | `TorchSharp.torch+Tensor.rand_like` | ✅ | ✅ | ❌ | ✅ | ❌ | 14 |
| `aten::randint` | `TorchSharp.torch.randint` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::randint.low` | `TorchSharp.torch.randint` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::randint_like` | `TorchSharp.torch+Tensor.randint_like` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::randint_like.low_dtype` | `TorchSharp.torch+Tensor.randint_like` | ✅ | ✅ | ❌ | ✅ | ❌ | 9 |
| `aten::randn` | `TorchSharp.torch.randn` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::randn_like` | `TorchSharp.torch+Tensor.randn_like` | ✅ | ✅ | ❌ | ✅ | ❌ | 4 |
| `aten::reciprocal` | `TorchSharp.torch+Tensor.reciprocal` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::reflection_pad1d` | `TorchSharp.Modules.ReflectionPad1d` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::reflection_pad2d` | `TorchSharp.Modules.ReflectionPad2d` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::reflection_pad3d` | `TorchSharp.Modules.ReflectionPad3d` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::relu` | `TorchSharp.Modules.ReLU` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::relu6` | `TorchSharp.Modules.ReLU6` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::remainder.Scalar` | `TorchSharp.torch+Tensor.remainder` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::remainder.Scalar_Tensor` | `TorchSharp.torch+Tensor.remainder` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::remainder.Tensor` | `TorchSharp.torch+Tensor.remainder` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::repeat` | `TorchSharp.torch+Tensor.repeat` | ✅ | ✅ | ❌ | ✅ | ❌ | 6 |
| `aten::repeat_interleave.Tensor` | `TorchSharp.torch+Tensor.repeat_interleave` | ✅ | ✅ | ❌ | ✅ | ❌ | 6 |
| `aten::repeat_interleave.self_int` | `TorchSharp.torch+Tensor.repeat_interleave` | ✅ | ✅ | ❌ | ✅ | ❌ | 6 |
| `aten::replication_pad1d` | `TorchSharp.Modules.ReplicationPad1d` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::replication_pad2d` | `TorchSharp.Modules.ReplicationPad2d` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::replication_pad3d` | `TorchSharp.Modules.ReplicationPad3d` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::reshape` | `TorchSharp.torch+Tensor.reshape` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::resolve_conj` | `TorchSharp.torch+Tensor.resolve_conj` | ✅ | ✅ | ❌ | ✅ | ❌ | 9 |
| `aten::resolve_neg` | `TorchSharp.torch+Tensor.resolve_neg` | ✅ | ✅ | ❌ | ✅ | ❌ | 10 |
| `aten::roll` | `TorchSharp.torch+Tensor.roll` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::round` | `TorchSharp.torch+Tensor.round` | ✅ | ✅ | ✅ | ✅ | ✅ | 32 |
| `aten::round.decimals` | `TorchSharp.torch+Tensor.round` | ✅ | ✅ | ✅ | ✅ | ✅ | 32 |
| `aten::rsqrt` | `TorchSharp.torch+Tensor.rsqrt` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::scalar_tensor` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::scaled_dot_product_attention` | `TorchSharp.torch+nn+functional.scaled_dot_product_attention` | ✅ | ❌ | ❌ | ❌ | ❌ | 2 |
| `aten::scatter.src` | `TorchSharp.torch+Tensor.scatter` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::scatter.value` | `TorchSharp.torch+Tensor.scatter` | ✅ | ❌ | ❌ | ❌ | ❌ | 18 |
| `aten::scatter_add` | `TorchSharp.torch+Tensor.scatter_add` | ✅ | ❌ | ❌ | ❌ | ❌ | 10 |
| `aten::scatter_reduce.two` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 1 |
| `aten::select.int` | `TorchSharp.torch+Tensor.select` | ✅ | ✅ | ❌ | ✅ | ❌ | 9 |
| `aten::select_scatter` | `TorchSharp.torch+Tensor.select_scatter` | ✅ | ❌ | ❌ | ❌ | ❌ | 8 |
| `aten::selu` | `TorchSharp.Modules.SELU` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::sigmoid` | `TorchSharp.Modules.Sigmoid` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::sign` | `TorchSharp.torch+Tensor.sign` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::signbit` | `TorchSharp.torch+Tensor.signbit` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::silu` | `TorchSharp.Modules.SiLU` | ✅ | ✅ | ❌ | ✅ | ❌ | 0 |
| `aten::sin` | `TorchSharp.torch+Tensor.sin` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::sinc` | `TorchSharp.torch+Tensor.sinc` | ✅ | ✅ | ❌ | ✅ | ❌ | 6 |
| `aten::sinh` | `TorchSharp.torch+Tensor.sinh` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::slice.Tensor` | `TorchSharp.torch+Size.Slice` | ✅ | ✅ | ✅ | ✅ | ✅ | 13 |
| `aten::slice_scatter` | `TorchSharp.torch+Tensor.slice_scatter` | ✅ | ❌ | ❌ | ❌ | ❌ | 12 |
| `aten::softmax.int` | `TorchSharp.Modules.Softmax` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::softplus` | `TorchSharp.Modules.Softplus` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::sort` | `TorchSharp.torch+Tensor.sort` | ✅ | ✅ | ❌ | ✅ | ❌ | 25 |
| `aten::special_erf` | `TorchSharp.torch+Tensor.erf` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `aten::special_erfc` | `TorchSharp.torch+Tensor.erfc` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::special_erfcx` | `TorchSharp.torch+special.erfcx` | ✅ | ✅ | ❌ | ✅ | ❌ | 3 |
| `aten::special_expm1` | `TorchSharp.torch+Tensor.expm1` | ✅ | ✅ | ❌ | ✅ | ❌ | 4 |
| `aten::special_log_softmax` | `TorchSharp.Modules.LogSoftmax` | ✅ | ✅ | ✅ | ✅ | ✅ | 9 |
| `aten::special_sinc` | `TorchSharp.torch+Tensor.sinc` | ✅ | ✅ | ❌ | ✅ | ❌ | 6 |
| `aten::special_softmax` | `TorchSharp.Modules.Softmax` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::split` | `TorchSharp.torch+Tensor.split` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::split.Tensor` | `TorchSharp.torch+Tensor.split` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::split_with_sizes` | `TorchSharp.torch+Tensor.split` | ✅ | ✅ | ✅ | ✅ | ✅ | 65 |
| `aten::sqrt` | `TorchSharp.torch+Tensor.sqrt` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::squeeze` | `TorchSharp.torch+Tensor.squeeze` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::squeeze.dim` | `TorchSharp.torch+Tensor.squeeze` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::stack` | `TorchSharp.torch+distributions+constraints.stack` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::stft` | `TorchSharp.torch+Tensor.stft` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::sub.Scalar` | `TorchSharp.torch+Tensor.sub` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::sub.Tensor` | `TorchSharp.torch+Tensor.sub` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::subtract.Scalar` | `TorchSharp.torch+Tensor.subtract` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::subtract.Tensor` | `TorchSharp.torch+Tensor.subtract` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::sum` | `TorchSharp.torch+Tensor.sum` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::sum.dim_IntList` | `TorchSharp.torch+Tensor.sum` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `aten::sym_size.int` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 7 |
| `aten::sym_storage_offset` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 2 |
| `aten::t` | `TorchSharp.torch+Tensor.t` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `aten::tan` | `TorchSharp.torch+Tensor.tan` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::tanh` | `TorchSharp.Modules.Tanh` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::tensor.bool` | `TorchSharp.torch+TensorIndex.Tensor` | ✅ | ✅ | ❌ | ✅ | ❌ | 54 |
| `aten::tensor.float` | `TorchSharp.torch+TensorIndex.Tensor` | ✅ | ✅ | ❌ | ✅ | ❌ | 55 |
| `aten::tensor.int` | `TorchSharp.torch+TensorIndex.Tensor` | ✅ | ✅ | ❌ | ✅ | ❌ | 50 |
| `aten::tile` | `TorchSharp.torch+Tensor.tile` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::topk` | `TorchSharp.torch+Tensor.topk` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::transpose.int` | `TorchSharp.torch+Tensor.transpose` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `aten::tril` | `TorchSharp.torch+Tensor.tril` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::triu` | `TorchSharp.torch+Tensor.triu` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::true_divide.Scalar` | `TorchSharp.torch+Tensor.true_divide` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::true_divide.Tensor` | `TorchSharp.torch+Tensor.true_divide` | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| `aten::trunc` | `TorchSharp.torch+Tensor.trunc` | ✅ | ✅ | ❌ | ✅ | ❌ | 1 |
| `aten::type_as` | `TorchSharp.torch+Tensor.type_as` | ✅ | ✅ | ✅ | ✅ | ✅ | 48 |
| `aten::unbind.int` | `TorchSharp.torch+Tensor.unbind` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::unflatten.int` | `TorchSharp.Modules.Unflatten` | ✅ | ✅ | ❌ | ✅ | ❌ | 1 |
| `aten::unfold` | `TorchSharp.Modules.Unfold` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::unique_consecutive` | `TorchSharp.torch+Tensor.unique_consecutive` | ✅ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::unique_dim` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `aten::unsafe_split.Tensor` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 6 |
| `aten::unsqueeze` | `TorchSharp.torch+Tensor.unsqueeze` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::upsample_bicubic2d` | `TorchSharp.Modules.Upsample` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_bicubic2d.vec` | `TorchSharp.Modules.Upsample` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_bilinear2d` | `TorchSharp.Modules.Upsample` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_bilinear2d.vec` | `TorchSharp.Modules.Upsample` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_linear1d` | `TorchSharp.Modules.Upsample` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_nearest1d` | `TorchSharp.torch+nn+functional.upsample_nearest1d` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_nearest1d.vec` | `TorchSharp.torch+nn+functional.upsample_nearest1d` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_nearest2d` | `TorchSharp.torch+nn+functional.upsample_nearest2d` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_nearest2d.vec` | `TorchSharp.torch+nn+functional.upsample_nearest2d` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_nearest3d` | `TorchSharp.torch+nn+functional.upsample_nearest3d` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_nearest3d.vec` | `TorchSharp.torch+nn+functional.upsample_nearest3d` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_trilinear3d` | `TorchSharp.Modules.Upsample` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::upsample_trilinear3d.vec` | `TorchSharp.Modules.Upsample` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `aten::view` | `TorchSharp.torch+Tensor.view` | ✅ | ✅ | ✅ | ✅ | ✅ | 12 |
| `aten::view_as` | `TorchSharp.torch+Tensor.view_as` | ✅ | ✅ | ✅ | ✅ | ✅ | 21 |
| `aten::view_as_complex` | `TorchSharp.torch+Tensor.view_as_complex` | ✅ | ❌ | ❌ | ❌ | ❌ | 20 |
| `aten::view_as_complex_copy` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 21 |
| `aten::view_as_real` | `TorchSharp.torch+Tensor.view_as_real` | ✅ | ❌ | ❌ | ❌ | ❌ | 22 |
| `aten::view_as_real_copy` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 23 |
| `aten::view_copy` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 12 |
| `aten::where.Scalar` | `TorchSharp.torch+Tensor.where` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::where.ScalarOther` | `TorchSharp.torch+Tensor.where` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::where.ScalarSelf` | `TorchSharp.torch+Tensor.where` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `aten::where.self` | `TorchSharp.torch+Tensor.where` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `aten::xlogy.Scalar_Other` | `TorchSharp.torch+Tensor.xlogy` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::xlogy.Scalar_Self` | `TorchSharp.torch+Tensor.xlogy` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::xlogy.Tensor` | `TorchSharp.torch+Tensor.xlogy` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `aten::zeros` | `TorchSharp.torch+Tensor.zeros` | ✅ | ✅ | ❌ | ✅ | ❌ | 5 |
| `aten::zeros_like` | `TorchSharp.torch+Tensor.zeros_like` | ✅ | ✅ | ❌ | ✅ | ❌ | 7 |
| `math::ceil` | `TorchSharp.torch+Tensor.ceil` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `math::floor` | `TorchSharp.torch+Tensor.floor` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `math::trunc` | `TorchSharp.torch+Tensor.trunc` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `prims::abs` | `TorchSharp.torch+Tensor.abs` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `prims::acos` | `TorchSharp.torch+Tensor.acos` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `prims::acosh` | `TorchSharp.torch+Tensor.acosh` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `prims::add` | `TorchSharp.torch+Tensor.add` | ✅ | ✅ | ✅ | ✅ | ✅ | 11 |
| `prims::asin` | `TorchSharp.torch+Tensor.asin` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `prims::asinh` | `TorchSharp.torch+Tensor.asinh` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `prims::atan` | `TorchSharp.torch+Tensor.atan` | ✅ | ✅ | ✅ | ✅ | ✅ | 12 |
| `prims::atanh` | `TorchSharp.torch+Tensor.atanh` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `prims::broadcast_in_dim` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 8 |
| `prims::ceil` | `TorchSharp.torch+Tensor.ceil` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `prims::convert_element_type` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 41 |
| `prims::cos` | `TorchSharp.torch+Tensor.cos` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `prims::cosh` | `TorchSharp.torch+Tensor.cosh` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `prims::device_put` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `prims::div` | `TorchSharp.torch+Tensor.div` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `prims::eq` | `TorchSharp.torch+Tensor.eq` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `prims::erf` | `TorchSharp.torch+Tensor.erf` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `prims::exp` | `TorchSharp.torch+Tensor.exp` | ✅ | ✅ | ✅ | ✅ | ✅ | 10 |
| `prims::floor` | `TorchSharp.torch+Tensor.floor` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `prims::ge` | `TorchSharp.torch+Tensor.ge` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `prims::gt` | `TorchSharp.torch+Tensor.gt` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `prims::le` | `TorchSharp.torch+Tensor.le` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `prims::log` | `TorchSharp.torch+Tensor.log` | ✅ | ✅ | ✅ | ✅ | ✅ | 9 |
| `prims::lt` | `TorchSharp.torch+Tensor.lt` | ✅ | ✅ | ✅ | ✅ | ✅ | 1 |
| `prims::mul` | `TorchSharp.torch+Tensor.mul` | ✅ | ✅ | ✅ | ✅ | ✅ | 16 |
| `prims::ne` | `TorchSharp.torch+Tensor.ne` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `prims::neg` | `TorchSharp.torch+Tensor.neg` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `prims::pow` | `TorchSharp.torch+Tensor.pow` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `prims::reshape` | `TorchSharp.torch+Tensor.reshape` | ✅ | ✅ | ✅ | ✅ | ✅ | 7 |
| `prims::resize` |  | ❌ | ❌ | ✅ | ❌ | ❌ | 2 |
| `prims::round` | `TorchSharp.torch+Tensor.round` | ✅ | ✅ | ✅ | ✅ | ✅ | 32 |
| `prims::sin` | `TorchSharp.torch+Tensor.sin` | ✅ | ✅ | ✅ | ✅ | ✅ | 6 |
| `prims::sinh` | `TorchSharp.torch+Tensor.sinh` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `prims::sqrt` | `TorchSharp.torch+Tensor.sqrt` | ✅ | ✅ | ✅ | ✅ | ✅ | 2 |
| `prims::squeeze` | `TorchSharp.torch+Tensor.squeeze` | ✅ | ✅ | ✅ | ✅ | ✅ | 5 |
| `prims::sub` | `TorchSharp.torch+Tensor.sub` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `prims::sum` | `TorchSharp.torch+Tensor.sum` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `prims::tan` | `TorchSharp.torch+Tensor.tan` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `prims::tanh` | `TorchSharp.Modules.Tanh` | ✅ | ✅ | ✅ | ✅ | ✅ | 3 |
| `prims::transpose` | `TorchSharp.torch+Tensor.transpose` | ✅ | ✅ | ✅ | ✅ | ✅ | 8 |
| `prims::var` | `TorchSharp.torch+Tensor.var` | ✅ | ✅ | ❌ | ✅ | ❌ | 2 |
| `prims::where` | `TorchSharp.torch+Tensor.where` | ✅ | ✅ | ✅ | ✅ | ✅ | 4 |
| `quantized_decomposed::dequantize_per_tensor` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 5 |
| `quantized_decomposed::dequantize_per_tensor.tensor` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 5 |
| `quantized_decomposed::dequantize_per_tensor.tensor2` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 5 |
| `quantized_decomposed::quantize_per_tensor` | `TorchSharp.torch.quantize_per_tensor` | ✅ | ❌ | ❌ | ❌ | ❌ | 5 |
| `quantized_decomposed::quantize_per_tensor.tensor` | `TorchSharp.torch.quantize_per_tensor` | ✅ | ❌ | ❌ | ❌ | ❌ | 5 |
| `quantized_decomposed::quantize_per_tensor.tensor2` | `TorchSharp.torch.quantize_per_tensor` | ✅ | ❌ | ❌ | ❌ | ❌ | 5 |
| `torchvision::nms` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 0 |
| `torchvision::roi_align` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 3 |
| `torchvision::roi_pool` |  | ❌ | ❌ | ❌ | ❌ | ❌ | 3 |
