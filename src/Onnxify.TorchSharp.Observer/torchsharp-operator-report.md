# TorchSharp operator coverage

Found: 83.73% (417/498)
Onnxify.TorchSharp coverage: 83.53% (416/498)
Onnxify.ModelGenerator coverage: 30.12% (150/498)

## Coverage Columns

- `Found` means the observer found a likely matching public TorchSharp API or module for the ONNXScript Torch operator name. This is a discovery signal, not an Onnxify implementation guarantee.
- `Onnxify.TorchSharp coverage` means `Onnxify.TorchSharp` declares exporter support for that Torch operator through `[TorchOp(...)]`, so TorchSharp code can be exported to ONNX through that converter path.
- `Onnxify.ModelGenerator coverage` means `Onnxify.ModelGenerator` declares reverse TorchModule reconstruction support through `[TorchSharpOp(...)]` for the matched TorchSharp API/module name or operator name, so an ONNX graph pattern can be regenerated as a TorchSharp module for that family.
- `✅` means the category is covered/found. `❌` means it is not covered/found.

| ONNXScript operator | TorchSharp module | Found | Onnxify.TorchSharp coverage | Onnxify.ModelGenerator coverage |
| --- | --- | --- | --- | --- |
| _operator::__lshift__ |  | ❌ | ✅ | ❌ |
| _operator::__rshift__ |  | ❌ | ✅ | ❌ |
| _operator::abs | TorchSharp.torch+Tensor.abs | ✅ | ✅ | ✅ |
| _operator::add | TorchSharp.torch+Tensor.add | ✅ | ✅ | ✅ |
| _operator::and_ |  | ❌ | ✅ | ✅ |
| _operator::eq | TorchSharp.torch+Tensor.eq | ✅ | ✅ | ❌ |
| _operator::floordiv |  | ❌ | ✅ | ❌ |
| _operator::ge | TorchSharp.torch+Tensor.ge | ✅ | ✅ | ❌ |
| _operator::getitem |  | ❌ | ❌ | ❌ |
| _operator::gt | TorchSharp.torch+Tensor.gt | ✅ | ✅ | ❌ |
| _operator::le | TorchSharp.torch+Tensor.le | ✅ | ✅ | ❌ |
| _operator::lt | TorchSharp.torch+Tensor.lt | ✅ | ✅ | ❌ |
| _operator::mod |  | ❌ | ✅ | ✅ |
| _operator::mul | TorchSharp.torch+Tensor.mul | ✅ | ✅ | ✅ |
| _operator::ne | TorchSharp.torch+Tensor.ne | ✅ | ✅ | ❌ |
| _operator::neg | TorchSharp.torch+Tensor.neg | ✅ | ✅ | ✅ |
| _operator::or_ |  | ❌ | ✅ | ✅ |
| _operator::pow | TorchSharp.torch+Tensor.pow | ✅ | ✅ | ✅ |
| _operator::sub | TorchSharp.torch+Tensor.sub | ✅ | ✅ | ✅ |
| _operator::truediv |  | ❌ | ✅ | ❌ |
| aten::__lshift__.Scalar |  | ❌ | ✅ | ❌ |
| aten::__rshift__.Scalar |  | ❌ | ✅ | ❌ |
| aten::_conj | TorchSharp.torch+Tensor.conj | ✅ | ✅ | ❌ |
| aten::_embedding_bag | TorchSharp.Modules.EmbeddingBag | ✅ | ❌ | ❌ |
| aten::_embedding_bag_forward_only |  | ❌ | ❌ | ❌ |
| aten::_fft_c2c |  | ❌ | ❌ | ❌ |
| aten::_fft_c2r |  | ❌ | ❌ | ❌ |
| aten::_fft_r2c |  | ❌ | ❌ | ❌ |
| aten::_linalg_det | TorchSharp.torch+Tensor.det | ✅ | ✅ | ✅ |
| aten::_local_scalar_dense |  | ❌ | ❌ | ❌ |
| aten::_log_softmax | TorchSharp.Modules.LogSoftmax | ✅ | ✅ | ✅ |
| aten::_native_batch_norm_legit |  | ❌ | ✅ | ❌ |
| aten::_native_batch_norm_legit.no_stats |  | ❌ | ✅ | ❌ |
| aten::_native_batch_norm_legit_functional |  | ❌ | ✅ | ❌ |
| aten::_native_batch_norm_legit_no_training |  | ❌ | ✅ | ❌ |
| aten::_prelu_kernel |  | ❌ | ❌ | ❌ |
| aten::_scaled_dot_product_efficient_attention |  | ❌ | ❌ | ❌ |
| aten::_scaled_dot_product_flash_attention |  | ❌ | ❌ | ❌ |
| aten::_scaled_dot_product_flash_attention_for_cpu |  | ❌ | ❌ | ❌ |
| aten::_softmax | TorchSharp.Modules.Softmax | ✅ | ✅ | ✅ |
| aten::_to_copy |  | ❌ | ✅ | ❌ |
| aten::_unique | TorchSharp.torch+Tensor.unique | ✅ | ❌ | ❌ |
| aten::_unique2 |  | ❌ | ❌ | ❌ |
| aten::_unsafe_index.Tensor |  | ❌ | ❌ | ❌ |
| aten::_unsafe_index_put |  | ❌ | ❌ | ❌ |
| aten::_unsafe_view |  | ❌ | ❌ | ❌ |
| aten::_upsample_bicubic2d_aa |  | ❌ | ❌ | ❌ |
| aten::_upsample_bilinear2d_aa |  | ❌ | ❌ | ❌ |
| aten::abs | TorchSharp.torch+Tensor.abs | ✅ | ✅ | ✅ |
| aten::acos | TorchSharp.torch+Tensor.acos | ✅ | ✅ | ✅ |
| aten::acosh | TorchSharp.torch+Tensor.acosh | ✅ | ✅ | ✅ |
| aten::add.Scalar | TorchSharp.torch+Tensor.add | ✅ | ✅ | ✅ |
| aten::add.Tensor | TorchSharp.torch+Tensor.add | ✅ | ✅ | ✅ |
| aten::addbmm | TorchSharp.torch+Tensor.addbmm | ✅ | ✅ | ❌ |
| aten::addcdiv | TorchSharp.torch+Tensor.addcdiv | ✅ | ✅ | ❌ |
| aten::addcmul | TorchSharp.torch+Tensor.addcmul | ✅ | ✅ | ❌ |
| aten::addmm | TorchSharp.torch+Tensor.addmm | ✅ | ✅ | ❌ |
| aten::addmv | TorchSharp.torch+Tensor.addmv | ✅ | ✅ | ❌ |
| aten::addr | TorchSharp.torch+Tensor.addr | ✅ | ✅ | ❌ |
| aten::alias | TorchSharp.torch+Tensor.alias | ✅ | ✅ | ❌ |
| aten::all | TorchSharp.torch+Tensor.all | ✅ | ✅ | ❌ |
| aten::all.dim | TorchSharp.torch+Tensor.all | ✅ | ✅ | ❌ |
| aten::all.dims | TorchSharp.torch+Tensor.all | ✅ | ✅ | ❌ |
| aten::allclose | TorchSharp.torch+Tensor.allclose | ✅ | ✅ | ❌ |
| aten::amax | TorchSharp.torch+Tensor.amax | ✅ | ✅ | ❌ |
| aten::amin | TorchSharp.torch+Tensor.amin | ✅ | ✅ | ❌ |
| aten::angle | TorchSharp.torch+Tensor.angle | ✅ | ✅ | ❌ |
| aten::any | TorchSharp.torch+Tensor.any | ✅ | ✅ | ❌ |
| aten::any.dim | TorchSharp.torch+Tensor.any | ✅ | ✅ | ❌ |
| aten::any.dims | TorchSharp.torch+Tensor.any | ✅ | ✅ | ❌ |
| aten::arange | TorchSharp.torch.arange | ✅ | ✅ | ❌ |
| aten::arange.start | TorchSharp.torch.arange | ✅ | ✅ | ❌ |
| aten::arange.start_step | TorchSharp.torch.arange | ✅ | ✅ | ❌ |
| aten::argmax | TorchSharp.torch+Tensor.argmax | ✅ | ✅ | ✅ |
| aten::argmin | TorchSharp.torch+Tensor.argmin | ✅ | ✅ | ✅ |
| aten::as_strided | TorchSharp.torch+Tensor.as_strided | ✅ | ✅ | ❌ |
| aten::asin | TorchSharp.torch+Tensor.asin | ✅ | ✅ | ✅ |
| aten::asinh | TorchSharp.torch+Tensor.asinh | ✅ | ✅ | ✅ |
| aten::atan | TorchSharp.torch+Tensor.atan | ✅ | ✅ | ✅ |
| aten::atan2 | TorchSharp.torch+Tensor.atan2 | ✅ | ✅ | ❌ |
| aten::atanh | TorchSharp.torch+Tensor.atanh | ✅ | ✅ | ✅ |
| aten::atleast_1d | TorchSharp.torch+Tensor.atleast_1d | ✅ | ✅ | ❌ |
| aten::atleast_1d.Sequence | TorchSharp.torch+Tensor.atleast_1d | ✅ | ✅ | ❌ |
| aten::atleast_2d | TorchSharp.torch+Tensor.atleast_2d | ✅ | ✅ | ❌ |
| aten::atleast_2d.Sequence | TorchSharp.torch+Tensor.atleast_2d | ✅ | ✅ | ❌ |
| aten::atleast_3d | TorchSharp.torch+Tensor.atleast_3d | ✅ | ✅ | ❌ |
| aten::atleast_3d.Sequence | TorchSharp.torch+Tensor.atleast_3d | ✅ | ✅ | ❌ |
| aten::avg_pool1d | TorchSharp.Modules.AvgPool1d | ✅ | ✅ | ❌ |
| aten::avg_pool2d | TorchSharp.Modules.AvgPool2d | ✅ | ✅ | ❌ |
| aten::avg_pool3d | TorchSharp.Modules.AvgPool3d | ✅ | ✅ | ❌ |
| aten::baddbmm | TorchSharp.torch+Tensor.baddbmm | ✅ | ✅ | ❌ |
| aten::bernoulli | TorchSharp.torch+Tensor.bernoulli | ✅ | ✅ | ❌ |
| aten::bernoulli.p | TorchSharp.torch+Tensor.bernoulli | ✅ | ✅ | ❌ |
| aten::bilinear | TorchSharp.Modules.Bilinear | ✅ | ❌ | ❌ |
| aten::bitwise_and.Scalar | TorchSharp.torch+Tensor.bitwise_and | ✅ | ✅ | ✅ |
| aten::bitwise_and.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_and | ✅ | ✅ | ✅ |
| aten::bitwise_and.Tensor | TorchSharp.torch+Tensor.bitwise_and | ✅ | ✅ | ✅ |
| aten::bitwise_left_shift.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_left_shift | ✅ | ✅ | ❌ |
| aten::bitwise_left_shift.Tensor | TorchSharp.torch+Tensor.bitwise_left_shift | ✅ | ✅ | ❌ |
| aten::bitwise_left_shift.Tensor_Scalar | TorchSharp.torch+Tensor.bitwise_left_shift | ✅ | ✅ | ❌ |
| aten::bitwise_not | TorchSharp.torch+Tensor.bitwise_not | ✅ | ✅ | ✅ |
| aten::bitwise_or.Scalar | TorchSharp.torch+Tensor.bitwise_or | ✅ | ✅ | ✅ |
| aten::bitwise_or.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_or | ✅ | ✅ | ✅ |
| aten::bitwise_or.Tensor | TorchSharp.torch+Tensor.bitwise_or | ✅ | ✅ | ✅ |
| aten::bitwise_right_shift.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_right_shift | ✅ | ✅ | ❌ |
| aten::bitwise_right_shift.Tensor | TorchSharp.torch+Tensor.bitwise_right_shift | ✅ | ✅ | ❌ |
| aten::bitwise_right_shift.Tensor_Scalar | TorchSharp.torch+Tensor.bitwise_right_shift | ✅ | ✅ | ❌ |
| aten::bitwise_xor.Scalar | TorchSharp.torch+Tensor.bitwise_xor | ✅ | ✅ | ✅ |
| aten::bitwise_xor.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_xor | ✅ | ✅ | ✅ |
| aten::bitwise_xor.Tensor | TorchSharp.torch+Tensor.bitwise_xor | ✅ | ✅ | ✅ |
| aten::blackman_window | TorchSharp.torch.blackman_window | ✅ | ✅ | ❌ |
| aten::bmm | TorchSharp.torch+Tensor.bmm | ✅ | ✅ | ❌ |
| aten::broadcast_to | TorchSharp.torch+Tensor.broadcast_to | ✅ | ✅ | ❌ |
| aten::cat | TorchSharp.torch+distributions+constraints.cat | ✅ | ✅ | ❌ |
| aten::ceil | TorchSharp.torch+Tensor.ceil | ✅ | ✅ | ✅ |
| aten::celu | TorchSharp.Modules.CELU | ✅ | ✅ | ✅ |
| aten::chunk | TorchSharp.torch+Tensor.chunk | ✅ | ✅ | ❌ |
| aten::clamp | TorchSharp.torch+Tensor.clamp | ✅ | ✅ | ❌ |
| aten::clamp.Tensor | TorchSharp.torch+Tensor.clamp | ✅ | ✅ | ❌ |
| aten::clamp_max | TorchSharp.torch+Tensor.clamp_max | ✅ | ✅ | ❌ |
| aten::clamp_max.Tensor | TorchSharp.torch+Tensor.clamp_max | ✅ | ✅ | ❌ |
| aten::clamp_min | TorchSharp.torch+Tensor.clamp_min | ✅ | ✅ | ❌ |
| aten::clamp_min.Tensor | TorchSharp.torch+Tensor.clamp_min | ✅ | ✅ | ❌ |
| aten::clone | TorchSharp.torch+Tensor.clone | ✅ | ✅ | ❌ |
| aten::col2im |  | ❌ | ❌ | ❌ |
| aten::complex | TorchSharp.torch.complex | ✅ | ❌ | ❌ |
| aten::concat | TorchSharp.torch.concat | ✅ | ✅ | ✅ |
| aten::concatenate | TorchSharp.torch.concatenate | ✅ | ✅ | ❌ |
| aten::conj | TorchSharp.torch+Tensor.conj | ✅ | ✅ | ❌ |
| aten::constant_pad_nd |  | ❌ | ❌ | ❌ |
| aten::contiguous | TorchSharp.torch+Tensor.contiguous | ✅ | ✅ | ❌ |
| aten::conv1d | TorchSharp.Modules.Conv1d | ✅ | ✅ | ❌ |
| aten::conv2d | TorchSharp.Modules.Conv2d | ✅ | ✅ | ✅ |
| aten::conv3d | TorchSharp.Modules.Conv3d | ✅ | ✅ | ❌ |
| aten::convolution | TorchSharp.Modules.Convolution | ✅ | ✅ | ❌ |
| aten::copy | TorchSharp.torch+Storage`1.copy_ | ✅ | ✅ | ❌ |
| aten::cos | TorchSharp.torch+Tensor.cos | ✅ | ✅ | ✅ |
| aten::cosh | TorchSharp.torch+Tensor.cosh | ✅ | ✅ | ✅ |
| aten::cross | TorchSharp.torch+Tensor.cross | ✅ | ✅ | ❌ |
| aten::cross_entropy_loss | TorchSharp.Modules.CrossEntropyLoss | ✅ | ✅ | ❌ |
| aten::cumsum | TorchSharp.torch+Tensor.cumsum | ✅ | ✅ | ✅ |
| aten::deg2rad | TorchSharp.torch+Tensor.deg2rad | ✅ | ✅ | ❌ |
| aten::det | TorchSharp.torch+Tensor.det | ✅ | ✅ | ✅ |
| aten::detach | TorchSharp.torch+Tensor.detach | ✅ | ✅ | ❌ |
| aten::diagonal | TorchSharp.torch+Tensor.diagonal | ✅ | ✅ | ❌ |
| aten::diagonal_copy |  | ❌ | ❌ | ❌ |
| aten::div.Scalar | TorchSharp.torch+Tensor.div | ✅ | ✅ | ✅ |
| aten::div.Scalar_mode | TorchSharp.torch+Tensor.div | ✅ | ✅ | ✅ |
| aten::div.Tensor | TorchSharp.torch+Tensor.div | ✅ | ✅ | ✅ |
| aten::div.Tensor_mode | TorchSharp.torch+Tensor.div | ✅ | ✅ | ✅ |
| aten::divide.Scalar | TorchSharp.torch+Tensor.divide | ✅ | ✅ | ❌ |
| aten::divide.Tensor | TorchSharp.torch+Tensor.divide | ✅ | ✅ | ❌ |
| aten::dot | TorchSharp.torch+Tensor.dot | ✅ | ✅ | ❌ |
| aten::dropout | TorchSharp.Modules.Dropout | ✅ | ✅ | ✅ |
| aten::einsum | TorchSharp.torch.einsum | ✅ | ❌ | ❌ |
| aten::elu | TorchSharp.Modules.ELU | ✅ | ✅ | ✅ |
| aten::embedding | TorchSharp.Modules.Embedding | ✅ | ✅ | ❌ |
| aten::embedding_bag | TorchSharp.Modules.EmbeddingBag | ✅ | ❌ | ❌ |
| aten::embedding_bag.padding_idx | TorchSharp.Modules.EmbeddingBag | ✅ | ❌ | ❌ |
| aten::embedding_renorm |  | ❌ | ❌ | ❌ |
| aten::empty.memory_format | TorchSharp.torch+Tensor.empty | ✅ | ✅ | ❌ |
| aten::empty_like | TorchSharp.torch+Tensor.empty_like | ✅ | ✅ | ❌ |
| aten::empty_strided | TorchSharp.torch.empty_strided | ✅ | ✅ | ❌ |
| aten::eq | TorchSharp.torch+Tensor.eq | ✅ | ✅ | ❌ |
| aten::eq.Scalar | TorchSharp.torch+Tensor.eq | ✅ | ✅ | ❌ |
| aten::eq.Tensor | TorchSharp.torch+Tensor.eq | ✅ | ✅ | ❌ |
| aten::equal | TorchSharp.torch+Tensor.equal | ✅ | ✅ | ✅ |
| aten::erf | TorchSharp.torch+Tensor.erf | ✅ | ✅ | ✅ |
| aten::erfc | TorchSharp.torch+Tensor.erfc | ✅ | ✅ | ❌ |
| aten::exp | TorchSharp.torch+Tensor.exp | ✅ | ✅ | ✅ |
| aten::exp2 | TorchSharp.torch+Tensor.exp2 | ✅ | ✅ | ❌ |
| aten::expand | TorchSharp.torch+Tensor.expand | ✅ | ✅ | ✅ |
| aten::expand_as | TorchSharp.torch+Tensor.expand_as | ✅ | ✅ | ❌ |
| aten::expm1 | TorchSharp.torch+Tensor.expm1 | ✅ | ✅ | ❌ |
| aten::fake_quantize_per_channel_affine | TorchSharp.torch.fake_quantize_per_channel_affine | ✅ | ❌ | ❌ |
| aten::fake_quantize_per_tensor_affine | TorchSharp.torch.fake_quantize_per_tensor_affine | ✅ | ❌ | ❌ |
| aten::fake_quantize_per_tensor_affine.tensor_qparams | TorchSharp.torch.fake_quantize_per_tensor_affine | ✅ | ❌ | ❌ |
| aten::fill.Scalar | TorchSharp.torch+Storage`1.fill_ | ✅ | ✅ | ❌ |
| aten::fill.Tensor | TorchSharp.torch+Storage`1.fill_ | ✅ | ✅ | ❌ |
| aten::flatten.using_ints | TorchSharp.Modules.Flatten | ✅ | ✅ | ✅ |
| aten::flip | TorchSharp.torch+Tensor.flip | ✅ | ✅ | ❌ |
| aten::floor | TorchSharp.torch+Tensor.floor | ✅ | ✅ | ✅ |
| aten::floor_divide | TorchSharp.torch+Tensor.floor_divide | ✅ | ✅ | ❌ |
| aten::fmod.Scalar | TorchSharp.torch+Tensor.fmod | ✅ | ✅ | ❌ |
| aten::fmod.Tensor | TorchSharp.torch+Tensor.fmod | ✅ | ✅ | ❌ |
| aten::frac | TorchSharp.torch+Tensor.frac | ✅ | ✅ | ❌ |
| aten::full | TorchSharp.torch+Tensor.full | ✅ | ✅ | ❌ |
| aten::full_like | TorchSharp.torch+Tensor.full_like | ✅ | ✅ | ❌ |
| aten::gather | TorchSharp.torch+Tensor.gather | ✅ | ✅ | ✅ |
| aten::ge.Scalar | TorchSharp.torch+Tensor.ge | ✅ | ✅ | ❌ |
| aten::ge.Tensor | TorchSharp.torch+Tensor.ge | ✅ | ✅ | ❌ |
| aten::gelu | TorchSharp.Modules.GELU | ✅ | ✅ | ✅ |
| aten::getitem |  | ❌ | ❌ | ❌ |
| aten::glu | TorchSharp.Modules.GLU | ✅ | ✅ | ❌ |
| aten::greater.Tensor | TorchSharp.torch+Tensor.greater | ✅ | ✅ | ✅ |
| aten::greater_equal.Tensor | TorchSharp.torch+Tensor.greater_equal | ✅ | ✅ | ❌ |
| aten::grid_sampler |  | ❌ | ❌ | ❌ |
| aten::grid_sampler_2d |  | ❌ | ❌ | ❌ |
| aten::group_norm | TorchSharp.Modules.GroupNorm | ✅ | ✅ | ❌ |
| aten::gru.input | TorchSharp.Modules.GRU | ✅ | ✅ | ✅ |
| aten::gt.Scalar | TorchSharp.torch+Tensor.gt | ✅ | ✅ | ❌ |
| aten::gt.Tensor | TorchSharp.torch+Tensor.gt | ✅ | ✅ | ❌ |
| aten::hamming_window | TorchSharp.torch.hamming_window | ✅ | ✅ | ❌ |
| aten::hann_window | TorchSharp.torch.hann_window | ✅ | ✅ | ❌ |
| aten::hardsigmoid | TorchSharp.Modules.Hardsigmoid | ✅ | ✅ | ✅ |
| aten::hardswish | TorchSharp.Modules.Hardswish | ✅ | ✅ | ✅ |
| aten::hardtanh | TorchSharp.Modules.Hardtanh | ✅ | ✅ | ❌ |
| aten::hardtanh_backward |  | ❌ | ❌ | ❌ |
| aten::heaviside | TorchSharp.torch+Tensor.heaviside | ✅ | ✅ | ❌ |
| aten::histc | TorchSharp.torch+Tensor.histc | ✅ | ❌ | ❌ |
| aten::im2col |  | ❌ | ❌ | ❌ |
| aten::index.Tensor | TorchSharp.torch+Tensor.index | ✅ | ❌ | ❌ |
| aten::index_put | TorchSharp.torch+Tensor.index_put_ | ✅ | ❌ | ❌ |
| aten::index_select | TorchSharp.torch+Tensor.index_select | ✅ | ✅ | ❌ |
| aten::instance_norm | TorchSharp.Modules.InstanceNorm | ✅ | ✅ | ❌ |
| aten::is_nonzero | TorchSharp.torch+Tensor.is_nonzero | ✅ | ✅ | ❌ |
| aten::isclose | TorchSharp.torch+Tensor.isclose | ✅ | ✅ | ❌ |
| aten::isfinite | TorchSharp.torch+Tensor.isfinite | ✅ | ✅ | ❌ |
| aten::isinf | TorchSharp.torch+Tensor.isinf | ✅ | ✅ | ✅ |
| aten::isnan | TorchSharp.torch+Tensor.isnan | ✅ | ✅ | ✅ |
| aten::isneginf | TorchSharp.torch+Tensor.isneginf | ✅ | ✅ | ❌ |
| aten::isposinf | TorchSharp.torch+Tensor.isposinf | ✅ | ✅ | ❌ |
| aten::layer_norm | TorchSharp.Modules.LayerNorm | ✅ | ✅ | ❌ |
| aten::le.Scalar | TorchSharp.torch+Tensor.le | ✅ | ✅ | ❌ |
| aten::le.Tensor | TorchSharp.torch+Tensor.le | ✅ | ✅ | ❌ |
| aten::leaky_relu | TorchSharp.Modules.LeakyReLU | ✅ | ✅ | ✅ |
| aten::lerp.Scalar | TorchSharp.torch+Tensor.lerp | ✅ | ✅ | ❌ |
| aten::lerp.Tensor | TorchSharp.torch+Tensor.lerp | ✅ | ✅ | ❌ |
| aten::less.Tensor | TorchSharp.torch+Tensor.less | ✅ | ✅ | ✅ |
| aten::less_equal.Tensor | TorchSharp.torch+Tensor.less_equal | ✅ | ✅ | ❌ |
| aten::lift_fresh_copy |  | ❌ | ❌ | ❌ |
| aten::linalg_cross |  | ❌ | ❌ | ❌ |
| aten::linalg_det | TorchSharp.torch+Tensor.det | ✅ | ✅ | ✅ |
| aten::linalg_vector_norm | TorchSharp.torch+linalg.vector_norm | ✅ | ✅ | ❌ |
| aten::linear | TorchSharp.Modules.Linear | ✅ | ✅ | ✅ |
| aten::linspace | TorchSharp.torch.linspace | ✅ | ✅ | ❌ |
| aten::log | TorchSharp.torch+Tensor.log | ✅ | ✅ | ✅ |
| aten::log10 | TorchSharp.torch+Tensor.log10 | ✅ | ✅ | ❌ |
| aten::log1p | TorchSharp.torch+Tensor.log1p | ✅ | ✅ | ❌ |
| aten::log2 | TorchSharp.torch+Tensor.log2 | ✅ | ✅ | ❌ |
| aten::log_sigmoid | TorchSharp.Modules.LogSigmoid | ✅ | ✅ | ❌ |
| aten::log_softmax.int | TorchSharp.Modules.LogSoftmax | ✅ | ✅ | ✅ |
| aten::logaddexp | TorchSharp.torch+Tensor.logaddexp | ✅ | ✅ | ❌ |
| aten::logaddexp2 | TorchSharp.torch+Tensor.logaddexp2 | ✅ | ✅ | ❌ |
| aten::logcumsumexp | TorchSharp.torch+Tensor.logcumsumexp | ✅ | ✅ | ❌ |
| aten::logdet | TorchSharp.torch+Tensor.logdet | ✅ | ✅ | ❌ |
| aten::logical_and | TorchSharp.torch+Tensor.logical_and | ✅ | ✅ | ❌ |
| aten::logical_not | TorchSharp.torch+Tensor.logical_not | ✅ | ✅ | ❌ |
| aten::logical_or | TorchSharp.torch+Tensor.logical_or | ✅ | ✅ | ❌ |
| aten::logical_xor | TorchSharp.torch+Tensor.logical_xor | ✅ | ✅ | ❌ |
| aten::logit | TorchSharp.torch+Tensor.logit | ✅ | ✅ | ❌ |
| aten::logsumexp | TorchSharp.torch+Tensor.logsumexp | ✅ | ✅ | ❌ |
| aten::lstm.input | TorchSharp.Modules.LSTM | ✅ | ✅ | ✅ |
| aten::lt.Scalar | TorchSharp.torch+Tensor.lt | ✅ | ✅ | ❌ |
| aten::lt.Tensor | TorchSharp.torch+Tensor.lt | ✅ | ✅ | ❌ |
| aten::mH |  | ❌ | ❌ | ❌ |
| aten::mT |  | ❌ | ❌ | ❌ |
| aten::masked_fill.Scalar | TorchSharp.torch+Tensor.masked_fill | ✅ | ✅ | ❌ |
| aten::masked_fill.Tensor | TorchSharp.torch+Tensor.masked_fill | ✅ | ✅ | ❌ |
| aten::masked_scatter | TorchSharp.torch+Tensor.masked_scatter | ✅ | ❌ | ❌ |
| aten::matmul | TorchSharp.torch+Tensor.matmul | ✅ | ✅ | ✅ |
| aten::max | TorchSharp.torch+Tensor.max | ✅ | ✅ | ✅ |
| aten::max.dim | TorchSharp.torch+Tensor.max | ✅ | ✅ | ✅ |
| aten::max_pool1d | TorchSharp.Modules.MaxPool1d | ✅ | ✅ | ❌ |
| aten::max_pool1d_with_indices | TorchSharp.torch+nn+functional.max_pool1d_with_indices | ✅ | ❌ | ❌ |
| aten::max_pool2d | TorchSharp.Modules.MaxPool2d | ✅ | ✅ | ✅ |
| aten::max_pool2d_with_indices | TorchSharp.torch+nn+functional.max_pool2d_with_indices | ✅ | ✅ | ❌ |
| aten::max_pool3d | TorchSharp.Modules.MaxPool3d | ✅ | ✅ | ❌ |
| aten::max_pool3d_with_indices | TorchSharp.torch+nn+functional.max_pool3d_with_indices | ✅ | ❌ | ❌ |
| aten::maximum | TorchSharp.torch+Tensor.maximum | ✅ | ✅ | ❌ |
| aten::mean | TorchSharp.torch+Tensor.mean | ✅ | ✅ | ❌ |
| aten::mean.dim | TorchSharp.torch+Tensor.mean | ✅ | ✅ | ❌ |
| aten::min | TorchSharp.torch+Tensor.min | ✅ | ✅ | ✅ |
| aten::min.dim | TorchSharp.torch+Tensor.min | ✅ | ✅ | ✅ |
| aten::minimum | TorchSharp.torch+Tensor.minimum | ✅ | ✅ | ❌ |
| aten::mish | TorchSharp.Modules.Mish | ✅ | ✅ | ✅ |
| aten::mm | TorchSharp.torch+Tensor.mm | ✅ | ✅ | ❌ |
| aten::mse_loss | TorchSharp.Modules.MSELoss | ✅ | ✅ | ❌ |
| aten::mul | TorchSharp.torch+Tensor.mul | ✅ | ✅ | ✅ |
| aten::mul.Tensor | TorchSharp.torch+Tensor.mul | ✅ | ✅ | ✅ |
| aten::multinomial | TorchSharp.torch+Tensor.multinomial | ✅ | ✅ | ❌ |
| aten::multiply.Tensor | TorchSharp.torch+Tensor.multiply | ✅ | ✅ | ❌ |
| aten::mv | TorchSharp.torch+Tensor.mv | ✅ | ✅ | ❌ |
| aten::narrow | TorchSharp.torch+Tensor.narrow | ✅ | ✅ | ❌ |
| aten::native_batch_norm |  | ❌ | ✅ | ❌ |
| aten::native_dropout |  | ❌ | ✅ | ❌ |
| aten::native_group_norm |  | ❌ | ✅ | ❌ |
| aten::native_layer_norm |  | ❌ | ✅ | ❌ |
| aten::ne | TorchSharp.torch+Tensor.ne | ✅ | ✅ | ❌ |
| aten::ne.Scalar | TorchSharp.torch+Tensor.ne | ✅ | ✅ | ❌ |
| aten::ne.Tensor | TorchSharp.torch+Tensor.ne | ✅ | ✅ | ❌ |
| aten::neg | TorchSharp.torch+Tensor.neg | ✅ | ✅ | ✅ |
| aten::new_empty | TorchSharp.torch+Tensor.new_empty | ✅ | ✅ | ❌ |
| aten::new_empty_strided |  | ❌ | ❌ | ❌ |
| aten::new_full | TorchSharp.torch+Tensor.new_full | ✅ | ✅ | ❌ |
| aten::new_ones | TorchSharp.torch+Tensor.new_ones | ✅ | ✅ | ❌ |
| aten::new_zeros | TorchSharp.torch+Tensor.new_zeros | ✅ | ✅ | ❌ |
| aten::nll_loss | TorchSharp.Modules.NLLLoss | ✅ | ✅ | ❌ |
| aten::nll_loss_forward |  | ❌ | ❌ | ❌ |
| aten::nonzero | TorchSharp.torch+Tensor.nonzero | ✅ | ✅ | ✅ |
| aten::normal.Tensor_Tensor | TorchSharp.torch+Tensor.normal_ | ✅ | ✅ | ❌ |
| aten::normal.Tensor_float | TorchSharp.torch+Tensor.normal_ | ✅ | ✅ | ❌ |
| aten::normal.float_Tensor | TorchSharp.torch+Tensor.normal_ | ✅ | ✅ | ❌ |
| aten::normal.float_float | TorchSharp.torch+Tensor.normal_ | ✅ | ✅ | ❌ |
| aten::normal_functional |  | ❌ | ❌ | ❌ |
| aten::ones | TorchSharp.torch+Tensor.ones | ✅ | ✅ | ❌ |
| aten::ones_like | TorchSharp.torch+Tensor.ones_like | ✅ | ✅ | ❌ |
| aten::pad | TorchSharp.torch+nn+functional.pad | ✅ | ✅ | ✅ |
| aten::permute | TorchSharp.torch+Tensor.permute | ✅ | ✅ | ❌ |
| aten::pixel_shuffle | TorchSharp.Modules.PixelShuffle | ✅ | ✅ | ❌ |
| aten::pixel_unshuffle | TorchSharp.Modules.PixelUnshuffle | ✅ | ✅ | ❌ |
| aten::polar | TorchSharp.torch.polar | ✅ | ❌ | ❌ |
| aten::pow.Scalar | TorchSharp.torch+Tensor.pow | ✅ | ✅ | ✅ |
| aten::pow.Tensor_Scalar | TorchSharp.torch+Tensor.pow | ✅ | ✅ | ✅ |
| aten::pow.Tensor_Tensor | TorchSharp.torch+Tensor.pow | ✅ | ✅ | ✅ |
| aten::prelu | TorchSharp.Modules.PReLU | ✅ | ✅ | ✅ |
| aten::prod | TorchSharp.torch+Tensor.prod | ✅ | ✅ | ❌ |
| aten::prod.dim_int | TorchSharp.torch+Tensor.prod | ✅ | ✅ | ❌ |
| aten::rad2deg | TorchSharp.torch+Tensor.rad2deg | ✅ | ✅ | ❌ |
| aten::rand | TorchSharp.torch.rand | ✅ | ✅ | ❌ |
| aten::rand_like | TorchSharp.torch+Tensor.rand_like | ✅ | ✅ | ❌ |
| aten::randint | TorchSharp.torch.randint | ✅ | ✅ | ❌ |
| aten::randint.low | TorchSharp.torch.randint | ✅ | ✅ | ❌ |
| aten::randint_like | TorchSharp.torch+Tensor.randint_like | ✅ | ✅ | ❌ |
| aten::randint_like.low_dtype | TorchSharp.torch+Tensor.randint_like | ✅ | ✅ | ❌ |
| aten::randn | TorchSharp.torch.randn | ✅ | ✅ | ❌ |
| aten::randn_like | TorchSharp.torch+Tensor.randn_like | ✅ | ✅ | ❌ |
| aten::reciprocal | TorchSharp.torch+Tensor.reciprocal | ✅ | ✅ | ✅ |
| aten::reflection_pad1d | TorchSharp.Modules.ReflectionPad1d | ✅ | ✅ | ❌ |
| aten::reflection_pad2d | TorchSharp.Modules.ReflectionPad2d | ✅ | ✅ | ❌ |
| aten::reflection_pad3d | TorchSharp.Modules.ReflectionPad3d | ✅ | ✅ | ❌ |
| aten::relu | TorchSharp.Modules.ReLU | ✅ | ✅ | ✅ |
| aten::relu6 | TorchSharp.Modules.ReLU6 | ✅ | ✅ | ✅ |
| aten::remainder.Scalar | TorchSharp.torch+Tensor.remainder | ✅ | ✅ | ❌ |
| aten::remainder.Scalar_Tensor | TorchSharp.torch+Tensor.remainder | ✅ | ✅ | ❌ |
| aten::remainder.Tensor | TorchSharp.torch+Tensor.remainder | ✅ | ✅ | ❌ |
| aten::repeat | TorchSharp.torch+Tensor.repeat | ✅ | ✅ | ❌ |
| aten::repeat_interleave.Tensor | TorchSharp.torch+Tensor.repeat_interleave | ✅ | ✅ | ❌ |
| aten::repeat_interleave.self_int | TorchSharp.torch+Tensor.repeat_interleave | ✅ | ✅ | ❌ |
| aten::replication_pad1d | TorchSharp.Modules.ReplicationPad1d | ✅ | ✅ | ❌ |
| aten::replication_pad2d | TorchSharp.Modules.ReplicationPad2d | ✅ | ✅ | ❌ |
| aten::replication_pad3d | TorchSharp.Modules.ReplicationPad3d | ✅ | ✅ | ❌ |
| aten::reshape | TorchSharp.torch+Tensor.reshape | ✅ | ✅ | ✅ |
| aten::resolve_conj | TorchSharp.torch+Tensor.resolve_conj | ✅ | ✅ | ❌ |
| aten::resolve_neg | TorchSharp.torch+Tensor.resolve_neg | ✅ | ✅ | ❌ |
| aten::roll | TorchSharp.torch+Tensor.roll | ✅ | ✅ | ❌ |
| aten::round | TorchSharp.torch+Tensor.round | ✅ | ✅ | ✅ |
| aten::round.decimals | TorchSharp.torch+Tensor.round | ✅ | ✅ | ✅ |
| aten::rsqrt | TorchSharp.torch+Tensor.rsqrt | ✅ | ✅ | ❌ |
| aten::scalar_tensor |  | ❌ | ❌ | ❌ |
| aten::scaled_dot_product_attention | TorchSharp.torch+nn+functional.scaled_dot_product_attention | ✅ | ❌ | ❌ |
| aten::scatter.src | TorchSharp.torch+Tensor.scatter | ✅ | ❌ | ❌ |
| aten::scatter.value | TorchSharp.torch+Tensor.scatter | ✅ | ❌ | ❌ |
| aten::scatter_add | TorchSharp.torch+Tensor.scatter_add | ✅ | ❌ | ❌ |
| aten::scatter_reduce.two |  | ❌ | ❌ | ❌ |
| aten::select.int | TorchSharp.torch+Tensor.select | ✅ | ✅ | ❌ |
| aten::select_scatter | TorchSharp.torch+Tensor.select_scatter | ✅ | ❌ | ❌ |
| aten::selu | TorchSharp.Modules.SELU | ✅ | ✅ | ✅ |
| aten::sigmoid | TorchSharp.Modules.Sigmoid | ✅ | ✅ | ✅ |
| aten::sign | TorchSharp.torch+Tensor.sign | ✅ | ✅ | ✅ |
| aten::signbit | TorchSharp.torch+Tensor.signbit | ✅ | ✅ | ❌ |
| aten::silu | TorchSharp.Modules.SiLU | ✅ | ✅ | ❌ |
| aten::sin | TorchSharp.torch+Tensor.sin | ✅ | ✅ | ✅ |
| aten::sinc | TorchSharp.torch+Tensor.sinc | ✅ | ✅ | ❌ |
| aten::sinh | TorchSharp.torch+Tensor.sinh | ✅ | ✅ | ✅ |
| aten::slice.Tensor | TorchSharp.torch+Size.Slice | ✅ | ✅ | ✅ |
| aten::slice_scatter | TorchSharp.torch+Tensor.slice_scatter | ✅ | ❌ | ❌ |
| aten::softmax.int | TorchSharp.Modules.Softmax | ✅ | ✅ | ✅ |
| aten::softplus | TorchSharp.Modules.Softplus | ✅ | ✅ | ✅ |
| aten::sort | TorchSharp.torch+Tensor.sort | ✅ | ✅ | ❌ |
| aten::special_erf | TorchSharp.torch+Tensor.erf | ✅ | ✅ | ✅ |
| aten::special_erfc | TorchSharp.torch+Tensor.erfc | ✅ | ✅ | ❌ |
| aten::special_erfcx | TorchSharp.torch+special.erfcx | ✅ | ✅ | ❌ |
| aten::special_expm1 | TorchSharp.torch+Tensor.expm1 | ✅ | ✅ | ❌ |
| aten::special_log_softmax | TorchSharp.Modules.LogSoftmax | ✅ | ✅ | ✅ |
| aten::special_sinc | TorchSharp.torch+Tensor.sinc | ✅ | ✅ | ❌ |
| aten::special_softmax |  | ❌ | ✅ | ❌ |
| aten::split | TorchSharp.torch+Tensor.split | ✅ | ✅ | ✅ |
| aten::split.Tensor | TorchSharp.torch+Tensor.split | ✅ | ✅ | ✅ |
| aten::split_with_sizes |  | ❌ | ✅ | ❌ |
| aten::sqrt | TorchSharp.torch+Tensor.sqrt | ✅ | ✅ | ✅ |
| aten::squeeze | TorchSharp.torch+Tensor.squeeze | ✅ | ✅ | ✅ |
| aten::squeeze.dim | TorchSharp.torch+Tensor.squeeze | ✅ | ✅ | ✅ |
| aten::stack | TorchSharp.torch+distributions+constraints.stack | ✅ | ✅ | ❌ |
| aten::stft | TorchSharp.torch+Tensor.stft | ✅ | ❌ | ❌ |
| aten::sub.Scalar | TorchSharp.torch+Tensor.sub | ✅ | ✅ | ✅ |
| aten::sub.Tensor | TorchSharp.torch+Tensor.sub | ✅ | ✅ | ✅ |
| aten::subtract.Scalar | TorchSharp.torch+Tensor.subtract | ✅ | ✅ | ❌ |
| aten::subtract.Tensor | TorchSharp.torch+Tensor.subtract | ✅ | ✅ | ❌ |
| aten::sum | TorchSharp.torch+Tensor.sum | ✅ | ✅ | ❌ |
| aten::sum.dim_IntList | TorchSharp.torch+Tensor.sum | ✅ | ✅ | ❌ |
| aten::sym_size.int |  | ❌ | ❌ | ❌ |
| aten::sym_storage_offset |  | ❌ | ❌ | ❌ |
| aten::t | TorchSharp.torch+Tensor.t | ✅ | ✅ | ❌ |
| aten::tan | TorchSharp.torch+Tensor.tan | ✅ | ✅ | ✅ |
| aten::tanh | TorchSharp.Modules.Tanh | ✅ | ✅ | ✅ |
| aten::tensor.bool | TorchSharp.torch+TensorIndex.Tensor | ✅ | ✅ | ❌ |
| aten::tensor.float | TorchSharp.torch+TensorIndex.Tensor | ✅ | ✅ | ❌ |
| aten::tensor.int | TorchSharp.torch+TensorIndex.Tensor | ✅ | ✅ | ❌ |
| aten::tile | TorchSharp.torch+Tensor.tile | ✅ | ✅ | ✅ |
| aten::topk | TorchSharp.torch+Tensor.topk | ✅ | ✅ | ✅ |
| aten::transpose.int | TorchSharp.torch+Tensor.transpose | ✅ | ✅ | ✅ |
| aten::tril | TorchSharp.torch+Tensor.tril | ✅ | ✅ | ❌ |
| aten::triu | TorchSharp.torch+Tensor.triu | ✅ | ✅ | ❌ |
| aten::true_divide.Scalar | TorchSharp.torch+Tensor.true_divide | ✅ | ✅ | ❌ |
| aten::true_divide.Tensor | TorchSharp.torch+Tensor.true_divide | ✅ | ✅ | ❌ |
| aten::trunc | TorchSharp.torch+Tensor.trunc | ✅ | ✅ | ❌ |
| aten::type_as | TorchSharp.torch+Tensor.type_as | ✅ | ✅ | ❌ |
| aten::unbind.int | TorchSharp.torch+Tensor.unbind | ✅ | ✅ | ❌ |
| aten::unflatten.int | TorchSharp.Modules.Unflatten | ✅ | ✅ | ❌ |
| aten::unfold | TorchSharp.Modules.Unfold | ✅ | ❌ | ❌ |
| aten::unique_consecutive | TorchSharp.torch+Tensor.unique_consecutive | ✅ | ❌ | ❌ |
| aten::unique_dim |  | ❌ | ❌ | ❌ |
| aten::unsafe_split.Tensor |  | ❌ | ❌ | ❌ |
| aten::unsqueeze | TorchSharp.torch+Tensor.unsqueeze | ✅ | ✅ | ✅ |
| aten::upsample_bicubic2d |  | ❌ | ✅ | ❌ |
| aten::upsample_bicubic2d.vec |  | ❌ | ✅ | ❌ |
| aten::upsample_bilinear2d |  | ❌ | ✅ | ❌ |
| aten::upsample_bilinear2d.vec |  | ❌ | ✅ | ❌ |
| aten::upsample_linear1d |  | ❌ | ✅ | ❌ |
| aten::upsample_nearest1d | TorchSharp.torch+nn+functional.upsample_nearest1d | ✅ | ✅ | ❌ |
| aten::upsample_nearest1d.vec | TorchSharp.torch+nn+functional.upsample_nearest1d | ✅ | ✅ | ❌ |
| aten::upsample_nearest2d | TorchSharp.torch+nn+functional.upsample_nearest2d | ✅ | ✅ | ❌ |
| aten::upsample_nearest2d.vec | TorchSharp.torch+nn+functional.upsample_nearest2d | ✅ | ✅ | ❌ |
| aten::upsample_nearest3d | TorchSharp.torch+nn+functional.upsample_nearest3d | ✅ | ✅ | ❌ |
| aten::upsample_nearest3d.vec | TorchSharp.torch+nn+functional.upsample_nearest3d | ✅ | ✅ | ❌ |
| aten::upsample_trilinear3d |  | ❌ | ✅ | ❌ |
| aten::upsample_trilinear3d.vec |  | ❌ | ✅ | ❌ |
| aten::view | TorchSharp.torch+Tensor.view | ✅ | ✅ | ❌ |
| aten::view_as | TorchSharp.torch+Tensor.view_as | ✅ | ✅ | ❌ |
| aten::view_as_complex | TorchSharp.torch+Tensor.view_as_complex | ✅ | ❌ | ❌ |
| aten::view_as_complex_copy |  | ❌ | ❌ | ❌ |
| aten::view_as_real | TorchSharp.torch+Tensor.view_as_real | ✅ | ❌ | ❌ |
| aten::view_as_real_copy |  | ❌ | ❌ | ❌ |
| aten::view_copy |  | ❌ | ❌ | ❌ |
| aten::where.Scalar | TorchSharp.torch+Tensor.where | ✅ | ✅ | ✅ |
| aten::where.ScalarOther | TorchSharp.torch+Tensor.where | ✅ | ✅ | ✅ |
| aten::where.ScalarSelf | TorchSharp.torch+Tensor.where | ✅ | ✅ | ✅ |
| aten::where.self | TorchSharp.torch+Tensor.where | ✅ | ✅ | ✅ |
| aten::xlogy.Scalar_Other | TorchSharp.torch+Tensor.xlogy | ✅ | ✅ | ❌ |
| aten::xlogy.Scalar_Self | TorchSharp.torch+Tensor.xlogy | ✅ | ✅ | ❌ |
| aten::xlogy.Tensor | TorchSharp.torch+Tensor.xlogy | ✅ | ✅ | ❌ |
| aten::zeros | TorchSharp.torch+Tensor.zeros | ✅ | ✅ | ❌ |
| aten::zeros_like | TorchSharp.torch+Tensor.zeros_like | ✅ | ✅ | ❌ |
| math::ceil | TorchSharp.torch+Tensor.ceil | ✅ | ✅ | ✅ |
| math::floor | TorchSharp.torch+Tensor.floor | ✅ | ✅ | ✅ |
| math::trunc | TorchSharp.torch+Tensor.trunc | ✅ | ✅ | ❌ |
| prims::abs | TorchSharp.torch+Tensor.abs | ✅ | ✅ | ✅ |
| prims::acos | TorchSharp.torch+Tensor.acos | ✅ | ✅ | ✅ |
| prims::acosh | TorchSharp.torch+Tensor.acosh | ✅ | ✅ | ✅ |
| prims::add | TorchSharp.torch+Tensor.add | ✅ | ✅ | ✅ |
| prims::asin | TorchSharp.torch+Tensor.asin | ✅ | ✅ | ✅ |
| prims::asinh | TorchSharp.torch+Tensor.asinh | ✅ | ✅ | ✅ |
| prims::atan | TorchSharp.torch+Tensor.atan | ✅ | ✅ | ✅ |
| prims::atanh | TorchSharp.torch+Tensor.atanh | ✅ | ✅ | ✅ |
| prims::broadcast_in_dim |  | ❌ | ❌ | ❌ |
| prims::ceil | TorchSharp.torch+Tensor.ceil | ✅ | ✅ | ✅ |
| prims::convert_element_type |  | ❌ | ❌ | ❌ |
| prims::cos | TorchSharp.torch+Tensor.cos | ✅ | ✅ | ✅ |
| prims::cosh | TorchSharp.torch+Tensor.cosh | ✅ | ✅ | ✅ |
| prims::device_put |  | ❌ | ❌ | ❌ |
| prims::div | TorchSharp.torch+Tensor.div | ✅ | ✅ | ✅ |
| prims::eq | TorchSharp.torch+Tensor.eq | ✅ | ✅ | ❌ |
| prims::erf | TorchSharp.torch+Tensor.erf | ✅ | ✅ | ✅ |
| prims::exp | TorchSharp.torch+Tensor.exp | ✅ | ✅ | ✅ |
| prims::floor | TorchSharp.torch+Tensor.floor | ✅ | ✅ | ✅ |
| prims::ge | TorchSharp.torch+Tensor.ge | ✅ | ✅ | ❌ |
| prims::gt | TorchSharp.torch+Tensor.gt | ✅ | ✅ | ❌ |
| prims::le | TorchSharp.torch+Tensor.le | ✅ | ✅ | ❌ |
| prims::log | TorchSharp.torch+Tensor.log | ✅ | ✅ | ✅ |
| prims::lt | TorchSharp.torch+Tensor.lt | ✅ | ✅ | ❌ |
| prims::mul | TorchSharp.torch+Tensor.mul | ✅ | ✅ | ✅ |
| prims::ne | TorchSharp.torch+Tensor.ne | ✅ | ✅ | ❌ |
| prims::neg | TorchSharp.torch+Tensor.neg | ✅ | ✅ | ✅ |
| prims::pow | TorchSharp.torch+Tensor.pow | ✅ | ✅ | ✅ |
| prims::reshape | TorchSharp.torch+Tensor.reshape | ✅ | ✅ | ✅ |
| prims::resize |  | ❌ | ❌ | ✅ |
| prims::round | TorchSharp.torch+Tensor.round | ✅ | ✅ | ✅ |
| prims::sin | TorchSharp.torch+Tensor.sin | ✅ | ✅ | ✅ |
| prims::sinh | TorchSharp.torch+Tensor.sinh | ✅ | ✅ | ✅ |
| prims::sqrt | TorchSharp.torch+Tensor.sqrt | ✅ | ✅ | ✅ |
| prims::squeeze | TorchSharp.torch+Tensor.squeeze | ✅ | ✅ | ✅ |
| prims::sub | TorchSharp.torch+Tensor.sub | ✅ | ✅ | ✅ |
| prims::sum | TorchSharp.torch+Tensor.sum | ✅ | ✅ | ❌ |
| prims::tan | TorchSharp.torch+Tensor.tan | ✅ | ✅ | ✅ |
| prims::tanh | TorchSharp.Modules.Tanh | ✅ | ✅ | ✅ |
| prims::transpose | TorchSharp.torch+Tensor.transpose | ✅ | ✅ | ✅ |
| prims::var | TorchSharp.torch+Tensor.var | ✅ | ✅ | ❌ |
| prims::where | TorchSharp.torch+Tensor.where | ✅ | ✅ | ✅ |
| quantized_decomposed::dequantize_per_tensor |  | ❌ | ❌ | ❌ |
| quantized_decomposed::dequantize_per_tensor.tensor |  | ❌ | ❌ | ❌ |
| quantized_decomposed::dequantize_per_tensor.tensor2 |  | ❌ | ❌ | ❌ |
| quantized_decomposed::quantize_per_tensor |  | ❌ | ❌ | ❌ |
| quantized_decomposed::quantize_per_tensor.tensor |  | ❌ | ❌ | ❌ |
| quantized_decomposed::quantize_per_tensor.tensor2 |  | ❌ | ❌ | ❌ |
| torchvision::nms |  | ❌ | ❌ | ❌ |
| torchvision::roi_align |  | ❌ | ❌ | ❌ |
| torchvision::roi_pool |  | ❌ | ❌ | ❌ |
