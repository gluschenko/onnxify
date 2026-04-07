# TorchSharp operator coverage

Found: 83.73% (417/498)
Coverage: 3.61% (18/498)

| ONNXScript operator | TorchSharp module | Found | Coverage |
| --- | --- | --- | --- |
| _operator::__lshift__ |  |  |  |
| _operator::__rshift__ |  |  |  |
| _operator::abs | TorchSharp.torch+Tensor.abs | &#10003; |  |
| _operator::add | TorchSharp.torch+Tensor.add | &#10003; |  |
| _operator::and_ |  |  |  |
| _operator::eq | TorchSharp.torch+Tensor.eq | &#10003; |  |
| _operator::floordiv |  |  |  |
| _operator::ge | TorchSharp.torch+Tensor.ge | &#10003; |  |
| _operator::getitem |  |  |  |
| _operator::gt | TorchSharp.torch+Tensor.gt | &#10003; |  |
| _operator::le | TorchSharp.torch+Tensor.le | &#10003; |  |
| _operator::lt | TorchSharp.torch+Tensor.lt | &#10003; |  |
| _operator::mod |  |  |  |
| _operator::mul | TorchSharp.torch+Tensor.mul | &#10003; |  |
| _operator::ne | TorchSharp.torch+Tensor.ne | &#10003; |  |
| _operator::neg | TorchSharp.torch+Tensor.neg | &#10003; |  |
| _operator::or_ |  |  |  |
| _operator::pow | TorchSharp.torch+Tensor.pow | &#10003; |  |
| _operator::sub | TorchSharp.torch+Tensor.sub | &#10003; |  |
| _operator::truediv |  |  |  |
| aten::__lshift__.Scalar |  |  |  |
| aten::__rshift__.Scalar |  |  |  |
| aten::_conj | TorchSharp.torch+Tensor.conj | &#10003; |  |
| aten::_embedding_bag | TorchSharp.Modules.EmbeddingBag | &#10003; |  |
| aten::_embedding_bag_forward_only |  |  |  |
| aten::_fft_c2c |  |  |  |
| aten::_fft_c2r |  |  |  |
| aten::_fft_r2c |  |  |  |
| aten::_linalg_det | TorchSharp.torch+Tensor.det | &#10003; |  |
| aten::_local_scalar_dense |  |  |  |
| aten::_log_softmax | TorchSharp.Modules.LogSoftmax | &#10003; |  |
| aten::_native_batch_norm_legit |  |  |  |
| aten::_native_batch_norm_legit.no_stats |  |  |  |
| aten::_native_batch_norm_legit_functional |  |  |  |
| aten::_native_batch_norm_legit_no_training |  |  |  |
| aten::_prelu_kernel |  |  |  |
| aten::_scaled_dot_product_efficient_attention |  |  |  |
| aten::_scaled_dot_product_flash_attention |  |  |  |
| aten::_scaled_dot_product_flash_attention_for_cpu |  |  |  |
| aten::_softmax | TorchSharp.Modules.Softmax | &#10003; |  |
| aten::_to_copy |  |  |  |
| aten::_unique | TorchSharp.torch+Tensor.unique | &#10003; |  |
| aten::_unique2 |  |  |  |
| aten::_unsafe_index.Tensor |  |  |  |
| aten::_unsafe_index_put |  |  |  |
| aten::_unsafe_view |  |  |  |
| aten::_upsample_bicubic2d_aa |  |  |  |
| aten::_upsample_bilinear2d_aa |  |  |  |
| aten::abs | TorchSharp.torch+Tensor.abs | &#10003; |  |
| aten::acos | TorchSharp.torch+Tensor.acos | &#10003; |  |
| aten::acosh | TorchSharp.torch+Tensor.acosh | &#10003; |  |
| aten::add.Scalar | TorchSharp.torch+Tensor.add | &#10003; |  |
| aten::add.Tensor | TorchSharp.torch+Tensor.add | &#10003; |  |
| aten::addbmm | TorchSharp.torch+Tensor.addbmm | &#10003; |  |
| aten::addcdiv | TorchSharp.torch+Tensor.addcdiv | &#10003; |  |
| aten::addcmul | TorchSharp.torch+Tensor.addcmul | &#10003; |  |
| aten::addmm | TorchSharp.torch+Tensor.addmm | &#10003; |  |
| aten::addmv | TorchSharp.torch+Tensor.addmv | &#10003; |  |
| aten::addr | TorchSharp.torch+Tensor.addr | &#10003; |  |
| aten::alias | TorchSharp.torch+Tensor.alias | &#10003; |  |
| aten::all | TorchSharp.torch+Tensor.all | &#10003; |  |
| aten::all.dim | TorchSharp.torch+Tensor.all | &#10003; |  |
| aten::all.dims | TorchSharp.torch+Tensor.all | &#10003; |  |
| aten::allclose | TorchSharp.torch+Tensor.allclose | &#10003; |  |
| aten::amax | TorchSharp.torch+Tensor.amax | &#10003; |  |
| aten::amin | TorchSharp.torch+Tensor.amin | &#10003; |  |
| aten::angle | TorchSharp.torch+Tensor.angle | &#10003; |  |
| aten::any | TorchSharp.torch+Tensor.any | &#10003; |  |
| aten::any.dim | TorchSharp.torch+Tensor.any | &#10003; |  |
| aten::any.dims | TorchSharp.torch+Tensor.any | &#10003; |  |
| aten::arange | TorchSharp.torch.arange | &#10003; |  |
| aten::arange.start | TorchSharp.torch.arange | &#10003; |  |
| aten::arange.start_step | TorchSharp.torch.arange | &#10003; |  |
| aten::argmax | TorchSharp.torch+Tensor.argmax | &#10003; |  |
| aten::argmin | TorchSharp.torch+Tensor.argmin | &#10003; |  |
| aten::as_strided | TorchSharp.torch+Tensor.as_strided | &#10003; |  |
| aten::asin | TorchSharp.torch+Tensor.asin | &#10003; |  |
| aten::asinh | TorchSharp.torch+Tensor.asinh | &#10003; |  |
| aten::atan | TorchSharp.torch+Tensor.atan | &#10003; |  |
| aten::atan2 | TorchSharp.torch+Tensor.atan2 | &#10003; |  |
| aten::atanh | TorchSharp.torch+Tensor.atanh | &#10003; |  |
| aten::atleast_1d | TorchSharp.torch+Tensor.atleast_1d | &#10003; |  |
| aten::atleast_1d.Sequence | TorchSharp.torch+Tensor.atleast_1d | &#10003; |  |
| aten::atleast_2d | TorchSharp.torch+Tensor.atleast_2d | &#10003; |  |
| aten::atleast_2d.Sequence | TorchSharp.torch+Tensor.atleast_2d | &#10003; |  |
| aten::atleast_3d | TorchSharp.torch+Tensor.atleast_3d | &#10003; |  |
| aten::atleast_3d.Sequence | TorchSharp.torch+Tensor.atleast_3d | &#10003; |  |
| aten::avg_pool1d | TorchSharp.Modules.AvgPool1d | &#10003; | &#10003; |
| aten::avg_pool2d | TorchSharp.Modules.AvgPool2d | &#10003; | &#10003; |
| aten::avg_pool3d | TorchSharp.Modules.AvgPool3d | &#10003; |  |
| aten::baddbmm | TorchSharp.torch+Tensor.baddbmm | &#10003; |  |
| aten::bernoulli | TorchSharp.torch+Tensor.bernoulli | &#10003; |  |
| aten::bernoulli.p | TorchSharp.torch+Tensor.bernoulli | &#10003; |  |
| aten::bilinear | TorchSharp.Modules.Bilinear | &#10003; |  |
| aten::bitwise_and.Scalar | TorchSharp.torch+Tensor.bitwise_and | &#10003; |  |
| aten::bitwise_and.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_and | &#10003; |  |
| aten::bitwise_and.Tensor | TorchSharp.torch+Tensor.bitwise_and | &#10003; |  |
| aten::bitwise_left_shift.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_left_shift | &#10003; |  |
| aten::bitwise_left_shift.Tensor | TorchSharp.torch+Tensor.bitwise_left_shift | &#10003; |  |
| aten::bitwise_left_shift.Tensor_Scalar | TorchSharp.torch+Tensor.bitwise_left_shift | &#10003; |  |
| aten::bitwise_not | TorchSharp.torch+Tensor.bitwise_not | &#10003; |  |
| aten::bitwise_or.Scalar | TorchSharp.torch+Tensor.bitwise_or | &#10003; |  |
| aten::bitwise_or.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_or | &#10003; |  |
| aten::bitwise_or.Tensor | TorchSharp.torch+Tensor.bitwise_or | &#10003; |  |
| aten::bitwise_right_shift.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_right_shift | &#10003; |  |
| aten::bitwise_right_shift.Tensor | TorchSharp.torch+Tensor.bitwise_right_shift | &#10003; |  |
| aten::bitwise_right_shift.Tensor_Scalar | TorchSharp.torch+Tensor.bitwise_right_shift | &#10003; |  |
| aten::bitwise_xor.Scalar | TorchSharp.torch+Tensor.bitwise_xor | &#10003; |  |
| aten::bitwise_xor.Scalar_Tensor | TorchSharp.torch+Tensor.bitwise_xor | &#10003; |  |
| aten::bitwise_xor.Tensor | TorchSharp.torch+Tensor.bitwise_xor | &#10003; |  |
| aten::blackman_window | TorchSharp.torch.blackman_window | &#10003; |  |
| aten::bmm | TorchSharp.torch+Tensor.bmm | &#10003; |  |
| aten::broadcast_to | TorchSharp.torch+Tensor.broadcast_to | &#10003; |  |
| aten::cat | TorchSharp.torch+distributions+constraints.cat | &#10003; |  |
| aten::ceil | TorchSharp.torch+Tensor.ceil | &#10003; |  |
| aten::celu | TorchSharp.Modules.CELU | &#10003; |  |
| aten::chunk | TorchSharp.torch+Tensor.chunk | &#10003; |  |
| aten::clamp | TorchSharp.torch+Tensor.clamp | &#10003; |  |
| aten::clamp.Tensor | TorchSharp.torch+Tensor.clamp | &#10003; |  |
| aten::clamp_max | TorchSharp.torch+Tensor.clamp_max | &#10003; |  |
| aten::clamp_max.Tensor | TorchSharp.torch+Tensor.clamp_max | &#10003; |  |
| aten::clamp_min | TorchSharp.torch+Tensor.clamp_min | &#10003; |  |
| aten::clamp_min.Tensor | TorchSharp.torch+Tensor.clamp_min | &#10003; |  |
| aten::clone | TorchSharp.torch+Tensor.clone | &#10003; |  |
| aten::col2im |  |  |  |
| aten::complex | TorchSharp.torch.complex | &#10003; |  |
| aten::concat | TorchSharp.torch.concat | &#10003; |  |
| aten::concatenate | TorchSharp.torch.concatenate | &#10003; |  |
| aten::conj | TorchSharp.torch+Tensor.conj | &#10003; |  |
| aten::constant_pad_nd |  |  |  |
| aten::contiguous | TorchSharp.torch+Tensor.contiguous | &#10003; |  |
| aten::conv1d | TorchSharp.Modules.Conv1d | &#10003; | &#10003; |
| aten::conv2d | TorchSharp.Modules.Conv2d | &#10003; | &#10003; |
| aten::conv3d | TorchSharp.Modules.Conv3d | &#10003; |  |
| aten::convolution | TorchSharp.Modules.Convolution | &#10003; |  |
| aten::copy | TorchSharp.torch+Storage`1.copy_ | &#10003; |  |
| aten::cos | TorchSharp.torch+Tensor.cos | &#10003; |  |
| aten::cosh | TorchSharp.torch+Tensor.cosh | &#10003; |  |
| aten::cross | TorchSharp.torch+Tensor.cross | &#10003; |  |
| aten::cross_entropy_loss | TorchSharp.Modules.CrossEntropyLoss | &#10003; |  |
| aten::cumsum | TorchSharp.torch+Tensor.cumsum | &#10003; |  |
| aten::deg2rad | TorchSharp.torch+Tensor.deg2rad | &#10003; |  |
| aten::det | TorchSharp.torch+Tensor.det | &#10003; |  |
| aten::detach | TorchSharp.torch+Tensor.detach | &#10003; |  |
| aten::diagonal | TorchSharp.torch+Tensor.diagonal | &#10003; |  |
| aten::diagonal_copy |  |  |  |
| aten::div.Scalar | TorchSharp.torch+Tensor.div | &#10003; |  |
| aten::div.Scalar_mode | TorchSharp.torch+Tensor.div | &#10003; |  |
| aten::div.Tensor | TorchSharp.torch+Tensor.div | &#10003; |  |
| aten::div.Tensor_mode | TorchSharp.torch+Tensor.div | &#10003; |  |
| aten::divide.Scalar | TorchSharp.torch+Tensor.divide | &#10003; |  |
| aten::divide.Tensor | TorchSharp.torch+Tensor.divide | &#10003; |  |
| aten::dot | TorchSharp.torch+Tensor.dot | &#10003; |  |
| aten::dropout | TorchSharp.Modules.Dropout | &#10003; | &#10003; |
| aten::einsum | TorchSharp.torch.einsum | &#10003; |  |
| aten::elu | TorchSharp.Modules.ELU | &#10003; | &#10003; |
| aten::embedding | TorchSharp.Modules.Embedding | &#10003; | &#10003; |
| aten::embedding_bag | TorchSharp.Modules.EmbeddingBag | &#10003; |  |
| aten::embedding_bag.padding_idx | TorchSharp.Modules.EmbeddingBag | &#10003; |  |
| aten::embedding_renorm |  |  |  |
| aten::empty.memory_format | TorchSharp.torch+Tensor.empty | &#10003; |  |
| aten::empty_like | TorchSharp.torch+Tensor.empty_like | &#10003; |  |
| aten::empty_strided | TorchSharp.torch.empty_strided | &#10003; |  |
| aten::eq | TorchSharp.torch+Tensor.eq | &#10003; |  |
| aten::eq.Scalar | TorchSharp.torch+Tensor.eq | &#10003; |  |
| aten::eq.Tensor | TorchSharp.torch+Tensor.eq | &#10003; |  |
| aten::equal | TorchSharp.torch+Tensor.equal | &#10003; |  |
| aten::erf | TorchSharp.torch+Tensor.erf | &#10003; |  |
| aten::erfc | TorchSharp.torch+Tensor.erfc | &#10003; |  |
| aten::exp | TorchSharp.torch+Tensor.exp | &#10003; |  |
| aten::exp2 | TorchSharp.torch+Tensor.exp2 | &#10003; |  |
| aten::expand | TorchSharp.torch+Tensor.expand | &#10003; |  |
| aten::expand_as | TorchSharp.torch+Tensor.expand_as | &#10003; |  |
| aten::expm1 | TorchSharp.torch+Tensor.expm1 | &#10003; |  |
| aten::fake_quantize_per_channel_affine | TorchSharp.torch.fake_quantize_per_channel_affine | &#10003; |  |
| aten::fake_quantize_per_tensor_affine | TorchSharp.torch.fake_quantize_per_tensor_affine | &#10003; |  |
| aten::fake_quantize_per_tensor_affine.tensor_qparams | TorchSharp.torch.fake_quantize_per_tensor_affine | &#10003; |  |
| aten::fill.Scalar | TorchSharp.torch+Storage`1.fill_ | &#10003; |  |
| aten::fill.Tensor | TorchSharp.torch+Storage`1.fill_ | &#10003; |  |
| aten::flatten.using_ints | TorchSharp.Modules.Flatten | &#10003; | &#10003; |
| aten::flip | TorchSharp.torch+Tensor.flip | &#10003; |  |
| aten::floor | TorchSharp.torch+Tensor.floor | &#10003; |  |
| aten::floor_divide | TorchSharp.torch+Tensor.floor_divide | &#10003; |  |
| aten::fmod.Scalar | TorchSharp.torch+Tensor.fmod | &#10003; |  |
| aten::fmod.Tensor | TorchSharp.torch+Tensor.fmod | &#10003; |  |
| aten::frac | TorchSharp.torch+Tensor.frac | &#10003; |  |
| aten::full | TorchSharp.torch+Tensor.full | &#10003; |  |
| aten::full_like | TorchSharp.torch+Tensor.full_like | &#10003; |  |
| aten::gather | TorchSharp.torch+Tensor.gather | &#10003; |  |
| aten::ge.Scalar | TorchSharp.torch+Tensor.ge | &#10003; |  |
| aten::ge.Tensor | TorchSharp.torch+Tensor.ge | &#10003; |  |
| aten::gelu | TorchSharp.Modules.GELU | &#10003; |  |
| aten::getitem |  |  |  |
| aten::glu | TorchSharp.Modules.GLU | &#10003; |  |
| aten::greater.Tensor | TorchSharp.torch+Tensor.greater | &#10003; |  |
| aten::greater_equal.Tensor | TorchSharp.torch+Tensor.greater_equal | &#10003; |  |
| aten::grid_sampler |  |  |  |
| aten::grid_sampler_2d |  |  |  |
| aten::group_norm | TorchSharp.Modules.GroupNorm | &#10003; |  |
| aten::gru.input | TorchSharp.Modules.GRU | &#10003; |  |
| aten::gt.Scalar | TorchSharp.torch+Tensor.gt | &#10003; |  |
| aten::gt.Tensor | TorchSharp.torch+Tensor.gt | &#10003; |  |
| aten::hamming_window | TorchSharp.torch.hamming_window | &#10003; |  |
| aten::hann_window | TorchSharp.torch.hann_window | &#10003; |  |
| aten::hardsigmoid | TorchSharp.Modules.Hardsigmoid | &#10003; |  |
| aten::hardswish | TorchSharp.Modules.Hardswish | &#10003; |  |
| aten::hardtanh | TorchSharp.Modules.Hardtanh | &#10003; |  |
| aten::hardtanh_backward |  |  |  |
| aten::heaviside | TorchSharp.torch+Tensor.heaviside | &#10003; |  |
| aten::histc | TorchSharp.torch+Tensor.histc | &#10003; |  |
| aten::im2col |  |  |  |
| aten::index.Tensor | TorchSharp.torch+Tensor.index | &#10003; |  |
| aten::index_put | TorchSharp.torch+Tensor.index_put_ | &#10003; |  |
| aten::index_select | TorchSharp.torch+Tensor.index_select | &#10003; |  |
| aten::instance_norm | TorchSharp.Modules.InstanceNorm | &#10003; |  |
| aten::is_nonzero | TorchSharp.torch+Tensor.is_nonzero | &#10003; |  |
| aten::isclose | TorchSharp.torch+Tensor.isclose | &#10003; |  |
| aten::isfinite | TorchSharp.torch+Tensor.isfinite | &#10003; |  |
| aten::isinf | TorchSharp.torch+Tensor.isinf | &#10003; |  |
| aten::isnan | TorchSharp.torch+Tensor.isnan | &#10003; |  |
| aten::isneginf | TorchSharp.torch+Tensor.isneginf | &#10003; |  |
| aten::isposinf | TorchSharp.torch+Tensor.isposinf | &#10003; |  |
| aten::layer_norm | TorchSharp.Modules.LayerNorm | &#10003; |  |
| aten::le.Scalar | TorchSharp.torch+Tensor.le | &#10003; |  |
| aten::le.Tensor | TorchSharp.torch+Tensor.le | &#10003; |  |
| aten::leaky_relu | TorchSharp.Modules.LeakyReLU | &#10003; | &#10003; |
| aten::lerp.Scalar | TorchSharp.torch+Tensor.lerp | &#10003; |  |
| aten::lerp.Tensor | TorchSharp.torch+Tensor.lerp | &#10003; |  |
| aten::less.Tensor | TorchSharp.torch+Tensor.less | &#10003; |  |
| aten::less_equal.Tensor | TorchSharp.torch+Tensor.less_equal | &#10003; |  |
| aten::lift_fresh_copy |  |  |  |
| aten::linalg_cross |  |  |  |
| aten::linalg_det | TorchSharp.torch+Tensor.det | &#10003; |  |
| aten::linalg_vector_norm | TorchSharp.torch+linalg.vector_norm | &#10003; |  |
| aten::linear | TorchSharp.Modules.Linear | &#10003; | &#10003; |
| aten::linspace | TorchSharp.torch.linspace | &#10003; |  |
| aten::log | TorchSharp.torch+Tensor.log | &#10003; |  |
| aten::log10 | TorchSharp.torch+Tensor.log10 | &#10003; |  |
| aten::log1p | TorchSharp.torch+Tensor.log1p | &#10003; |  |
| aten::log2 | TorchSharp.torch+Tensor.log2 | &#10003; |  |
| aten::log_sigmoid | TorchSharp.Modules.LogSigmoid | &#10003; |  |
| aten::log_softmax.int | TorchSharp.Modules.LogSoftmax | &#10003; | &#10003; |
| aten::logaddexp | TorchSharp.torch+Tensor.logaddexp | &#10003; |  |
| aten::logaddexp2 | TorchSharp.torch+Tensor.logaddexp2 | &#10003; |  |
| aten::logcumsumexp | TorchSharp.torch+Tensor.logcumsumexp | &#10003; |  |
| aten::logdet | TorchSharp.torch+Tensor.logdet | &#10003; |  |
| aten::logical_and | TorchSharp.torch+Tensor.logical_and | &#10003; |  |
| aten::logical_not | TorchSharp.torch+Tensor.logical_not | &#10003; |  |
| aten::logical_or | TorchSharp.torch+Tensor.logical_or | &#10003; |  |
| aten::logical_xor | TorchSharp.torch+Tensor.logical_xor | &#10003; |  |
| aten::logit | TorchSharp.torch+Tensor.logit | &#10003; |  |
| aten::logsumexp | TorchSharp.torch+Tensor.logsumexp | &#10003; |  |
| aten::lstm.input | TorchSharp.Modules.LSTM | &#10003; | &#10003; |
| aten::lt.Scalar | TorchSharp.torch+Tensor.lt | &#10003; |  |
| aten::lt.Tensor | TorchSharp.torch+Tensor.lt | &#10003; |  |
| aten::mH |  |  |  |
| aten::mT |  |  |  |
| aten::masked_fill.Scalar | TorchSharp.torch+Tensor.masked_fill | &#10003; |  |
| aten::masked_fill.Tensor | TorchSharp.torch+Tensor.masked_fill | &#10003; |  |
| aten::masked_scatter | TorchSharp.torch+Tensor.masked_scatter | &#10003; |  |
| aten::matmul | TorchSharp.torch+Tensor.matmul | &#10003; |  |
| aten::max | TorchSharp.torch+Tensor.max | &#10003; |  |
| aten::max.dim | TorchSharp.torch+Tensor.max | &#10003; |  |
| aten::max_pool1d | TorchSharp.Modules.MaxPool1d | &#10003; | &#10003; |
| aten::max_pool1d_with_indices | TorchSharp.torch+nn+functional.max_pool1d_with_indices | &#10003; |  |
| aten::max_pool2d | TorchSharp.Modules.MaxPool2d | &#10003; | &#10003; |
| aten::max_pool2d_with_indices | TorchSharp.torch+nn+functional.max_pool2d_with_indices | &#10003; |  |
| aten::max_pool3d | TorchSharp.Modules.MaxPool3d | &#10003; |  |
| aten::max_pool3d_with_indices | TorchSharp.torch+nn+functional.max_pool3d_with_indices | &#10003; |  |
| aten::maximum | TorchSharp.torch+Tensor.maximum | &#10003; |  |
| aten::mean | TorchSharp.torch+Tensor.mean | &#10003; |  |
| aten::mean.dim | TorchSharp.torch+Tensor.mean | &#10003; |  |
| aten::min | TorchSharp.torch+Tensor.min | &#10003; |  |
| aten::min.dim | TorchSharp.torch+Tensor.min | &#10003; |  |
| aten::minimum | TorchSharp.torch+Tensor.minimum | &#10003; |  |
| aten::mish | TorchSharp.Modules.Mish | &#10003; |  |
| aten::mm | TorchSharp.torch+Tensor.mm | &#10003; |  |
| aten::mse_loss | TorchSharp.Modules.MSELoss | &#10003; |  |
| aten::mul | TorchSharp.torch+Tensor.mul | &#10003; |  |
| aten::mul.Tensor | TorchSharp.torch+Tensor.mul | &#10003; |  |
| aten::multinomial | TorchSharp.torch+Tensor.multinomial | &#10003; |  |
| aten::multiply.Tensor | TorchSharp.torch+Tensor.multiply | &#10003; |  |
| aten::mv | TorchSharp.torch+Tensor.mv | &#10003; |  |
| aten::narrow | TorchSharp.torch+Tensor.narrow | &#10003; |  |
| aten::native_batch_norm |  |  |  |
| aten::native_dropout |  |  |  |
| aten::native_group_norm |  |  |  |
| aten::native_layer_norm |  |  |  |
| aten::ne | TorchSharp.torch+Tensor.ne | &#10003; |  |
| aten::ne.Scalar | TorchSharp.torch+Tensor.ne | &#10003; |  |
| aten::ne.Tensor | TorchSharp.torch+Tensor.ne | &#10003; |  |
| aten::neg | TorchSharp.torch+Tensor.neg | &#10003; |  |
| aten::new_empty | TorchSharp.torch+Tensor.new_empty | &#10003; |  |
| aten::new_empty_strided |  |  |  |
| aten::new_full | TorchSharp.torch+Tensor.new_full | &#10003; |  |
| aten::new_ones | TorchSharp.torch+Tensor.new_ones | &#10003; |  |
| aten::new_zeros | TorchSharp.torch+Tensor.new_zeros | &#10003; |  |
| aten::nll_loss | TorchSharp.Modules.NLLLoss | &#10003; |  |
| aten::nll_loss_forward |  |  |  |
| aten::nonzero | TorchSharp.torch+Tensor.nonzero | &#10003; |  |
| aten::normal.Tensor_Tensor | TorchSharp.torch+Tensor.normal_ | &#10003; |  |
| aten::normal.Tensor_float | TorchSharp.torch+Tensor.normal_ | &#10003; |  |
| aten::normal.float_Tensor | TorchSharp.torch+Tensor.normal_ | &#10003; |  |
| aten::normal.float_float | TorchSharp.torch+Tensor.normal_ | &#10003; |  |
| aten::normal_functional |  |  |  |
| aten::ones | TorchSharp.torch+Tensor.ones | &#10003; |  |
| aten::ones_like | TorchSharp.torch+Tensor.ones_like | &#10003; |  |
| aten::pad | TorchSharp.torch+nn+functional.pad | &#10003; |  |
| aten::permute | TorchSharp.torch+Tensor.permute | &#10003; |  |
| aten::pixel_shuffle | TorchSharp.Modules.PixelShuffle | &#10003; |  |
| aten::pixel_unshuffle | TorchSharp.Modules.PixelUnshuffle | &#10003; |  |
| aten::polar | TorchSharp.torch.polar | &#10003; |  |
| aten::pow.Scalar | TorchSharp.torch+Tensor.pow | &#10003; |  |
| aten::pow.Tensor_Scalar | TorchSharp.torch+Tensor.pow | &#10003; |  |
| aten::pow.Tensor_Tensor | TorchSharp.torch+Tensor.pow | &#10003; |  |
| aten::prelu | TorchSharp.Modules.PReLU | &#10003; |  |
| aten::prod | TorchSharp.torch+Tensor.prod | &#10003; |  |
| aten::prod.dim_int | TorchSharp.torch+Tensor.prod | &#10003; |  |
| aten::rad2deg | TorchSharp.torch+Tensor.rad2deg | &#10003; |  |
| aten::rand | TorchSharp.torch.rand | &#10003; |  |
| aten::rand_like | TorchSharp.torch+Tensor.rand_like | &#10003; |  |
| aten::randint | TorchSharp.torch.randint | &#10003; |  |
| aten::randint.low | TorchSharp.torch.randint | &#10003; |  |
| aten::randint_like | TorchSharp.torch+Tensor.randint_like | &#10003; |  |
| aten::randint_like.low_dtype | TorchSharp.torch+Tensor.randint_like | &#10003; |  |
| aten::randn | TorchSharp.torch.randn | &#10003; |  |
| aten::randn_like | TorchSharp.torch+Tensor.randn_like | &#10003; |  |
| aten::reciprocal | TorchSharp.torch+Tensor.reciprocal | &#10003; |  |
| aten::reflection_pad1d | TorchSharp.Modules.ReflectionPad1d | &#10003; |  |
| aten::reflection_pad2d | TorchSharp.Modules.ReflectionPad2d | &#10003; |  |
| aten::reflection_pad3d | TorchSharp.Modules.ReflectionPad3d | &#10003; |  |
| aten::relu | TorchSharp.Modules.ReLU | &#10003; | &#10003; |
| aten::relu6 | TorchSharp.Modules.ReLU6 | &#10003; |  |
| aten::remainder.Scalar | TorchSharp.torch+Tensor.remainder | &#10003; |  |
| aten::remainder.Scalar_Tensor | TorchSharp.torch+Tensor.remainder | &#10003; |  |
| aten::remainder.Tensor | TorchSharp.torch+Tensor.remainder | &#10003; |  |
| aten::repeat | TorchSharp.torch+Tensor.repeat | &#10003; |  |
| aten::repeat_interleave.Tensor | TorchSharp.torch+Tensor.repeat_interleave | &#10003; |  |
| aten::repeat_interleave.self_int | TorchSharp.torch+Tensor.repeat_interleave | &#10003; |  |
| aten::replication_pad1d | TorchSharp.Modules.ReplicationPad1d | &#10003; |  |
| aten::replication_pad2d | TorchSharp.Modules.ReplicationPad2d | &#10003; |  |
| aten::replication_pad3d | TorchSharp.Modules.ReplicationPad3d | &#10003; |  |
| aten::reshape | TorchSharp.torch+Tensor.reshape | &#10003; |  |
| aten::resolve_conj | TorchSharp.torch+Tensor.resolve_conj | &#10003; |  |
| aten::resolve_neg | TorchSharp.torch+Tensor.resolve_neg | &#10003; |  |
| aten::roll | TorchSharp.torch+Tensor.roll | &#10003; |  |
| aten::round | TorchSharp.torch+Tensor.round | &#10003; |  |
| aten::round.decimals | TorchSharp.torch+Tensor.round | &#10003; |  |
| aten::rsqrt | TorchSharp.torch+Tensor.rsqrt | &#10003; |  |
| aten::scalar_tensor |  |  |  |
| aten::scaled_dot_product_attention | TorchSharp.torch+nn+functional.scaled_dot_product_attention | &#10003; |  |
| aten::scatter.src | TorchSharp.torch+Tensor.scatter | &#10003; |  |
| aten::scatter.value | TorchSharp.torch+Tensor.scatter | &#10003; |  |
| aten::scatter_add | TorchSharp.torch+Tensor.scatter_add | &#10003; |  |
| aten::scatter_reduce.two |  |  |  |
| aten::select.int | TorchSharp.torch+Tensor.select | &#10003; |  |
| aten::select_scatter | TorchSharp.torch+Tensor.select_scatter | &#10003; |  |
| aten::selu | TorchSharp.Modules.SELU | &#10003; |  |
| aten::sigmoid | TorchSharp.Modules.Sigmoid | &#10003; | &#10003; |
| aten::sign | TorchSharp.torch+Tensor.sign | &#10003; |  |
| aten::signbit | TorchSharp.torch+Tensor.signbit | &#10003; |  |
| aten::silu | TorchSharp.Modules.SiLU | &#10003; |  |
| aten::sin | TorchSharp.torch+Tensor.sin | &#10003; |  |
| aten::sinc | TorchSharp.torch+Tensor.sinc | &#10003; |  |
| aten::sinh | TorchSharp.torch+Tensor.sinh | &#10003; |  |
| aten::slice.Tensor | TorchSharp.torch+Size.Slice | &#10003; |  |
| aten::slice_scatter | TorchSharp.torch+Tensor.slice_scatter | &#10003; |  |
| aten::softmax.int | TorchSharp.Modules.Softmax | &#10003; | &#10003; |
| aten::softplus | TorchSharp.Modules.Softplus | &#10003; |  |
| aten::sort | TorchSharp.torch+Tensor.sort | &#10003; |  |
| aten::special_erf | TorchSharp.torch+Tensor.erf | &#10003; |  |
| aten::special_erfc | TorchSharp.torch+Tensor.erfc | &#10003; |  |
| aten::special_erfcx | TorchSharp.torch+special.erfcx | &#10003; |  |
| aten::special_expm1 | TorchSharp.torch+Tensor.expm1 | &#10003; |  |
| aten::special_log_softmax | TorchSharp.Modules.LogSoftmax | &#10003; |  |
| aten::special_sinc | TorchSharp.torch+Tensor.sinc | &#10003; |  |
| aten::special_softmax |  |  |  |
| aten::split | TorchSharp.torch+Tensor.split | &#10003; |  |
| aten::split.Tensor | TorchSharp.torch+Tensor.split | &#10003; |  |
| aten::split_with_sizes |  |  |  |
| aten::sqrt | TorchSharp.torch+Tensor.sqrt | &#10003; |  |
| aten::squeeze | TorchSharp.torch+Tensor.squeeze | &#10003; |  |
| aten::squeeze.dim | TorchSharp.torch+Tensor.squeeze | &#10003; |  |
| aten::stack | TorchSharp.torch+distributions+constraints.stack | &#10003; |  |
| aten::stft | TorchSharp.torch+Tensor.stft | &#10003; |  |
| aten::sub.Scalar | TorchSharp.torch+Tensor.sub | &#10003; |  |
| aten::sub.Tensor | TorchSharp.torch+Tensor.sub | &#10003; |  |
| aten::subtract.Scalar | TorchSharp.torch+Tensor.subtract | &#10003; |  |
| aten::subtract.Tensor | TorchSharp.torch+Tensor.subtract | &#10003; |  |
| aten::sum | TorchSharp.torch+Tensor.sum | &#10003; |  |
| aten::sum.dim_IntList | TorchSharp.torch+Tensor.sum | &#10003; |  |
| aten::sym_size.int |  |  |  |
| aten::sym_storage_offset |  |  |  |
| aten::t | TorchSharp.torch+Tensor.t | &#10003; |  |
| aten::tan | TorchSharp.torch+Tensor.tan | &#10003; |  |
| aten::tanh | TorchSharp.Modules.Tanh | &#10003; | &#10003; |
| aten::tensor.bool | TorchSharp.torch+TensorIndex.Tensor | &#10003; |  |
| aten::tensor.float | TorchSharp.torch+TensorIndex.Tensor | &#10003; |  |
| aten::tensor.int | TorchSharp.torch+TensorIndex.Tensor | &#10003; |  |
| aten::tile | TorchSharp.torch+Tensor.tile | &#10003; |  |
| aten::topk | TorchSharp.torch+Tensor.topk | &#10003; |  |
| aten::transpose.int | TorchSharp.torch+Tensor.transpose | &#10003; |  |
| aten::tril | TorchSharp.torch+Tensor.tril | &#10003; |  |
| aten::triu | TorchSharp.torch+Tensor.triu | &#10003; |  |
| aten::true_divide.Scalar | TorchSharp.torch+Tensor.true_divide | &#10003; |  |
| aten::true_divide.Tensor | TorchSharp.torch+Tensor.true_divide | &#10003; |  |
| aten::trunc | TorchSharp.torch+Tensor.trunc | &#10003; |  |
| aten::type_as | TorchSharp.torch+Tensor.type_as | &#10003; |  |
| aten::unbind.int | TorchSharp.torch+Tensor.unbind | &#10003; |  |
| aten::unflatten.int | TorchSharp.Modules.Unflatten | &#10003; |  |
| aten::unfold | TorchSharp.Modules.Unfold | &#10003; |  |
| aten::unique_consecutive | TorchSharp.torch+Tensor.unique_consecutive | &#10003; |  |
| aten::unique_dim |  |  |  |
| aten::unsafe_split.Tensor |  |  |  |
| aten::unsqueeze | TorchSharp.torch+Tensor.unsqueeze | &#10003; |  |
| aten::upsample_bicubic2d |  |  |  |
| aten::upsample_bicubic2d.vec |  |  |  |
| aten::upsample_bilinear2d |  |  |  |
| aten::upsample_bilinear2d.vec |  |  |  |
| aten::upsample_linear1d |  |  |  |
| aten::upsample_nearest1d | TorchSharp.torch+nn+functional.upsample_nearest1d | &#10003; |  |
| aten::upsample_nearest1d.vec | TorchSharp.torch+nn+functional.upsample_nearest1d | &#10003; |  |
| aten::upsample_nearest2d | TorchSharp.torch+nn+functional.upsample_nearest2d | &#10003; |  |
| aten::upsample_nearest2d.vec | TorchSharp.torch+nn+functional.upsample_nearest2d | &#10003; |  |
| aten::upsample_nearest3d | TorchSharp.torch+nn+functional.upsample_nearest3d | &#10003; |  |
| aten::upsample_nearest3d.vec | TorchSharp.torch+nn+functional.upsample_nearest3d | &#10003; |  |
| aten::upsample_trilinear3d |  |  |  |
| aten::upsample_trilinear3d.vec |  |  |  |
| aten::view | TorchSharp.torch+Tensor.view | &#10003; |  |
| aten::view_as | TorchSharp.torch+Tensor.view_as | &#10003; |  |
| aten::view_as_complex | TorchSharp.torch+Tensor.view_as_complex | &#10003; |  |
| aten::view_as_complex_copy |  |  |  |
| aten::view_as_real | TorchSharp.torch+Tensor.view_as_real | &#10003; |  |
| aten::view_as_real_copy |  |  |  |
| aten::view_copy |  |  |  |
| aten::where.Scalar | TorchSharp.torch+Tensor.where | &#10003; |  |
| aten::where.ScalarOther | TorchSharp.torch+Tensor.where | &#10003; |  |
| aten::where.ScalarSelf | TorchSharp.torch+Tensor.where | &#10003; |  |
| aten::where.self | TorchSharp.torch+Tensor.where | &#10003; |  |
| aten::xlogy.Scalar_Other | TorchSharp.torch+Tensor.xlogy | &#10003; |  |
| aten::xlogy.Scalar_Self | TorchSharp.torch+Tensor.xlogy | &#10003; |  |
| aten::xlogy.Tensor | TorchSharp.torch+Tensor.xlogy | &#10003; |  |
| aten::zeros | TorchSharp.torch+Tensor.zeros | &#10003; |  |
| aten::zeros_like | TorchSharp.torch+Tensor.zeros_like | &#10003; |  |
| math::ceil | TorchSharp.torch+Tensor.ceil | &#10003; |  |
| math::floor | TorchSharp.torch+Tensor.floor | &#10003; |  |
| math::trunc | TorchSharp.torch+Tensor.trunc | &#10003; |  |
| prims::abs | TorchSharp.torch+Tensor.abs | &#10003; |  |
| prims::acos | TorchSharp.torch+Tensor.acos | &#10003; |  |
| prims::acosh | TorchSharp.torch+Tensor.acosh | &#10003; |  |
| prims::add | TorchSharp.torch+Tensor.add | &#10003; |  |
| prims::asin | TorchSharp.torch+Tensor.asin | &#10003; |  |
| prims::asinh | TorchSharp.torch+Tensor.asinh | &#10003; |  |
| prims::atan | TorchSharp.torch+Tensor.atan | &#10003; |  |
| prims::atanh | TorchSharp.torch+Tensor.atanh | &#10003; |  |
| prims::broadcast_in_dim |  |  |  |
| prims::ceil | TorchSharp.torch+Tensor.ceil | &#10003; |  |
| prims::convert_element_type |  |  |  |
| prims::cos | TorchSharp.torch+Tensor.cos | &#10003; |  |
| prims::cosh | TorchSharp.torch+Tensor.cosh | &#10003; |  |
| prims::device_put |  |  |  |
| prims::div | TorchSharp.torch+Tensor.div | &#10003; |  |
| prims::eq | TorchSharp.torch+Tensor.eq | &#10003; |  |
| prims::erf | TorchSharp.torch+Tensor.erf | &#10003; |  |
| prims::exp | TorchSharp.torch+Tensor.exp | &#10003; |  |
| prims::floor | TorchSharp.torch+Tensor.floor | &#10003; |  |
| prims::ge | TorchSharp.torch+Tensor.ge | &#10003; |  |
| prims::gt | TorchSharp.torch+Tensor.gt | &#10003; |  |
| prims::le | TorchSharp.torch+Tensor.le | &#10003; |  |
| prims::log | TorchSharp.torch+Tensor.log | &#10003; |  |
| prims::lt | TorchSharp.torch+Tensor.lt | &#10003; |  |
| prims::mul | TorchSharp.torch+Tensor.mul | &#10003; |  |
| prims::ne | TorchSharp.torch+Tensor.ne | &#10003; |  |
| prims::neg | TorchSharp.torch+Tensor.neg | &#10003; |  |
| prims::pow | TorchSharp.torch+Tensor.pow | &#10003; |  |
| prims::reshape | TorchSharp.torch+Tensor.reshape | &#10003; |  |
| prims::resize |  |  |  |
| prims::round | TorchSharp.torch+Tensor.round | &#10003; |  |
| prims::sin | TorchSharp.torch+Tensor.sin | &#10003; |  |
| prims::sinh | TorchSharp.torch+Tensor.sinh | &#10003; |  |
| prims::sqrt | TorchSharp.torch+Tensor.sqrt | &#10003; |  |
| prims::squeeze | TorchSharp.torch+Tensor.squeeze | &#10003; |  |
| prims::sub | TorchSharp.torch+Tensor.sub | &#10003; |  |
| prims::sum | TorchSharp.torch+Tensor.sum | &#10003; |  |
| prims::tan | TorchSharp.torch+Tensor.tan | &#10003; |  |
| prims::tanh | TorchSharp.Modules.Tanh | &#10003; |  |
| prims::transpose | TorchSharp.torch+Tensor.transpose | &#10003; |  |
| prims::var | TorchSharp.torch+Tensor.var | &#10003; |  |
| prims::where | TorchSharp.torch+Tensor.where | &#10003; |  |
| quantized_decomposed::dequantize_per_tensor |  |  |  |
| quantized_decomposed::dequantize_per_tensor.tensor |  |  |  |
| quantized_decomposed::dequantize_per_tensor.tensor2 |  |  |  |
| quantized_decomposed::quantize_per_tensor |  |  |  |
| quantized_decomposed::quantize_per_tensor.tensor |  |  |  |
| quantized_decomposed::quantize_per_tensor.tensor2 |  |  |  |
| torchvision::nms |  |  |  |
| torchvision::roi_align |  |  |  |
| torchvision::roi_pool |  |  |  |
