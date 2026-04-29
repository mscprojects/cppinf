#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

// Applies SiLU elementwise and preserves the input shape and dtype.
tensors::Tensor silu(const tensors::TensorView& input);

// Applies softmax over the last dimension of a rank >= 1 tensor and preserves the input shape and dtype.
tensors::Tensor softmax_last_dim(const tensors::TensorView& input);

// Applies RMSNorm over the last dimension using a rank-1 weight tensor of matching width.
// BF16 inputs are computed in f32 and cast back to bf16 for the result.
tensors::Tensor rms_norm(const tensors::TensorView& input, const tensors::TensorView& weight, float epsilon);

} // namespace cppinf::ops
