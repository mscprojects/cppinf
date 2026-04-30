#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

// Applies SiLU elementwise and preserves the input shape and dtype.
tensors::Tensor silu(const tensors::TensorView& input);

// Applies softmax over the last dimension and preserves the dtype.
// Shapes: [d0, ..., dn] -> [d0, ..., dn].
tensors::Tensor softmax_last_dim(const tensors::TensorView& input);

// Applies f32 scale, causal masking over the last two dimensions, and softmax over the last dimension.
// Shapes: [batch..., query, key] -> [batch..., query, key].
tensors::Tensor scaled_causal_softmax_last_dim(const tensors::TensorView& input, float scale);

// Applies RMSNorm over the last dimension using a rank-1 weight tensor of matching width.
// Shapes: [d0, ..., width], [width] -> [d0, ..., width].
// BF16 inputs are computed in f32 and cast back to bf16 for the result.
tensors::Tensor rms_norm(const tensors::TensorView& input, const tensors::TensorView& weight, float epsilon);

} // namespace cppinf::ops
