#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

// Multiplies two rank-2 tensors of shape [m, k] and [k, n] and returns [m, n].
// Requires matching f32 or bf16 dtypes. BF16 inputs are computed in f32 and cast back to bf16 for the result.
tensors::Tensor matmul(const tensors::TensorView& lhs, const tensors::TensorView& rhs);

} // namespace cppinf::ops
