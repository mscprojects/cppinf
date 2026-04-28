#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

tensors::Tensor silu(const tensors::TensorView& input);
tensors::Tensor softmax_last_dim(const tensors::TensorView& input);
tensors::Tensor rms_norm(const tensors::TensorView& input, const tensors::TensorView& weight, float epsilon);

} // namespace cppinf::ops
