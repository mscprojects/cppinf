#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

tensors::Tensor matmul(const tensors::TensorView& lhs, const tensors::TensorView& rhs);

} // namespace cppinf::ops
