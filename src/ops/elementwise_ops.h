#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

tensors::Tensor add(const tensors::TensorView& lhs, const tensors::TensorView& rhs);
tensors::Tensor mul(const tensors::TensorView& lhs, const tensors::TensorView& rhs);

} // namespace cppinf::ops
