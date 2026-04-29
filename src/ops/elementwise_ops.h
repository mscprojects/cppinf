#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

// Adds two tensors with matching shape and f32 or bf16 dtype.
// Preserves the input dtype.
tensors::Tensor add(const tensors::TensorView& lhs, const tensors::TensorView& rhs);

// Multiplies two tensors with matching shape and f32 or bf16 dtype.
// Preserves the input dtype.
tensors::Tensor mul(const tensors::TensorView& lhs, const tensors::TensorView& rhs);

} // namespace cppinf::ops
