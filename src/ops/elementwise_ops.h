#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

// Adds two tensors with matching shape and f32 or bf16 dtype.
// BF16 inputs are computed in f32 and cast back to bf16 for the result.
tensors::Tensor add(const tensors::TensorView& lhs, const tensors::TensorView& rhs);

// Multiplies two tensors with matching shape and f32 or bf16 dtype.
// BF16 inputs are computed in f32 and cast back to bf16 for the result.
tensors::Tensor mul(const tensors::TensorView& lhs, const tensors::TensorView& rhs);

} // namespace cppinf::ops
