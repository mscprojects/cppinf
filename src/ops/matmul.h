#pragma once

#include "tensors/dtype.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

// Multiplies rank-2 [m, k] and [k, n] tensors or rank-3 [b, m, k] and [b, k, n] tensors.
// Requires matching f32 or bf16 dtypes. BF16 inputs are computed in f32 and cast back to bf16 for the result.
tensors::Tensor matmul(const tensors::TensorView& lhs, const tensors::TensorView& rhs);

// Multiplies rank-2 [m, k] and [k, n] tensors or rank-3 [b, m, k] and [b, k, n] tensors.
// Requires matching f32 or bf16 input dtypes. Supports same-dtype output, plus BF16 inputs may request an F32 result.
tensors::Tensor matmul(const tensors::TensorView& lhs, const tensors::TensorView& rhs, tensors::DType output_dtype);

} // namespace cppinf::ops
