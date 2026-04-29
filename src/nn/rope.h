#pragma once

#include <cstddef>

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn {

// Applies split-half RoPE to a rank-3 tensor with head dim in the last axis and the sequence axis at dim 0 or 1.
// Positions start at sequence_position_offset, and the result returns in the input dtype.
tensors::Tensor apply_rope(const tensors::TensorView& input, std::size_t sequence_position_offset = 0,
                           float rope_base = 1000000.0f, std::size_t sequence_dimension = 1);

} // namespace cppinf::nn
