#pragma once

#include <cstddef>

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn {

// Applies split-half RoPE to a rank-3 [heads, sequence, dim] tensor.
// Head dim must be even, positions start at sequence_position_offset, and the result returns in the input dtype.
tensors::Tensor apply_rope(const tensors::TensorView& input, std::size_t sequence_position_offset = 0,
                           float rope_base = 1000000.0f);

} // namespace cppinf::nn
