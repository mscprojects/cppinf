#pragma once

#include <cstddef>

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn {

// Applies causal self-attention to rank-3 [heads, sequence, dim] query, key, and value tensors.
// Key/value sequence length must equal past_sequence_length + query sequence length, and the output keeps the input
// dtype.
tensors::Tensor causal_self_attention(const tensors::TensorView& query, const tensors::TensorView& key,
                                      const tensors::TensorView& value, std::size_t past_sequence_length = 0);

} // namespace cppinf::nn
