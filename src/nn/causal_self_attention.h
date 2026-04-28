#pragma once

#include <cstddef>

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn {

tensors::Tensor causal_self_attention(const tensors::TensorView& query, const tensors::TensorView& key,
                                      const tensors::TensorView& value, std::size_t past_sequence_length = 0);

} // namespace cppinf::nn
