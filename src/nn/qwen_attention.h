#pragma once

#include <cstddef>

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn {

// Applies bias-free Qwen attention to rank-2 [sequence, hidden] hidden states with projection weights.
// Supports grouped KV heads, positions start at sequence_position_offset, BF16 inputs compute in f32.
tensors::Tensor qwen_attention(const tensors::TensorView& hidden_states, const tensors::TensorView& q_proj_weight,
                               const tensors::TensorView& k_proj_weight, const tensors::TensorView& v_proj_weight,
                               const tensors::TensorView& o_proj_weight, std::size_t num_attention_heads,
                               std::size_t num_key_value_heads, std::size_t sequence_position_offset = 0,
                               float rope_base = 1000000.0f);

} // namespace cppinf::nn
