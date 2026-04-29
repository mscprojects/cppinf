#pragma once

#include <cstddef>

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn {

struct QwenAttentionWeights {
    tensors::TensorView q_proj_weight;
    tensors::TensorView q_norm_weight;
    tensors::TensorView k_proj_weight;
    tensors::TensorView k_norm_weight;
    tensors::TensorView v_proj_weight;
    tensors::TensorView o_proj_weight;
};

// Applies bias-free Qwen attention to rank-2 [sequence, hidden] hidden states with projection and q/k norm weights.
// Fuses the Qwen-specific projection, q/k norm, RoPE, grouped KV handling, causal attention, and output projection
// into one high-level op while preserving the caller-visible dtype.
tensors::Tensor qwen_attention(const tensors::TensorView& hidden_states, const QwenAttentionWeights& weights,
                               std::size_t num_attention_heads, std::size_t num_key_value_heads, std::size_t head_dim,
                               float norm_epsilon, std::size_t sequence_position_offset = 0,
                               float rope_base = 1000000.0f);

} // namespace cppinf::nn
