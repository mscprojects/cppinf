#pragma once

#include <cstddef>
#include <span>

#include "nn/qwen_cache.h"
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

// Applies Qwen attention to rank-3 [batch, sequence, hidden] hidden states.
// sequence_lengths mark the valid prefix length for each batch row, and padded rows are ignored by the attention mask.
tensors::Tensor qwen_attention(const tensors::TensorView& hidden_states, std::span<const std::size_t> sequence_lengths,
                               const QwenAttentionWeights& weights, std::size_t num_attention_heads,
                               std::size_t num_key_value_heads, std::size_t head_dim, float norm_epsilon,
                               float rope_base = 1000000.0f);

// Applies cached Qwen attention to rank-3 [batch, sequence, hidden] hidden states.
// sequence_lengths mark the valid uncached suffix length for each batch row, and cache carries the already materialized
// prefix length for every batch row.
tensors::Tensor qwen_attention_with_cache(const tensors::TensorView& hidden_states,
                                          std::span<const std::size_t> sequence_lengths,
                                          const QwenAttentionWeights& weights, QwenAttentionCache& cache,
                                          std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                          std::size_t head_dim, float norm_epsilon, float rope_base = 1000000.0f);

} // namespace cppinf::nn
