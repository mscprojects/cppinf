#pragma once

#include <cstddef>

#include "nn/qwen_attention.h"
#include "nn/qwen_mlp.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn {

struct QwenDecoderBlockWeights {
    tensors::TensorView input_layernorm_weight;
    tensors::TensorView post_attention_layernorm_weight;
    QwenAttentionWeights attention;
    QwenMlpWeights mlp;
};

// Applies one Qwen decoder block to rank-2 [sequence, hidden] hidden states.
// Uses pre-norm residual paths, explicit head_dim, a temporary empty cache, and keeps the public tensor dtype.
tensors::Tensor qwen_decoder_block(const tensors::TensorView& hidden_states, const QwenDecoderBlockWeights& weights,
                                   std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                   std::size_t head_dim, float norm_epsilon, float rope_base = 1000000.0f);

// Applies one Qwen decoder block while appending this call's K/V tensors to cache for incremental decoding.
// The input and output are rank-2 [sequence, hidden], and cache stores the attention prefix for this layer.
tensors::Tensor qwen_decoder_block_with_cache(const tensors::TensorView& hidden_states,
                                              const QwenDecoderBlockWeights& weights, QwenDecoderBlockCache& cache,
                                              std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                              std::size_t head_dim, float norm_epsilon, float rope_base = 1000000.0f);

} // namespace cppinf::nn
