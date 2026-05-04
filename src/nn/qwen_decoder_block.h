#pragma once

#include <cstddef>
#include <span>

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

// Applies one Qwen decoder block to rank-3 [batch, sequence, hidden] hidden states while appending K/V tensors to
// cache. sequence_lengths describe the valid suffix in this call for each batch row.
tensors::Tensor qwen_decoder_block_with_cache(const tensors::TensorView& hidden_states,
                                              std::span<const std::size_t> sequence_lengths,
                                              const QwenDecoderBlockWeights& weights, QwenDecoderBlockCache& cache,
                                              std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                              std::size_t head_dim, float norm_epsilon, float rope_base = 1000000.0f);

} // namespace cppinf::nn
