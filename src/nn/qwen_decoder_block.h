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
// Uses pre-norm residual paths, explicit head_dim, no cache, BF16 inputs compute in f32 and cast back at the end.
tensors::Tensor qwen_decoder_block(const tensors::TensorView& hidden_states, const QwenDecoderBlockWeights& weights,
                                   std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                   std::size_t head_dim, float norm_epsilon, std::size_t sequence_position_offset = 0,
                                   float rope_base = 1000000.0f);

} // namespace cppinf::nn
