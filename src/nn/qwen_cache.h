#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include "tensors/dtype.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn {

struct QwenAttentionCache {
    std::optional<tensors::Tensor> key;
    std::optional<tensors::Tensor> value;
    std::size_t sequence_length{};
    std::size_t sequence_position_offset{};
};

struct QwenDecoderBlockCache {
    QwenAttentionCache attention;
};

// Allocates fixed-capacity K/V tensors for one attention layer, append grows them if capacity is exceeded.
// Shapes: key/value [max_sequence, kv_heads, head_dim], sequence_length starts at 0.
QwenAttentionCache make_qwen_attention_cache(std::string_view key_name, std::string_view value_name,
                                             tensors::DType dtype, std::size_t max_sequence_length,
                                             std::size_t num_key_value_heads, std::size_t head_dim);

// Appends current rotated keys and values to cache, allocating or growing storage when needed.
// Current shapes: key/value [new_sequence, kv_heads, head_dim], positions must be contiguous with the cache prefix.
void append_to_qwen_attention_cache(QwenAttentionCache& cache, const tensors::TensorView& key,
                                    const tensors::TensorView& value, std::size_t sequence_position_offset);

// Returns a view over the filled key prefix, not the unused cache capacity.
// Shape: [sequence_length, kv_heads, head_dim].
tensors::TensorView qwen_attention_cache_key_view(const QwenAttentionCache& cache);

// Returns a view over the filled value prefix, not the unused cache capacity.
// Shape: [sequence_length, kv_heads, head_dim].
tensors::TensorView qwen_attention_cache_value_view(const QwenAttentionCache& cache);

} // namespace cppinf::nn
