#include "nn/qwen_attention.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <fmt/format.h>

#include "nn/qwen_cache.h"
#include "nn/rope.h"
#include "ops/matmul.h"
#include "ops/nn_ops.h"
#include "ops/one_dnn_utils.h"
#include "ops/op_utils.h"
#include "ops/tensor_ops.h"
#include "tensors/dtype.h"
#include "tensors/shape.h"
#include "tensors/tensor_info.h"
#include "tensors/tensor_utils.h"

namespace cppinf::nn {
namespace {

std::size_t checked_positive_dim_to_size(std::int64_t dim, std::string_view field_name) {
    if (dim < 0) {
        throw std::invalid_argument(fmt::format("{} must be non-negative.", field_name));
    }

    const auto value = static_cast<std::size_t>(dim);
    if (value == 0) {
        throw std::invalid_argument(fmt::format("{} must be non-zero.", field_name));
    }

    return value;
}

tensors::Tensor linear_project(const tensors::TensorView& input, const tensors::TensorView& weight) {
    // A transformer linear layer stores weights as [out_features, in_features], so transpose to multiply tokens [seq,
    // in_features] by [in_features, out_features].
    const auto transposed_weight = ops::transpose_2d(weight);
    return ops::matmul(input, transposed_weight.view());
}

tensors::TensorView reshape_heads(const tensors::TensorView& input, std::size_t head_count, std::size_t head_dim,
                                  std::string_view result_name) {
    if (input.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("reshape_heads requires a rank-2 tensor.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto sequence_length = checked_positive_dim_to_size(dims[0], fmt::format("{} sequence length", result_name));
    const auto merged_head_size =
        checked_positive_dim_to_size(dims[1], fmt::format("{} merged head size", result_name));
    if (merged_head_size != head_count * head_dim) {
        throw std::invalid_argument(
            fmt::format("{} requires merged head size to equal head_count * head_dim.", result_name));
    }

    return ops::reshape(input,
                        tensors::Shape({static_cast<std::int64_t>(sequence_length),
                                        static_cast<std::int64_t>(head_count), static_cast<std::int64_t>(head_dim)}));
}

void validate_projection_weight(const tensors::TensorView& weight, std::string_view name, std::size_t output_size,
                                std::size_t input_size) {
    if (weight.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument(fmt::format("{} must be rank-2.", name));
    }

    const auto& dims = weight.tensor_info().shape.dims();
    if (checked_positive_dim_to_size(dims[0], fmt::format("{} rows", name)) != output_size ||
        checked_positive_dim_to_size(dims[1], fmt::format("{} cols", name)) != input_size) {
        throw std::invalid_argument(fmt::format("{} has an unexpected shape.", name));
    }
}

void validate_norm_weight(const tensors::TensorView& weight, std::string_view name, std::size_t head_dim) {
    if (weight.tensor_info().shape.rank() != 1) {
        throw std::invalid_argument(fmt::format("{} must be rank-1.", name));
    }

    if (checked_positive_dim_to_size(weight.tensor_info().shape.dims()[0], fmt::format("{} size", name)) != head_dim) {
        throw std::invalid_argument(fmt::format("{} must have size head_dim.", name));
    }
}

void validate_qwen_attention_inputs(const tensors::TensorView& hidden_states, const QwenAttentionWeights& weights,
                                    std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                    std::size_t head_dim, float norm_epsilon, float rope_base) {
    if (num_attention_heads == 0 || num_key_value_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("qwen_attention requires non-zero head counts and head_dim.");
    }

    if (num_attention_heads % num_key_value_heads != 0) {
        throw std::invalid_argument("qwen_attention requires num_attention_heads to divide by num_key_value_heads.");
    }

    if (!std::isfinite(norm_epsilon) || norm_epsilon < 0.0f) {
        throw std::invalid_argument("qwen_attention requires a non-negative finite norm epsilon.");
    }

    if (!std::isfinite(rope_base) || rope_base <= 0.0f) {
        throw std::invalid_argument("qwen_attention requires a positive finite rope base.");
    }

    ops::detail::validate_supported_float_dtype(hidden_states.tensor_info().dtype, "qwen_attention");
    const auto dtype = hidden_states.tensor_info().dtype;
    if (dtype != weights.q_proj_weight.tensor_info().dtype || dtype != weights.q_norm_weight.tensor_info().dtype ||
        dtype != weights.k_proj_weight.tensor_info().dtype || dtype != weights.k_norm_weight.tensor_info().dtype ||
        dtype != weights.v_proj_weight.tensor_info().dtype || dtype != weights.o_proj_weight.tensor_info().dtype) {
        throw std::invalid_argument("qwen_attention requires matching tensor dtypes.");
    }

    if (hidden_states.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("qwen_attention requires rank-2 hidden states.");
    }

    const auto& hidden_dims = hidden_states.tensor_info().shape.dims();
    checked_positive_dim_to_size(hidden_dims[0], "qwen_attention sequence length");
    const auto hidden_size = checked_positive_dim_to_size(hidden_dims[1], "qwen_attention hidden size");

    validate_projection_weight(weights.q_proj_weight, "qwen_attention q_proj_weight", num_attention_heads * head_dim,
                               hidden_size);
    validate_norm_weight(weights.q_norm_weight, "qwen_attention q_norm_weight", head_dim);
    validate_projection_weight(weights.k_proj_weight, "qwen_attention k_proj_weight", num_key_value_heads * head_dim,
                               hidden_size);
    validate_norm_weight(weights.k_norm_weight, "qwen_attention k_norm_weight", head_dim);
    validate_projection_weight(weights.v_proj_weight, "qwen_attention v_proj_weight", num_key_value_heads * head_dim,
                               hidden_size);
    validate_projection_weight(weights.o_proj_weight, "qwen_attention o_proj_weight", hidden_size,
                               num_attention_heads * head_dim);
}

tensors::Tensor scaled_position_causal_softmax_last_dim(const tensors::TensorView& input, float scale,
                                                        std::size_t query_position_offset,
                                                        std::size_t key_position_offset) {
    if (input.tensor_info().shape.rank() < 2) {
        throw std::invalid_argument("qwen_attention cached softmax requires a tensor with rank at least 2.");
    }

    if (!std::isfinite(scale)) {
        throw std::invalid_argument("qwen_attention cached softmax requires a finite scale.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto query_length = checked_positive_dim_to_size(dims[dims.size() - 2], "qwen_attention query length");
    const auto key_length = checked_positive_dim_to_size(dims[dims.size() - 1], "qwen_attention key length");

    std::optional<tensors::Tensor> input_storage;
    const auto input_f32 =
        ops::detail::maybe_cast_to_dtype(input, tensors::DType::F32, input_storage, "qwen_attention_probabilities");

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        "qwen_attention_probabilities", tensors::DType::F32, input.tensor_info().shape));
    const auto matrix_size = query_length * key_length;
    const auto matrix_count = input.tensor_info().shape.num_elements() / matrix_size;

    // Treat every leading dimension as an independent attention matrix. In Qwen attention the shape is usually
    // [q_heads, query_seq, key_seq], and cached decoding makes query_seq smaller than key_seq after prefill.
    for (std::size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        const auto matrix_offset = matrix_index * matrix_size;
        for (std::size_t query_index = 0; query_index < query_length; ++query_index) {
            const auto row_offset = matrix_offset + query_index * key_length;
            const auto query_position = query_position_offset + query_index;
            auto max_value = -std::numeric_limits<float>::infinity();

            // Cached model keys normally start at position 0, while direct attention calls can pass a non-zero
            // sequence offset for oracle parity.
            for (std::size_t key_index = 0; key_index < key_length; ++key_index) {
                const auto key_position = key_position_offset + key_index;
                if (key_position <= query_position) {
                    const auto value = ops::detail::load_float_value(input_f32, row_offset + key_index) * scale;
                    max_value = std::max(max_value, value);
                }
            }

            // Store exp(score - max) for visible keys and zero for masked future keys, then normalize this row below.
            auto sum = 0.0f;
            for (std::size_t key_index = 0; key_index < key_length; ++key_index) {
                const auto flat_index = row_offset + key_index;
                const auto key_position = key_position_offset + key_index;
                const auto value =
                    key_position <= query_position
                        ? std::exp(ops::detail::load_float_value(input_f32, flat_index) * scale - max_value)
                        : 0.0f;
                ops::detail::store_float_value(tensors::DType::F32, result.mutable_data(), flat_index, value);
                sum += value;
            }

            // Finish softmax by dividing each visible exponential by the row sum, masked positions stay exactly zero.
            for (std::size_t key_index = 0; key_index < key_length; ++key_index) {
                const auto flat_index = row_offset + key_index;
                const auto value = ops::detail::load_float_value(tensors::DType::F32, result.data(), flat_index) / sum;
                ops::detail::store_float_value(tensors::DType::F32, result.mutable_data(), flat_index, value);
            }
        }
    }

    return result;
}

tensors::Tensor fused_causal_attention(const tensors::TensorView& query, const tensors::TensorView& key,
                                       const tensors::TensorView& value, std::size_t query_position_offset,
                                       std::size_t key_position_offset) {
    if (query.tensor_info().shape.rank() != 3 || key.tensor_info().shape.rank() != 3 ||
        value.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("qwen_attention requires rank-3 query, key, and value tensors.");
    }

    const auto& query_dims = query.tensor_info().shape.dims();
    const auto& key_dims = key.tensor_info().shape.dims();
    const auto& value_dims = value.tensor_info().shape.dims();
    checked_positive_dim_to_size(query_dims[0], "qwen_attention query sequence length");
    const auto query_head_count = checked_positive_dim_to_size(query_dims[1], "qwen_attention query head count");
    const auto key_sequence_length = checked_positive_dim_to_size(key_dims[0], "qwen_attention key sequence length");
    const auto key_head_count = checked_positive_dim_to_size(key_dims[1], "qwen_attention key head count");
    const auto value_sequence_length =
        checked_positive_dim_to_size(value_dims[0], "qwen_attention value sequence length");
    const auto value_head_count = checked_positive_dim_to_size(value_dims[1], "qwen_attention value head count");
    const auto query_head_size = checked_positive_dim_to_size(query_dims[2], "qwen_attention query head size");
    const auto key_head_size = checked_positive_dim_to_size(key_dims[2], "qwen_attention key head size");
    const auto value_head_size = checked_positive_dim_to_size(value_dims[2], "qwen_attention value head size");
    if (key_head_count != value_head_count) {
        throw std::invalid_argument("qwen_attention requires matching key/value head counts.");
    }

    if (query_head_count % key_head_count != 0) {
        throw std::invalid_argument("qwen_attention requires query heads to divide by key/value heads.");
    }

    if (key_sequence_length != value_sequence_length) {
        throw std::invalid_argument("qwen_attention requires matching key/value sequence lengths.");
    }

    if (query_head_size != key_head_size || query_head_size != value_head_size) {
        throw std::invalid_argument("qwen_attention requires matching query/key/value head sizes.");
    }

    const auto score_scale = 1.0f / std::sqrt(static_cast<float>(query_head_size));
    const auto queries_per_key_value_head = query_head_count / key_head_count;

    // Move from token-major Q/K/V to head-major mini-batches: Q [q_heads, seq, head_dim], K [kv_heads, head_dim, seq],
    // V [kv_heads, seq, head_dim].
    const std::array<std::size_t, 3> query_axes = {1, 0, 2};
    const std::array<std::size_t, 3> key_axes = {1, 2, 0};
    const std::array<std::size_t, 3> value_axes = {1, 0, 2};
    const std::array<std::size_t, 3> output_axes = {1, 0, 2};
    const auto query_batch = ops::permute(query, query_axes);
    const auto key_by_kv_head = ops::permute(key, key_axes);
    const auto value_by_kv_head = ops::permute(value, value_axes);

    // Qwen can use grouped-query attention, many query heads share fewer key/value heads, so repeat K/V batches until
    // every query head has a matching K/V batch.
    const auto key_batch = ops::repeat_interleave(key_by_kv_head.view(), 0, queries_per_key_value_head);
    const auto value_batch = ops::repeat_interleave(value_by_kv_head.view(), 0, queries_per_key_value_head);

    // Each score is the dot product between one query token and one cached key token inside a head, producing
    // [q_heads, query_seq, key_seq] before the causal mask.
    const auto attention_scores =
        ops::matmul(query_batch.view(), key_batch.view(), ops::MatmulOptions{.output_dtype = tensors::DType::F32});

    // Scale by 1/sqrt(head_dim), mask future tokens, and softmax across source positions so each row becomes weights
    // over the visible context.
    auto probabilities = scaled_position_causal_softmax_last_dim(attention_scores.view(), score_scale,
                                                                 query_position_offset, key_position_offset);

    // Hugging Face Qwen casts BF16 attention probabilities back down before the value matmul, so mirror that dtype
    // boundary for golden parity.
    probabilities = ops::detail::maybe_cast_result(std::move(probabilities), query.tensor_info().dtype,
                                                   "qwen_attention_probabilities");

    // Weight the value vectors by the attention probabilities to produce one context vector per token and query head,
    // [q_heads, seq, head_dim].
    auto batch_result = ops::matmul(probabilities.view(), value_batch.view());
    batch_result = ops::detail::maybe_cast_result(std::move(batch_result), query.tensor_info().dtype,
                                                  "qwen_attention_attention_output_batch");

    // Return to token-major layout [seq, q_heads, head_dim] so the caller can merge heads back into the hidden
    // dimension.
    return ops::permute(batch_result.view(), output_axes);
}

tensors::Tensor qwen_attention_at_position(const tensors::TensorView& hidden_states,
                                           const QwenAttentionWeights& weights, QwenAttentionCache& cache,
                                           std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                           std::size_t head_dim, float norm_epsilon,
                                           std::size_t sequence_position_offset, float rope_base) {
    validate_qwen_attention_inputs(hidden_states, weights, num_attention_heads, num_key_value_heads, head_dim,
                                   norm_epsilon, rope_base);

    // Project each token embedding [seq, hidden] into the three streams attention needs: queries ask what this token is
    // looking for, keys describe what each token offers, and values carry the information to mix.
    const auto query_projection = linear_project(hidden_states, weights.q_proj_weight);
    const auto key_projection = linear_project(hidden_states, weights.k_proj_weight);
    const auto value_projection = linear_project(hidden_states, weights.v_proj_weight);

    // Reinterpret the packed per-token layout [seq, heads * head_dim] as [seq, heads, head_dim] without copying.
    const auto query_heads =
        reshape_heads(query_projection.view(), num_attention_heads, head_dim, "qwen_attention_query_heads");
    const auto key_heads =
        reshape_heads(key_projection.view(), num_key_value_heads, head_dim, "qwen_attention_key_heads");
    const auto value_heads =
        reshape_heads(value_projection.view(), num_key_value_heads, head_dim, "qwen_attention_value_heads");

    // Normalize each query/key head independently, then apply RoPE, which encodes absolute token position by rotating
    // feature pairs before similarity scores are computed.
    const auto normalized_query = ops::rms_norm(query_heads, weights.q_norm_weight, norm_epsilon);
    const auto normalized_key = ops::rms_norm(key_heads, weights.k_norm_weight, norm_epsilon);
    const auto rotated_query = apply_rope(normalized_query.view(), sequence_position_offset, rope_base, 0);
    const auto rotated_key = apply_rope(normalized_key.view(), sequence_position_offset, rope_base, 0);
    if (sequence_position_offset < cache.sequence_length) {
        throw std::invalid_argument("qwen_attention sequence position cannot be shorter than the cached prefix.");
    }
    const auto key_position_offset = sequence_position_offset - cache.sequence_length;

    // Append only this call's new keys/values. The cache view used below expands to [cached_seq + new_seq, kv_heads,
    // head_dim], while rotated_query remains [new_seq, q_heads, head_dim].
    append_to_qwen_attention_cache(cache, rotated_key.view(), value_heads);

    // Build token context with causal attention, then flatten [seq, q_heads, head_dim] back to [seq, q_heads *
    // head_dim] for the output projection.
    const auto cached_key = qwen_attention_cache_key_view(cache);
    const auto cached_value = qwen_attention_cache_value_view(cache);
    const auto attention_output = fused_causal_attention(rotated_query.view(), cached_key, cached_value,
                                                         sequence_position_offset, key_position_offset);
    const auto packed_attention =
        ops::reshape(attention_output.view(), tensors::Shape({query_projection.tensor_info().shape.dims()[0],
                                                              query_projection.tensor_info().shape.dims()[1]}));

    // The output projection mixes information across heads and returns to the model hidden size expected by the
    // residual stream.
    return tensors::rename_tensor("qwen_attention_result", linear_project(packed_attention, weights.o_proj_weight));
}

} // namespace

tensors::Tensor qwen_attention(const tensors::TensorView& hidden_states, const QwenAttentionWeights& weights,
                               std::size_t num_attention_heads, std::size_t num_key_value_heads, std::size_t head_dim,
                               float norm_epsilon, float rope_base) {
    auto cache = make_qwen_attention_cache("qwen_attention_key", "qwen_attention_value",
                                           hidden_states.tensor_info().dtype, 1, num_key_value_heads, head_dim);
    return qwen_attention_at_position(hidden_states, weights, cache, num_attention_heads, num_key_value_heads, head_dim,
                                      norm_epsilon, 0, rope_base);
}

tensors::Tensor qwen_attention_with_cache(const tensors::TensorView& hidden_states, const QwenAttentionWeights& weights,
                                          QwenAttentionCache& cache, std::size_t num_attention_heads,
                                          std::size_t num_key_value_heads, std::size_t head_dim, float norm_epsilon,
                                          float rope_base) {
    return qwen_attention_at_position(hidden_states, weights, cache, num_attention_heads, num_key_value_heads, head_dim,
                                      norm_epsilon, cache.sequence_length, rope_base);
}

} // namespace cppinf::nn
