#include "nn/qwen_attention.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

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

std::size_t checked_non_negative_dim_to_size(std::int64_t dim, std::string_view field_name) {
    if (dim < 0) {
        throw std::invalid_argument(fmt::format("{} must be non-negative.", field_name));
    }

    return static_cast<std::size_t>(dim);
}

std::size_t checked_positive_dim_to_size(std::int64_t dim, std::string_view field_name) {
    const auto value = checked_non_negative_dim_to_size(dim, field_name);
    if (value == 0) {
        throw std::invalid_argument(fmt::format("{} must be non-zero.", field_name));
    }

    return value;
}

std::int64_t checked_size_to_dim(std::size_t value, std::string_view field_name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(fmt::format("{} does not fit in int64_t.", field_name));
    }

    return static_cast<std::int64_t>(value);
}

void validate_sequence_lengths(std::span<const std::size_t> sequence_lengths, std::size_t batch_size,
                               std::size_t sequence_capacity, std::string_view op_name) {
    if (sequence_lengths.size() != batch_size) {
        throw std::invalid_argument(fmt::format("{} sequence_lengths must match the batch size.", op_name));
    }

    for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        if (sequence_lengths[batch_index] > sequence_capacity) {
            throw std::invalid_argument(
                fmt::format("{} sequence length exceeds the padded sequence capacity.", op_name));
        }
    }
}

tensors::Tensor linear_project(const tensors::TensorView& input, const tensors::TensorView& weight,
                               std::string_view result_name) {
    // A transformer linear layer stores weights as [out_features, in_features], so transpose to multiply tokens
    // [..., in_features] by [in_features, out_features].
    const auto transposed_weight = ops::transpose_2d(weight);
    if (input.tensor_info().shape.rank() == 2) {
        return tensors::rename_tensor(result_name, ops::matmul(input, transposed_weight.view()));
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto batch_size = checked_positive_dim_to_size(dims[0], fmt::format("{} batch size", result_name));
    const auto sequence_length = checked_positive_dim_to_size(dims[1], fmt::format("{} sequence length", result_name));
    const auto hidden_size = checked_positive_dim_to_size(dims[2], fmt::format("{} hidden size", result_name));
    const auto flat_input = ops::reshape(input, tensors::Shape({static_cast<std::int64_t>(batch_size * sequence_length),
                                                                static_cast<std::int64_t>(hidden_size)}));
    auto projected = ops::matmul(flat_input, transposed_weight.view());
    return tensors::materialize_tensor(
        result_name, ops::reshape(projected.view(), tensors::Shape({static_cast<std::int64_t>(batch_size),
                                                                    static_cast<std::int64_t>(sequence_length),
                                                                    projected.tensor_info().shape.dims()[1]})));
}

tensors::TensorView reshape_heads(const tensors::TensorView& input, std::size_t head_count, std::size_t head_dim,
                                  std::string_view result_name) {
    if (input.tensor_info().shape.rank() != 2 && input.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("reshape_heads requires a rank-2 or rank-3 tensor.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto merged_head_size =
        checked_positive_dim_to_size(dims.back(), fmt::format("{} merged head size", result_name));
    if (merged_head_size != head_count * head_dim) {
        throw std::invalid_argument(
            fmt::format("{} requires merged head size to equal head_count * head_dim.", result_name));
    }

    if (input.tensor_info().shape.rank() == 2) {
        return ops::reshape(input, tensors::Shape({dims[0], static_cast<std::int64_t>(head_count),
                                                   static_cast<std::int64_t>(head_dim)}));
    }

    return ops::reshape(input, tensors::Shape({dims[0], dims[1], static_cast<std::int64_t>(head_count),
                                               static_cast<std::int64_t>(head_dim)}));
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

void validate_qwen_attention_inputs(const tensors::TensorView& hidden_states,
                                    std::span<const std::size_t> sequence_lengths, const QwenAttentionWeights& weights,
                                    std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                    std::size_t head_dim, float norm_epsilon, float rope_base,
                                    std::string_view op_name) {
    if (num_attention_heads == 0 || num_key_value_heads == 0 || head_dim == 0) {
        throw std::invalid_argument(fmt::format("{} requires non-zero head counts and head_dim.", op_name));
    }

    if (num_attention_heads % num_key_value_heads != 0) {
        throw std::invalid_argument(
            fmt::format("{} requires num_attention_heads to divide by num_key_value_heads.", op_name));
    }

    if (!std::isfinite(norm_epsilon) || norm_epsilon < 0.0f) {
        throw std::invalid_argument(fmt::format("{} requires a non-negative finite norm epsilon.", op_name));
    }

    if (!std::isfinite(rope_base) || rope_base <= 0.0f) {
        throw std::invalid_argument(fmt::format("{} requires a positive finite rope base.", op_name));
    }

    ops::detail::validate_supported_float_dtype(hidden_states.tensor_info().dtype, op_name);
    const auto dtype = hidden_states.tensor_info().dtype;
    if (dtype != weights.q_proj_weight.tensor_info().dtype || dtype != weights.q_norm_weight.tensor_info().dtype ||
        dtype != weights.k_proj_weight.tensor_info().dtype || dtype != weights.k_norm_weight.tensor_info().dtype ||
        dtype != weights.v_proj_weight.tensor_info().dtype || dtype != weights.o_proj_weight.tensor_info().dtype) {
        throw std::invalid_argument(fmt::format("{} requires matching tensor dtypes.", op_name));
    }

    if (hidden_states.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument(fmt::format("{} requires rank-3 hidden states.", op_name));
    }

    const auto& hidden_dims = hidden_states.tensor_info().shape.dims();
    const auto batch_size = checked_positive_dim_to_size(hidden_dims[0], fmt::format("{} batch size", op_name));
    const auto sequence_capacity =
        checked_positive_dim_to_size(hidden_dims[1], fmt::format("{} sequence length", op_name));
    validate_sequence_lengths(sequence_lengths, batch_size, sequence_capacity, op_name);
    const auto hidden_size = checked_positive_dim_to_size(hidden_dims[2], fmt::format("{} hidden size", op_name));

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

tensors::Tensor apply_rope_batched(const tensors::TensorView& input,
                                   std::span<const std::size_t> sequence_position_offsets, float rope_base,
                                   std::string_view result_name) {
    if (input.tensor_info().shape.rank() != 4) {
        throw std::invalid_argument(fmt::format("{} requires a rank-4 tensor.", result_name));
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto batch_size = checked_positive_dim_to_size(dims[0], fmt::format("{} batch size", result_name));
    validate_sequence_lengths(sequence_position_offsets, batch_size, std::numeric_limits<std::size_t>::max(),
                              result_name);

    auto result = tensors::Tensor::zeros(
        tensors::make_result_tensor_info(result_name, input.tensor_info().dtype, input.tensor_info().shape));
    const auto batch_byte_size = checked_positive_dim_to_size(dims[1], fmt::format("{} sequence length", result_name)) *
                                 checked_positive_dim_to_size(dims[2], fmt::format("{} head count", result_name)) *
                                 checked_positive_dim_to_size(dims[3], fmt::format("{} head dim", result_name)) *
                                 tensors::element_size_bytes(input.tensor_info().dtype);
    for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        const auto batch_input = ops::squeeze(ops::narrow(input, 0, batch_index, 1), 0);
        const auto rotated_batch = apply_rope(batch_input, sequence_position_offsets[batch_index], rope_base, 0);
        std::memcpy(result.mutable_data().data() + batch_index * batch_byte_size, rotated_batch.data().data(),
                    batch_byte_size);
    }

    return result;
}

tensors::Tensor scaled_position_causal_softmax_last_dim(const tensors::TensorView& input, float scale,
                                                        std::span<const std::size_t> query_position_offsets,
                                                        std::span<const std::size_t> query_lengths,
                                                        std::span<const std::size_t> key_lengths) {
    if (input.tensor_info().shape.rank() != 4) {
        throw std::invalid_argument("qwen_attention batched softmax requires a rank-4 tensor.");
    }

    if (!std::isfinite(scale)) {
        throw std::invalid_argument("qwen_attention batched softmax requires a finite scale.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto batch_size = checked_positive_dim_to_size(dims[0], "qwen_attention batch size");
    const auto head_count = checked_positive_dim_to_size(dims[1], "qwen_attention head count");
    const auto query_capacity = checked_positive_dim_to_size(dims[2], "qwen_attention query length");
    const auto key_capacity = checked_positive_dim_to_size(dims[3], "qwen_attention key length");
    validate_sequence_lengths(query_position_offsets, batch_size, std::numeric_limits<std::size_t>::max(),
                              "qwen_attention query offsets");
    validate_sequence_lengths(query_lengths, batch_size, query_capacity, "qwen_attention query lengths");
    validate_sequence_lengths(key_lengths, batch_size, key_capacity, "qwen_attention key lengths");

    std::optional<tensors::Tensor> input_storage;
    const auto input_f32 =
        ops::detail::maybe_cast_to_dtype(input, tensors::DType::F32, input_storage, "qwen_attention_probabilities");

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        "qwen_attention_probabilities", tensors::DType::F32, input.tensor_info().shape));
    const auto matrix_size = query_capacity * key_capacity;
    for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        for (std::size_t head_index = 0; head_index < head_count; ++head_index) {
            const auto matrix_offset = (batch_index * head_count + head_index) * matrix_size;
            for (std::size_t query_index = 0; query_index < query_capacity; ++query_index) {
                const auto row_offset = matrix_offset + query_index * key_capacity;
                if (query_index >= query_lengths[batch_index]) {
                    continue;
                }

                const auto query_position = query_position_offsets[batch_index] + query_index;
                auto max_value = -std::numeric_limits<float>::infinity();
                for (std::size_t key_index = 0; key_index < key_lengths[batch_index]; ++key_index) {
                    if (key_index <= query_position) {
                        const auto value = ops::detail::load_float_value(input_f32, row_offset + key_index) * scale;
                        max_value = std::max(max_value, value);
                    }
                }

                auto sum = 0.0f;
                for (std::size_t key_index = 0; key_index < key_capacity; ++key_index) {
                    const auto flat_index = row_offset + key_index;
                    const auto is_visible = key_index < key_lengths[batch_index] && key_index <= query_position;
                    const auto value =
                        is_visible ? std::exp(ops::detail::load_float_value(input_f32, flat_index) * scale - max_value)
                                   : 0.0f;
                    ops::detail::store_float_value(tensors::DType::F32, result.mutable_data(), flat_index, value);
                    sum += value;
                }

                for (std::size_t key_index = 0; key_index < key_capacity; ++key_index) {
                    const auto flat_index = row_offset + key_index;
                    const auto value =
                        sum == 0.0f
                            ? 0.0f
                            : ops::detail::load_float_value(tensors::DType::F32, result.data(), flat_index) / sum;
                    ops::detail::store_float_value(tensors::DType::F32, result.mutable_data(), flat_index, value);
                }
            }
        }
    }

    return result;
}

tensors::Tensor fused_causal_attention(const tensors::TensorView& query, const tensors::TensorView& key,
                                       const tensors::TensorView& value,
                                       std::span<const std::size_t> query_position_offsets,
                                       std::span<const std::size_t> query_lengths,
                                       std::span<const std::size_t> key_lengths) {
    if (query.tensor_info().shape.rank() != 4 || key.tensor_info().shape.rank() != 4 ||
        value.tensor_info().shape.rank() != 4) {
        throw std::invalid_argument("qwen_attention requires rank-4 query, key, and value tensors.");
    }

    const auto& query_dims = query.tensor_info().shape.dims();
    const auto& key_dims = key.tensor_info().shape.dims();
    const auto& value_dims = value.tensor_info().shape.dims();
    const auto batch_size = checked_positive_dim_to_size(query_dims[0], "qwen_attention batch size");
    const auto query_sequence_capacity =
        checked_positive_dim_to_size(query_dims[1], "qwen_attention query sequence length");
    const auto query_head_count = checked_positive_dim_to_size(query_dims[2], "qwen_attention query head count");
    const auto key_sequence_capacity = checked_positive_dim_to_size(key_dims[1], "qwen_attention key sequence length");
    const auto key_head_count = checked_positive_dim_to_size(key_dims[2], "qwen_attention key head count");
    const auto value_sequence_capacity =
        checked_positive_dim_to_size(value_dims[1], "qwen_attention value sequence length");
    const auto value_head_count = checked_positive_dim_to_size(value_dims[2], "qwen_attention value head count");
    const auto query_head_size = checked_positive_dim_to_size(query_dims[3], "qwen_attention query head size");
    const auto key_head_size = checked_positive_dim_to_size(key_dims[3], "qwen_attention key head size");
    const auto value_head_size = checked_positive_dim_to_size(value_dims[3], "qwen_attention value head size");
    if (key_dims[0] != query_dims[0] || value_dims[0] != query_dims[0]) {
        throw std::invalid_argument("qwen_attention requires matching batch dimensions.");
    }
    if (key_head_count != value_head_count) {
        throw std::invalid_argument("qwen_attention requires matching key/value head counts.");
    }
    if (query_head_count % key_head_count != 0) {
        throw std::invalid_argument("qwen_attention requires query heads to divide by key/value heads.");
    }
    if (key_sequence_capacity != value_sequence_capacity) {
        throw std::invalid_argument("qwen_attention requires matching key/value sequence lengths.");
    }
    if (query_head_size != key_head_size || query_head_size != value_head_size) {
        throw std::invalid_argument("qwen_attention requires matching query/key/value head sizes.");
    }
    validate_sequence_lengths(query_position_offsets, batch_size, std::numeric_limits<std::size_t>::max(),
                              "qwen_attention query offsets");
    validate_sequence_lengths(query_lengths, batch_size, query_sequence_capacity, "qwen_attention query lengths");
    validate_sequence_lengths(key_lengths, batch_size, key_sequence_capacity, "qwen_attention key lengths");

    const auto score_scale = 1.0f / std::sqrt(static_cast<float>(query_head_size));
    const auto queries_per_key_value_head = query_head_count / key_head_count;

    // Move from token-major Q/K/V to [batch, heads, ...] mini-batches so the matmul batch dimension becomes batch *
    // heads while grouped-query attention repeats shared K/V heads within each batch row.
    const std::array<std::size_t, 4> query_axes = {0, 2, 1, 3};
    const std::array<std::size_t, 4> key_axes = {0, 2, 3, 1};
    const std::array<std::size_t, 4> value_axes = {0, 2, 1, 3};
    const auto query_by_head = ops::permute(query, query_axes);
    const auto key_by_kv_head = ops::permute(key, key_axes);
    const auto value_by_kv_head = ops::permute(value, value_axes);
    const auto key_by_head = ops::repeat_interleave(key_by_kv_head.view(), 1, queries_per_key_value_head);
    const auto value_by_head = ops::repeat_interleave(value_by_kv_head.view(), 1, queries_per_key_value_head);

    const auto head_batch_size = batch_size * query_head_count;
    const auto flat_query = ops::reshape(
        query_by_head.view(), tensors::Shape({checked_size_to_dim(head_batch_size, "qwen_attention flat query batch"),
                                              static_cast<std::int64_t>(query_sequence_capacity),
                                              static_cast<std::int64_t>(query_head_size)}));
    const auto flat_key = ops::reshape(
        key_by_head.view(),
        tensors::Shape({checked_size_to_dim(head_batch_size, "qwen_attention flat key batch"),
                        static_cast<std::int64_t>(query_head_size), static_cast<std::int64_t>(key_sequence_capacity)}));
    const auto flat_value = ops::reshape(
        value_by_head.view(),
        tensors::Shape({checked_size_to_dim(head_batch_size, "qwen_attention flat value batch"),
                        static_cast<std::int64_t>(key_sequence_capacity), static_cast<std::int64_t>(query_head_size)}));

    const auto attention_scores =
        ops::matmul(flat_query, flat_key, ops::MatmulOptions{.output_dtype = tensors::DType::F32});
    auto attention_scores_batched = tensors::materialize_tensor(
        "qwen_attention_scores",
        ops::reshape(attention_scores.view(),
                     tensors::Shape({static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(query_head_count),
                                     static_cast<std::int64_t>(query_sequence_capacity),
                                     static_cast<std::int64_t>(key_sequence_capacity)})));

    // Scale by 1/sqrt(head_dim), mask future or padded keys, and softmax across source positions so each row becomes
    // weights over the visible context.
    auto probabilities = scaled_position_causal_softmax_last_dim(attention_scores_batched.view(), score_scale,
                                                                 query_position_offsets, query_lengths, key_lengths);

    // Hugging Face Qwen casts BF16 attention probabilities back down before the value matmul, so mirror that dtype
    // boundary for golden parity.
    probabilities = ops::detail::maybe_cast_result(std::move(probabilities), query.tensor_info().dtype,
                                                   "qwen_attention_probabilities");

    const auto flat_probabilities =
        ops::reshape(probabilities.view(),
                     tensors::Shape({checked_size_to_dim(head_batch_size, "qwen_attention flat probability batch"),
                                     static_cast<std::int64_t>(query_sequence_capacity),
                                     static_cast<std::int64_t>(key_sequence_capacity)}));
    auto batch_result = ops::matmul(flat_probabilities, flat_value);
    batch_result = ops::detail::maybe_cast_result(std::move(batch_result), query.tensor_info().dtype,
                                                  "qwen_attention_attention_output_batch");

    auto attention_output = tensors::materialize_tensor(
        "qwen_attention_context_by_head",
        ops::reshape(batch_result.view(),
                     tensors::Shape({static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(query_head_count),
                                     static_cast<std::int64_t>(query_sequence_capacity),
                                     static_cast<std::int64_t>(query_head_size)})));
    const std::array<std::size_t, 4> output_axes = {0, 2, 1, 3};
    return ops::permute(attention_output.view(), output_axes);
}

tensors::Tensor qwen_attention_impl(const tensors::TensorView& hidden_states,
                                    std::span<const std::size_t> sequence_lengths, const QwenAttentionWeights& weights,
                                    QwenAttentionCache* cache, std::size_t num_attention_heads,
                                    std::size_t num_key_value_heads, std::size_t head_dim, float norm_epsilon,
                                    float rope_base) {
    validate_qwen_attention_inputs(hidden_states, sequence_lengths, weights, num_attention_heads, num_key_value_heads,
                                   head_dim, norm_epsilon, rope_base,
                                   cache ? "qwen_attention_with_cache" : "qwen_attention");

    const auto batch_size =
        checked_positive_dim_to_size(hidden_states.tensor_info().shape.dims()[0], "qwen_attention batch size");
    const auto sequence_capacity =
        checked_positive_dim_to_size(hidden_states.tensor_info().shape.dims()[1], "qwen_attention sequence length");
    if (cache != nullptr && cache->sequence_lengths.size() != batch_size) {
        throw std::invalid_argument("qwen_attention cache batch size must match the hidden state batch size.");
    }

    // Project each token embedding [batch, seq, hidden] into the three streams attention needs: queries ask what this
    // token is looking for, keys describe what each token offers, and values carry the information to mix.
    const auto query_projection =
        linear_project(hidden_states, weights.q_proj_weight, "qwen_attention_query_projection");
    const auto key_projection = linear_project(hidden_states, weights.k_proj_weight, "qwen_attention_key_projection");
    const auto value_projection =
        linear_project(hidden_states, weights.v_proj_weight, "qwen_attention_value_projection");

    // Reinterpret the packed per-token layout [batch, seq, heads * head_dim] as [batch, seq, heads, head_dim] without
    // copying.
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

    std::vector<std::size_t> query_position_offsets(batch_size, 0);
    if (cache != nullptr) {
        query_position_offsets = cache->sequence_lengths;
    }
    const auto rotated_query =
        apply_rope_batched(normalized_query.view(), query_position_offsets, rope_base, "qwen_attention_query_rope");
    const auto rotated_key =
        apply_rope_batched(normalized_key.view(), query_position_offsets, rope_base, "qwen_attention_key_rope");

    std::vector<std::size_t> key_lengths(sequence_lengths.begin(), sequence_lengths.end());
    tensors::TensorView cached_key = rotated_key.view();
    tensors::TensorView cached_value = value_heads;
    if (cache != nullptr) {
        append_to_qwen_attention_cache(*cache, rotated_key.view(), value_heads, sequence_lengths);
        key_lengths = cache->sequence_lengths;
        cached_key = cache->key.view();
        cached_value = cache->value.view();
    }

    // Build token context with causal attention, then flatten [batch, seq, q_heads, head_dim] back to
    // [batch, seq, q_heads * head_dim] for the output projection.
    const auto attention_output = fused_causal_attention(rotated_query.view(), cached_key, cached_value,
                                                         query_position_offsets, sequence_lengths, key_lengths);
    const auto packed_attention =
        ops::reshape(attention_output.view(), tensors::Shape({static_cast<std::int64_t>(batch_size),
                                                              static_cast<std::int64_t>(sequence_capacity),
                                                              query_projection.tensor_info().shape.dims()[2]}));

    // The output projection mixes information across heads and returns to the model hidden size expected by the
    // residual stream.
    return tensors::rename_tensor("qwen_attention_result",
                                  linear_project(packed_attention, weights.o_proj_weight, "qwen_attention_output"));
}

} // namespace

tensors::Tensor qwen_attention(const tensors::TensorView& hidden_states, std::span<const std::size_t> sequence_lengths,
                               const QwenAttentionWeights& weights, std::size_t num_attention_heads,
                               std::size_t num_key_value_heads, std::size_t head_dim, float norm_epsilon,
                               float rope_base) {
    const auto sequence_capacity =
        checked_positive_dim_to_size(hidden_states.tensor_info().shape.dims()[1], "qwen_attention sequence length");
    auto cache =
        make_qwen_attention_cache("qwen_attention_key", "qwen_attention_value", hidden_states.tensor_info().dtype,
                                  sequence_lengths.size(), sequence_capacity, num_key_value_heads, head_dim);
    return qwen_attention_impl(hidden_states, sequence_lengths, weights, &cache, num_attention_heads,
                               num_key_value_heads, head_dim, norm_epsilon, rope_base);
}

tensors::Tensor qwen_attention_with_cache(const tensors::TensorView& hidden_states,
                                          std::span<const std::size_t> sequence_lengths,
                                          const QwenAttentionWeights& weights, QwenAttentionCache& cache,
                                          std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                          std::size_t head_dim, float norm_epsilon, float rope_base) {
    return qwen_attention_impl(hidden_states, sequence_lengths, weights, &cache, num_attention_heads,
                               num_key_value_heads, head_dim, norm_epsilon, rope_base);
}

} // namespace cppinf::nn
