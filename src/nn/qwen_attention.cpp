#include "nn/qwen_attention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

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

tensors::Tensor scale_and_causal_mask_scores(const tensors::TensorView& scores, float scale) {
    const auto rank = scores.tensor_info().shape.rank();
    if (rank != 2 && rank != 3) {
        throw std::invalid_argument("qwen_attention requires rank-2 or rank-3 attention score matrices.");
    }

    const auto& dims = scores.tensor_info().shape.dims();
    const auto batch_count =
        rank == 3 ? checked_positive_dim_to_size(dims[0], "qwen_attention score batch count") : std::size_t{1};
    const auto sequence_dim = rank == 3 ? std::size_t{1} : std::size_t{0};
    const auto sequence_length = checked_positive_dim_to_size(dims[sequence_dim], "qwen_attention sequence length");
    const auto key_sequence_length =
        checked_positive_dim_to_size(dims[sequence_dim + 1], "qwen_attention key sequence length");
    if (sequence_length != key_sequence_length) {
        throw std::invalid_argument("qwen_attention requires square attention score matrices.");
    }

    auto masked_scores = tensors::Tensor::zeros(
        tensors::make_result_tensor_info("qwen_attention_scores", tensors::DType::F32, scores.tensor_info().shape));

    for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        for (std::size_t query_index = 0; query_index < sequence_length; ++query_index) {
            for (std::size_t key_index = 0; key_index < key_sequence_length; ++key_index) {
                const auto matrix_index = query_index * key_sequence_length + key_index;
                const auto flat_index = (batch_index * sequence_length * key_sequence_length) + matrix_index;
                const auto value =
                    key_index <= query_index
                        ? ops::detail::load_float_value(scores.tensor_info().dtype, scores.data(), flat_index) * scale
                        : -std::numeric_limits<float>::infinity();
                ops::detail::store_float_value(tensors::DType::F32, masked_scores.mutable_data(), flat_index, value);
            }
        }
    }

    return masked_scores;
}

tensors::Tensor linear_project(const tensors::TensorView& input, const tensors::TensorView& weight) {
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

tensors::Tensor make_grouped_query_batch(const tensors::TensorView& query, std::size_t query_head_begin,
                                         std::size_t queries_per_key_value_head) {
    const auto& dims = query.tensor_info().shape.dims();
    const auto sequence_length = static_cast<std::size_t>(dims[0]);
    const auto query_head_count = static_cast<std::size_t>(dims[1]);
    const auto head_size = static_cast<std::size_t>(dims[2]);

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        "qwen_attention_query_batch", query.tensor_info().dtype,
        tensors::Shape({static_cast<std::int64_t>(queries_per_key_value_head), dims[0], dims[2]})));
    const auto element_size = tensors::element_size_bytes(query.tensor_info().dtype);
    const auto head_bytes = head_size * element_size;
    auto result_data = result.mutable_data();
    const auto query_data = query.data();

    for (std::size_t group_head_index = 0; group_head_index < queries_per_key_value_head; ++group_head_index) {
        const auto query_head_index = query_head_begin + group_head_index;
        for (std::size_t sequence_index = 0; sequence_index < sequence_length; ++sequence_index) {
            const auto source_offset = ((sequence_index * query_head_count) + query_head_index) * head_bytes;
            const auto destination_offset = ((group_head_index * sequence_length) + sequence_index) * head_bytes;
            std::memcpy(result_data.data() + destination_offset, query_data.data() + source_offset, head_bytes);
        }
    }

    return result;
}

tensors::Tensor make_grouped_key_batch(const tensors::TensorView& key, std::size_t key_value_head_index,
                                       std::size_t queries_per_key_value_head) {
    const auto& dims = key.tensor_info().shape.dims();
    const auto sequence_length = static_cast<std::size_t>(dims[0]);
    const auto key_head_count = static_cast<std::size_t>(dims[1]);
    const auto head_size = static_cast<std::size_t>(dims[2]);

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        "qwen_attention_key_batch", key.tensor_info().dtype,
        tensors::Shape({static_cast<std::int64_t>(queries_per_key_value_head), dims[2], dims[0]})));
    const auto element_size = tensors::element_size_bytes(key.tensor_info().dtype);
    const auto key_data = key.data();
    auto result_data = result.mutable_data();

    for (std::size_t group_head_index = 0; group_head_index < queries_per_key_value_head; ++group_head_index) {
        for (std::size_t head_index = 0; head_index < head_size; ++head_index) {
            for (std::size_t sequence_index = 0; sequence_index < sequence_length; ++sequence_index) {
                const auto source_index =
                    ((sequence_index * key_head_count + key_value_head_index) * head_size) + head_index;
                const auto destination_index =
                    ((group_head_index * head_size + head_index) * sequence_length) + sequence_index;
                std::memcpy(result_data.data() + destination_index * element_size,
                            key_data.data() + source_index * element_size, element_size);
            }
        }
    }

    return result;
}

tensors::Tensor make_grouped_value_batch(const tensors::TensorView& value, std::size_t key_value_head_index,
                                         std::size_t queries_per_key_value_head) {
    const auto& dims = value.tensor_info().shape.dims();
    const auto sequence_length = static_cast<std::size_t>(dims[0]);
    const auto value_head_count = static_cast<std::size_t>(dims[1]);
    const auto head_size = static_cast<std::size_t>(dims[2]);

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        "qwen_attention_value_batch", value.tensor_info().dtype,
        tensors::Shape({static_cast<std::int64_t>(queries_per_key_value_head), dims[0], dims[2]})));
    const auto element_size = tensors::element_size_bytes(value.tensor_info().dtype);
    const auto head_bytes = head_size * element_size;
    const auto value_data = value.data();
    auto result_data = result.mutable_data();

    for (std::size_t group_head_index = 0; group_head_index < queries_per_key_value_head; ++group_head_index) {
        for (std::size_t sequence_index = 0; sequence_index < sequence_length; ++sequence_index) {
            const auto source_offset = ((sequence_index * value_head_count) + key_value_head_index) * head_bytes;
            const auto destination_offset = ((group_head_index * sequence_length) + sequence_index) * head_bytes;
            std::memcpy(result_data.data() + destination_offset, value_data.data() + source_offset, head_bytes);
        }
    }

    return result;
}

void copy_grouped_attention_result(const tensors::TensorView& group_result, std::size_t query_head_begin,
                                   std::size_t query_head_count, std::span<std::byte> result_data) {
    const auto& dims = group_result.tensor_info().shape.dims();
    const auto queries_per_key_value_head = static_cast<std::size_t>(dims[0]);
    const auto sequence_length = static_cast<std::size_t>(dims[1]);
    const auto head_size = static_cast<std::size_t>(dims[2]);
    const auto head_bytes = head_size * tensors::element_size_bytes(group_result.tensor_info().dtype);
    const auto group_data = group_result.data();

    for (std::size_t group_head_index = 0; group_head_index < queries_per_key_value_head; ++group_head_index) {
        const auto query_head_index = query_head_begin + group_head_index;
        for (std::size_t sequence_index = 0; sequence_index < sequence_length; ++sequence_index) {
            const auto source_offset = ((group_head_index * sequence_length) + sequence_index) * head_bytes;
            const auto destination_offset = ((sequence_index * query_head_count) + query_head_index) * head_bytes;
            std::memcpy(result_data.data() + destination_offset, group_data.data() + source_offset, head_bytes);
        }
    }
}

tensors::Tensor fused_causal_attention(const tensors::TensorView& query, const tensors::TensorView& key,
                                       const tensors::TensorView& value) {
    if (query.tensor_info().shape.rank() != 3 || key.tensor_info().shape.rank() != 3 ||
        value.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("qwen_attention requires rank-3 query, key, and value tensors.");
    }

    const auto& query_dims = query.tensor_info().shape.dims();
    const auto& key_dims = key.tensor_info().shape.dims();
    const auto& value_dims = value.tensor_info().shape.dims();
    const auto sequence_length = checked_positive_dim_to_size(query_dims[0], "qwen_attention query sequence length");
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

    if (sequence_length != key_sequence_length || sequence_length != value_sequence_length) {
        throw std::invalid_argument("qwen_attention requires matching query/key/value sequence lengths.");
    }

    if (query_head_size != key_head_size || query_head_size != value_head_size) {
        throw std::invalid_argument("qwen_attention requires matching query/key/value head sizes.");
    }

    const auto score_scale = 1.0f / std::sqrt(static_cast<float>(query_head_size));
    const auto queries_per_key_value_head = query_head_count / key_head_count;
    auto result = tensors::Tensor::zeros(
        tensors::make_result_tensor_info("qwen_attention_attention_output", query.tensor_info().dtype,
                                         tensors::Shape({query_dims[0], query_dims[1], query_dims[2]})));
    auto result_data = result.mutable_data();

    for (std::size_t key_value_head_index = 0; key_value_head_index < key_head_count; ++key_value_head_index) {
        const auto query_head_begin = key_value_head_index * queries_per_key_value_head;

        const auto query_batch = make_grouped_query_batch(query, query_head_begin, queries_per_key_value_head);
        const auto key_batch = make_grouped_key_batch(key, key_value_head_index, queries_per_key_value_head);
        const auto value_batch = make_grouped_value_batch(value, key_value_head_index, queries_per_key_value_head);
        const auto attention_scores =
            ops::matmul(query_batch.view(), key_batch.view(), ops::MatmulOptions{.output_dtype = tensors::DType::F32});
        const auto masked_scores = scale_and_causal_mask_scores(attention_scores.view(), score_scale);
        const auto probabilities = ops::softmax_last_dim(masked_scores.view());
        auto group_result = ops::matmul(probabilities.view(), value_batch.view());
        group_result = ops::detail::maybe_cast_result(std::move(group_result), query.tensor_info().dtype,
                                                      "qwen_attention_attention_output_group");
        copy_grouped_attention_result(group_result.view(), query_head_begin, query_head_count, result_data);
    }

    return result;
}

} // namespace

tensors::Tensor qwen_attention(const tensors::TensorView& hidden_states, const QwenAttentionWeights& weights,
                               std::size_t num_attention_heads, std::size_t num_key_value_heads, std::size_t head_dim,
                               float norm_epsilon, std::size_t sequence_position_offset, float rope_base) {
    validate_qwen_attention_inputs(hidden_states, weights, num_attention_heads, num_key_value_heads, head_dim,
                                   norm_epsilon, rope_base);

    // Project hidden states [seq, hidden] into packed query, key, and value channels with shapes
    // [seq, q_heads * head_dim], [seq, kv_heads * head_dim], and [seq, kv_heads * head_dim].
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

    // Normalize queries and keys per head, then apply RoPE so token position rotates each feature pair before
    // attention sees the vectors. Keep the packed [seq, heads, dim] layout so split/merge stay metadata-only.
    const auto normalized_query = ops::rms_norm(query_heads, weights.q_norm_weight, norm_epsilon);
    const auto normalized_key = ops::rms_norm(key_heads, weights.k_norm_weight, norm_epsilon);
    const auto rotated_query = apply_rope(normalized_query.view(), sequence_position_offset, rope_base, 0);
    const auto rotated_key = apply_rope(normalized_key.view(), sequence_position_offset, rope_base, 0);

    // Keep keys and values at KV-head cardinality, then map each query head to its grouped KV head inside the fused
    // causal attention core instead of materializing repeated K/V tensors.
    const auto attention_output = fused_causal_attention(rotated_query.view(), rotated_key.view(), value_heads);
    const auto packed_attention =
        ops::reshape(attention_output.view(), tensors::Shape({query_projection.tensor_info().shape.dims()[0],
                                                              query_projection.tensor_info().shape.dims()[1]}));
    return tensors::rename_tensor("qwen_attention_result", linear_project(packed_attention, weights.o_proj_weight));
}

} // namespace cppinf::nn
