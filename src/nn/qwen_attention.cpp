#include "nn/qwen_attention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
    const auto& dims = scores.tensor_info().shape.dims();
    const auto head_count = checked_positive_dim_to_size(dims[0], "qwen_attention attention head count");
    const auto sequence_length = checked_positive_dim_to_size(dims[1], "qwen_attention sequence length");
    const auto key_sequence_length = checked_positive_dim_to_size(dims[2], "qwen_attention key sequence length");
    if (sequence_length != key_sequence_length) {
        throw std::invalid_argument("qwen_attention requires square attention score matrices.");
    }

    auto masked_scores = tensors::Tensor::zeros(
        tensors::make_result_tensor_info("qwen_attention_scores", tensors::DType::F32, scores.tensor_info().shape));

    for (std::size_t head_index = 0; head_index < head_count; ++head_index) {
        for (std::size_t query_index = 0; query_index < sequence_length; ++query_index) {
            for (std::size_t key_index = 0; key_index < key_sequence_length; ++key_index) {
                const auto flat_index =
                    ((head_index * sequence_length) + query_index) * key_sequence_length + key_index;
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

tensors::Tensor split_heads(const tensors::TensorView& input, std::size_t head_count, std::size_t head_dim,
                            std::string_view result_name) {
    if (input.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("split_heads requires a rank-2 tensor.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto sequence_length = checked_positive_dim_to_size(dims[0], fmt::format("{} sequence length", result_name));
    const auto merged_head_size =
        checked_positive_dim_to_size(dims[1], fmt::format("{} merged head size", result_name));
    if (merged_head_size != head_count * head_dim) {
        throw std::invalid_argument(
            fmt::format("{} requires merged head size to equal head_count * head_dim.", result_name));
    }

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        result_name, input.tensor_info().dtype,
        tensors::Shape({static_cast<std::int64_t>(head_count), static_cast<std::int64_t>(sequence_length),
                        static_cast<std::int64_t>(head_dim)})));

    // Move from the packed per-token layout [seq, heads * head_dim] to the head-major layout [heads, seq, head_dim]
    // used by attention.
    const auto element_size = tensors::element_size_bytes(input.tensor_info().dtype);
    const auto head_bytes = head_dim * element_size;
    for (std::size_t head_index = 0; head_index < head_count; ++head_index) {
        for (std::size_t sequence_index = 0; sequence_index < sequence_length; ++sequence_index) {
            const auto source_offset = (sequence_index * merged_head_size + head_index * head_dim) * element_size;
            const auto destination_offset = ((head_index * sequence_length) + sequence_index) * head_bytes;
            std::memcpy(result.mutable_data().data() + destination_offset, input.data().data() + source_offset,
                        head_bytes);
        }
    }

    return result;
}

tensors::Tensor merge_heads(const tensors::TensorView& input, std::string_view result_name) {
    if (input.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("merge_heads requires a rank-3 tensor.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto head_count = checked_positive_dim_to_size(dims[0], fmt::format("{} head count", result_name));
    const auto sequence_length = checked_positive_dim_to_size(dims[1], fmt::format("{} sequence length", result_name));
    const auto head_dim = checked_positive_dim_to_size(dims[2], fmt::format("{} head dim", result_name));
    const auto merged_head_size = head_count * head_dim;

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        result_name, input.tensor_info().dtype,
        tensors::Shape({static_cast<std::int64_t>(sequence_length), static_cast<std::int64_t>(merged_head_size)})));

    // Undo split_heads by packing the head-major layout [heads, seq, head_dim] back into [seq, heads * head_dim].
    const auto element_size = tensors::element_size_bytes(input.tensor_info().dtype);
    const auto head_bytes = head_dim * element_size;
    for (std::size_t head_index = 0; head_index < head_count; ++head_index) {
        for (std::size_t sequence_index = 0; sequence_index < sequence_length; ++sequence_index) {
            const auto source_offset = ((head_index * sequence_length) + sequence_index) * head_bytes;
            const auto destination_offset = (sequence_index * merged_head_size + head_index * head_dim) * element_size;
            std::memcpy(result.mutable_data().data() + destination_offset, input.data().data() + source_offset,
                        head_bytes);
        }
    }

    return result;
}

tensors::Tensor repeat_heads(const tensors::TensorView& input, std::size_t target_head_count,
                             std::string_view result_name) {
    if (input.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("repeat_heads requires a rank-3 tensor.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto input_head_count = checked_positive_dim_to_size(dims[0], fmt::format("{} head count", result_name));
    const auto sequence_length = checked_positive_dim_to_size(dims[1], fmt::format("{} sequence length", result_name));
    const auto head_dim = checked_positive_dim_to_size(dims[2], fmt::format("{} head dim", result_name));
    if (target_head_count % input_head_count != 0) {
        throw std::invalid_argument(
            fmt::format("{} requires target head count to divide by input head count.", result_name));
    }

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        result_name, input.tensor_info().dtype,
        tensors::Shape({static_cast<std::int64_t>(target_head_count), static_cast<std::int64_t>(sequence_length),
                        static_cast<std::int64_t>(head_dim)})));

    // Grouped-query attention keeps fewer key/value heads than query heads, so duplicate each K/V head until the
    // tensor has one head slot per query head.
    const auto repeat_count = target_head_count / input_head_count;
    const auto bytes_per_head = sequence_length * head_dim * tensors::element_size_bytes(input.tensor_info().dtype);
    for (std::size_t head_index = 0; head_index < target_head_count; ++head_index) {
        const auto source_head_index = head_index / repeat_count;
        std::memcpy(result.mutable_data().data() + head_index * bytes_per_head,
                    input.data().data() + source_head_index * bytes_per_head, bytes_per_head);
    }

    return result;
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

tensors::Tensor fused_causal_attention(const tensors::TensorView& query, const tensors::TensorView& key,
                                       const tensors::TensorView& value) {
    const auto& query_dims = query.tensor_info().shape.dims();
    const auto query_head_size = checked_positive_dim_to_size(query_dims[2], "qwen_attention query head size");
    const auto score_scale = 1.0f / std::sqrt(static_cast<float>(query_head_size));

    const auto transposed_key = ops::transpose_last_two_dims(key);
    const auto attention_scores =
        ops::matmul(query, transposed_key.view(), ops::MatmulOptions{.output_dtype = tensors::DType::F32});
    const auto masked_scores = scale_and_causal_mask_scores(attention_scores.view(), score_scale);
    const auto probabilities = ops::softmax_last_dim(masked_scores.view());
    auto result_f32 = ops::matmul(probabilities.view(), value);
    result_f32 = tensors::rename_tensor("qwen_attention_attention_output", result_f32);
    return ops::detail::maybe_cast_result(std::move(result_f32), query.tensor_info().dtype,
                                          "qwen_attention_attention_output");
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

    // Split the packed head dimension so attention can work head-by-head: queries become [q_heads, seq, head_dim], and
    // keys/values become [kv_heads, seq, head_dim].
    const auto query_heads =
        split_heads(query_projection.view(), num_attention_heads, head_dim, "qwen_attention_query_heads");
    const auto key_heads =
        split_heads(key_projection.view(), num_key_value_heads, head_dim, "qwen_attention_key_heads");
    const auto value_heads =
        split_heads(value_projection.view(), num_key_value_heads, head_dim, "qwen_attention_value_heads");

    // Normalize queries and keys per head, then apply RoPE so token position rotates each feature pair before
    // attention sees the vectors.
    const auto normalized_query = ops::rms_norm(query_heads.view(), weights.q_norm_weight, norm_epsilon);
    const auto normalized_key = ops::rms_norm(key_heads.view(), weights.k_norm_weight, norm_epsilon);
    const auto rotated_query = apply_rope(normalized_query.view(), sequence_position_offset, rope_base);
    const auto rotated_key = apply_rope(normalized_key.view(), sequence_position_offset, rope_base);

    // Qwen uses grouped KV heads, so duplicate each key/value head until keys and values match the query-head count
    // [q_heads, seq, head_dim].
    const auto repeated_key = repeat_heads(rotated_key.view(), num_attention_heads, "qwen_attention_key_repeat");
    const auto repeated_value = repeat_heads(value_heads.view(), num_attention_heads, "qwen_attention_value_repeat");

    // Fuse the causal attention core into qwen_attention so the Qwen-specific path owns the score construction,
    // masking, softmax, and value mixing directly.
    const auto attention_output =
        fused_causal_attention(rotated_query.view(), repeated_key.view(), repeated_value.view());
    const auto merged_attention = merge_heads(attention_output.view(), "qwen_attention_merged_heads");
    return tensors::rename_tensor("qwen_attention_result",
                                  linear_project(merged_attention.view(), weights.o_proj_weight));
}

} // namespace cppinf::nn
