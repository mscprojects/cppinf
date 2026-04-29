#include "nn/causal_self_attention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

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

tensors::Tensor scale_and_causal_mask_scores(const tensors::TensorView& scores, float scale,
                                             std::size_t past_sequence_length) {
    if (scores.tensor_info().dtype != tensors::DType::F32 || scores.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("scale_and_causal_mask_scores requires a rank-3 f32 tensor.");
    }

    const auto& dims = scores.tensor_info().shape.dims();
    const auto head_count = checked_positive_dim_to_size(dims[0], "attention head count");
    const auto query_sequence_length = checked_positive_dim_to_size(dims[1], "attention query sequence length");
    const auto key_sequence_length = checked_positive_dim_to_size(dims[2], "attention key sequence length");

    auto masked_scores = tensors::Tensor::zeros(
        tensors::make_result_tensor_info("causal_attention_scores", tensors::DType::F32, scores.tensor_info().shape));

    // Each head shares the same causal rule: query position i may only see the cached prefix plus tokens up to i in the
    // current chunk, so future keys become -inf before softmax.
    for (std::size_t head_index = 0; head_index < head_count; ++head_index) {
        for (std::size_t query_index = 0; query_index < query_sequence_length; ++query_index) {
            const auto last_allowed_key = past_sequence_length + query_index;
            for (std::size_t key_index = 0; key_index < key_sequence_length; ++key_index) {
                const auto flat_index =
                    ((head_index * query_sequence_length) + query_index) * key_sequence_length + key_index;
                const auto value =
                    key_index <= last_allowed_key
                        ? ops::detail::load_float_value(scores.tensor_info().dtype, scores.data(), flat_index) * scale
                        : -std::numeric_limits<float>::infinity();
                ops::detail::store_float_value(tensors::DType::F32, masked_scores.mutable_data(), flat_index, value);
            }
        }
    }

    return masked_scores;
}

void validate_attention_inputs(const tensors::TensorView& query, const tensors::TensorView& key,
                               const tensors::TensorView& value, std::size_t past_sequence_length) {
    ops::detail::validate_supported_float_dtype(query.tensor_info().dtype, "causal_self_attention");
    if (query.tensor_info().dtype != key.tensor_info().dtype ||
        query.tensor_info().dtype != value.tensor_info().dtype) {
        throw std::invalid_argument("causal_self_attention requires matching tensor dtypes.");
    }

    if (query.tensor_info().shape.rank() != 3 || key.tensor_info().shape.rank() != 3 ||
        value.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("causal_self_attention requires rank-3 tensors.");
    }

    const auto& query_dims = query.tensor_info().shape.dims();
    const auto& key_dims = key.tensor_info().shape.dims();
    const auto& value_dims = value.tensor_info().shape.dims();

    const auto query_heads = checked_positive_dim_to_size(query_dims[0], "attention query heads");
    const auto key_heads = checked_positive_dim_to_size(key_dims[0], "attention key heads");
    const auto value_heads = checked_positive_dim_to_size(value_dims[0], "attention value heads");
    if (query_heads != key_heads || query_heads != value_heads) {
        throw std::invalid_argument("causal_self_attention requires matching head counts.");
    }

    const auto query_sequence_length = checked_positive_dim_to_size(query_dims[1], "attention query sequence length");
    const auto key_sequence_length = checked_positive_dim_to_size(key_dims[1], "attention key sequence length");
    const auto value_sequence_length = checked_positive_dim_to_size(value_dims[1], "attention value sequence length");
    if (key_sequence_length != value_sequence_length) {
        throw std::invalid_argument("causal_self_attention requires matching key and value sequence lengths.");
    }

    if (key_sequence_length != past_sequence_length + query_sequence_length) {
        throw std::invalid_argument(
            "causal_self_attention requires key/value sequence length to equal past_sequence_length + query length.");
    }

    const auto query_head_size = checked_positive_dim_to_size(query_dims[2], "attention query head size");
    const auto key_head_size = checked_positive_dim_to_size(key_dims[2], "attention key head size");
    checked_positive_dim_to_size(value_dims[2], "attention value head size");
    if (query_head_size != key_head_size) {
        throw std::invalid_argument("causal_self_attention requires matching query and key head sizes.");
    }
}

} // namespace

tensors::Tensor causal_self_attention(const tensors::TensorView& query, const tensors::TensorView& key,
                                      const tensors::TensorView& value, std::size_t past_sequence_length) {
    validate_attention_inputs(query, key, value, past_sequence_length);

    const auto& query_dims = query.tensor_info().shape.dims();
    const auto query_head_size = static_cast<std::size_t>(query_dims[2]);
    const auto score_scale = 1.0f / std::sqrt(static_cast<float>(query_head_size));

    // Keep q/k in their storage dtype and materialize the score tensor in f32, so the backend can handle bf16
    // input accumulation while masking and softmax still operate on stable f32 scores.
    // Keep the head axis batched so oneDNN can build all attention score matrices together:
    // [heads, seq_q, head_dim] x [heads, head_dim, seq_k] -> [heads, seq_q, seq_k].
    const auto transposed_key = ops::transpose_last_two_dims(key);
    const auto attention_scores =
        ops::matmul(query, transposed_key.view(), ops::MatmulOptions{.output_dtype = tensors::DType::F32});
    const auto masked_scores = scale_and_causal_mask_scores(attention_scores.view(), score_scale, past_sequence_length);
    const auto probabilities = ops::softmax_last_dim(masked_scores.view());

    // Apply the attention weights to V for every head in one go:
    // [heads, seq_q, seq_k] x [heads, seq_k, value_dim] -> [heads, seq_q, value_dim].
    auto result_f32 = ops::matmul(probabilities.view(), value);
    result_f32 = tensors::rename_tensor("causal_self_attention_result", result_f32);

    return ops::detail::maybe_cast_result(std::move(result_f32), query.tensor_info().dtype,
                                          "causal_self_attention_result");
}

} // namespace cppinf::nn
