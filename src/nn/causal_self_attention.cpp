#include "nn/causal_self_attention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
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
    if (scores.tensor_info().dtype != tensors::DType::F32 || scores.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("scale_and_causal_mask_scores requires a rank-2 f32 tensor.");
    }

    const auto& dims = scores.tensor_info().shape.dims();
    const auto query_sequence_length = checked_positive_dim_to_size(dims[0], "attention query sequence length");
    const auto key_sequence_length = checked_positive_dim_to_size(dims[1], "attention key sequence length");

    auto masked_scores = tensors::Tensor::zeros(
        tensors::make_result_tensor_info("causal_attention_scores", tensors::DType::F32, scores.tensor_info().shape));

    for (std::size_t query_index = 0; query_index < query_sequence_length; ++query_index) {
        const auto last_allowed_key = past_sequence_length + query_index;
        for (std::size_t key_index = 0; key_index < key_sequence_length; ++key_index) {
            const auto flat_index = query_index * key_sequence_length + key_index;
            const auto value =
                key_index <= last_allowed_key
                    ? ops::detail::load_float_value(scores.tensor_info().dtype, scores.data(), flat_index) * scale
                    : -std::numeric_limits<float>::infinity();
            ops::detail::store_float_value(tensors::DType::F32, masked_scores.mutable_data(), flat_index, value);
        }
    }

    return masked_scores;
}

void copy_head_output_to_result(const tensors::Tensor& head_output, std::size_t head_index, tensors::Tensor& result) {
    const auto byte_offset = head_index * head_output.byte_size();
    std::memcpy(result.mutable_data().data() + byte_offset, head_output.data().data(), head_output.byte_size());
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
    const auto& key_dims = key.tensor_info().shape.dims();
    const auto& value_dims = value.tensor_info().shape.dims();

    const auto head_count = static_cast<std::size_t>(query_dims[0]);
    const auto query_sequence_length = static_cast<std::size_t>(query_dims[1]);
    const auto query_head_size = static_cast<std::size_t>(query_dims[2]);
    const auto key_sequence_length = static_cast<std::size_t>(key_dims[1]);
    const auto value_head_size = static_cast<std::size_t>(value_dims[2]);
    const auto score_scale = 1.0f / std::sqrt(static_cast<float>(query_head_size));

    std::optional<tensors::Tensor> query_storage;
    std::optional<tensors::Tensor> key_storage;
    std::optional<tensors::Tensor> value_storage;
    // Score construction stays in f32 so masking and softmax keep their stable reference behavior.
    const auto query_f32 =
        ops::detail::maybe_cast_to_dtype(query, tensors::DType::F32, query_storage, "causal_self_attention_result");
    const auto key_f32 =
        ops::detail::maybe_cast_to_dtype(key, tensors::DType::F32, key_storage, "causal_self_attention_result");
    const auto value_f32 =
        ops::detail::maybe_cast_to_dtype(value, tensors::DType::F32, value_storage, "causal_self_attention_result");

    auto result_f32 = tensors::Tensor::zeros(
        tensors::make_result_tensor_info("causal_self_attention_result", tensors::DType::F32,
                                         tensors::Shape({query_dims[0], query_dims[1], value_dims[2]})));

    for (std::size_t head_index = 0; head_index < head_count; ++head_index) {
        const auto query_head = ops::reshape(ops::narrow(query_f32, 0, head_index, 1),
                                             tensors::Shape({static_cast<std::int64_t>(query_sequence_length),
                                                             static_cast<std::int64_t>(query_head_size)}));
        const auto key_head = ops::reshape(ops::narrow(key_f32, 0, head_index, 1),
                                           tensors::Shape({static_cast<std::int64_t>(key_sequence_length),
                                                           static_cast<std::int64_t>(query_head_size)}));
        const auto value_head = ops::reshape(ops::narrow(value_f32, 0, head_index, 1),
                                             tensors::Shape({static_cast<std::int64_t>(key_sequence_length),
                                                             static_cast<std::int64_t>(value_head_size)}));

        const auto transposed_key = ops::transpose_2d(key_head);
        const auto attention_scores = ops::matmul(query_head, transposed_key.view());
        const auto masked_scores =
            scale_and_causal_mask_scores(attention_scores.view(), score_scale, past_sequence_length);
        const auto probabilities = ops::softmax_last_dim(masked_scores.view());
        const auto head_output = ops::matmul(probabilities.view(), value_head);
        copy_head_output_to_result(head_output, head_index, result_f32);
    }

    return ops::detail::maybe_cast_result(std::move(result_f32), query.tensor_info().dtype,
                                          "causal_self_attention_result");
}

} // namespace cppinf::nn
