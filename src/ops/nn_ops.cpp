#include "ops/nn_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

#include "ops/op_utils.h"

namespace cppinf::ops {
namespace {

tensors::TensorInfo make_result_info(std::string_view name, const tensors::TensorView& input) {
    return tensors::TensorInfo{
        .name = std::string(name),
        .dtype = input.tensor_info().dtype,
        .shape = input.tensor_info().shape,
        .byte_offset = 0,
    };
}

std::size_t checked_dim_to_size(std::int64_t dim, std::string_view field_name) {
    if (dim < 0) {
        throw std::invalid_argument(fmt::format("{} must be non-negative.", field_name));
    }

    return static_cast<std::size_t>(dim);
}

void validate_last_dim_operation_input(const tensors::TensorView& input, std::string_view op_name) {
    detail::validate_supported_float_dtype(input.tensor_info().dtype, op_name);
    if (input.tensor_info().shape.rank() == 0) {
        throw std::invalid_argument(fmt::format("{} requires a tensor with rank at least 1.", op_name));
    }
    if (checked_dim_to_size(input.tensor_info().shape.dims().back(), fmt::format("{} last dim", op_name)) == 0) {
        throw std::invalid_argument(fmt::format("{} requires a non-empty last dimension.", op_name));
    }
}

} // namespace

tensors::Tensor silu(const tensors::TensorView& input) {
    detail::validate_supported_float_dtype(input.tensor_info().dtype, "silu");

    tensors::Tensor result = tensors::Tensor::zeros(make_result_info("silu_result", input));
    const std::size_t element_count = input.tensor_info().shape.num_elements();
    for (std::size_t index = 0; index < element_count; ++index) {
        const float value = detail::load_float_value(input.tensor_info().dtype, input.data(), index);
        const float silu_value = value / (1.0f + std::exp(-value));
        detail::store_float_value(input.tensor_info().dtype, result.mutable_data(), index, silu_value);
    }

    return result;
}

tensors::Tensor softmax_last_dim(const tensors::TensorView& input) {
    validate_last_dim_operation_input(input, "softmax_last_dim");

    const std::size_t last_dim =
        checked_dim_to_size(input.tensor_info().shape.dims().back(), "softmax_last_dim last dim");
    const std::size_t row_count = input.tensor_info().shape.num_elements() / last_dim;
    tensors::Tensor result = tensors::Tensor::zeros(make_result_info("softmax_last_dim_result", input));

    for (std::size_t row = 0; row < row_count; ++row) {
        const std::size_t row_base = row * last_dim;

        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t index = 0; index < last_dim; ++index) {
            row_max =
                std::max(row_max, detail::load_float_value(input.tensor_info().dtype, input.data(), row_base + index));
        }

        float exp_sum = 0.0f;
        for (std::size_t index = 0; index < last_dim; ++index) {
            exp_sum +=
                std::exp(detail::load_float_value(input.tensor_info().dtype, input.data(), row_base + index) - row_max);
        }

        for (std::size_t index = 0; index < last_dim; ++index) {
            const float value =
                std::exp(detail::load_float_value(input.tensor_info().dtype, input.data(), row_base + index) -
                         row_max) /
                exp_sum;
            detail::store_float_value(input.tensor_info().dtype, result.mutable_data(), row_base + index, value);
        }
    }

    return result;
}

tensors::Tensor rms_norm(const tensors::TensorView& input, const tensors::TensorView& weight, float epsilon) {
    validate_last_dim_operation_input(input, "rms_norm");
    if (epsilon < 0.0f) {
        throw std::invalid_argument("rms_norm requires a non-negative epsilon.");
    }
    if (input.tensor_info().dtype != weight.tensor_info().dtype) {
        throw std::invalid_argument("rms_norm requires matching tensor dtypes.");
    }
    if (weight.tensor_info().shape.rank() != 1) {
        throw std::invalid_argument("rms_norm requires a rank-1 weight tensor.");
    }

    const std::size_t last_dim = checked_dim_to_size(input.tensor_info().shape.dims().back(), "rms_norm last dim");
    if (checked_dim_to_size(weight.tensor_info().shape.dims()[0], "rms_norm weight size") != last_dim) {
        throw std::invalid_argument("rms_norm weight size must match the input last dimension.");
    }

    const std::size_t row_count = input.tensor_info().shape.num_elements() / last_dim;
    tensors::Tensor result = tensors::Tensor::zeros(make_result_info("rms_norm_result", input));

    for (std::size_t row = 0; row < row_count; ++row) {
        const std::size_t row_base = row * last_dim;
        float sum_squares = 0.0f;
        for (std::size_t index = 0; index < last_dim; ++index) {
            const float value = detail::load_float_value(input.tensor_info().dtype, input.data(), row_base + index);
            sum_squares += value * value;
        }

        const float scale = 1.0f / std::sqrt(sum_squares / static_cast<float>(last_dim) + epsilon);
        for (std::size_t index = 0; index < last_dim; ++index) {
            const float input_value =
                detail::load_float_value(input.tensor_info().dtype, input.data(), row_base + index);
            const float weight_value = detail::load_float_value(weight.tensor_info().dtype, weight.data(), index);
            detail::store_float_value(input.tensor_info().dtype, result.mutable_data(), row_base + index,
                                      input_value * scale * weight_value);
        }
    }

    return result;
}

} // namespace cppinf::ops
