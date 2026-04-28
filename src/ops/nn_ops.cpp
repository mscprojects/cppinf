#include "ops/nn_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

#include "ops/one_dnn_utils.h"
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
    return detail::one_dnn_silu(input);
}

tensors::Tensor softmax_last_dim(const tensors::TensorView& input) {
    validate_last_dim_operation_input(input, "softmax_last_dim");
    return detail::one_dnn_softmax_last_dim(input);
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
    return detail::one_dnn_rms_norm(input, weight, epsilon);
}

} // namespace cppinf::ops
