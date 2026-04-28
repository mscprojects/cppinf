#include "ops/elementwise_ops.h"

#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

#include "ops/op_utils.h"

namespace cppinf::ops {
namespace {

void validate_elementwise_inputs(const tensors::TensorView& lhs, const tensors::TensorView& rhs,
                                 std::string_view op_name) {
    if (lhs.tensor_info().dtype != rhs.tensor_info().dtype) {
        throw std::invalid_argument(fmt::format("{} requires matching tensor dtypes.", op_name));
    }
    if (lhs.tensor_info().shape != rhs.tensor_info().shape) {
        throw std::invalid_argument(fmt::format("{} requires matching tensor shapes.", op_name));
    }
    detail::validate_supported_float_dtype(lhs.tensor_info().dtype, op_name);
}

tensors::TensorInfo make_result_info(std::string_view name, const tensors::TensorView& input) {
    return tensors::TensorInfo{
        .name = std::string(name),
        .dtype = input.tensor_info().dtype,
        .shape = input.tensor_info().shape,
        .byte_offset = 0,
    };
}

} // namespace

tensors::Tensor add(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    validate_elementwise_inputs(lhs, rhs, "add");

    tensors::Tensor result = tensors::Tensor::zeros(make_result_info("add_result", lhs));
    for (std::size_t index = 0; index < lhs.tensor_info().shape.num_elements(); ++index) {
        detail::store_float_value(lhs.tensor_info().dtype, result.mutable_data(), index,
                                  detail::load_float_value(lhs.tensor_info().dtype, lhs.data(), index) +
                                      detail::load_float_value(rhs.tensor_info().dtype, rhs.data(), index));
    }

    return result;
}

tensors::Tensor mul(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    validate_elementwise_inputs(lhs, rhs, "mul");

    tensors::Tensor result = tensors::Tensor::zeros(make_result_info("mul_result", lhs));
    for (std::size_t index = 0; index < lhs.tensor_info().shape.num_elements(); ++index) {
        detail::store_float_value(lhs.tensor_info().dtype, result.mutable_data(), index,
                                  detail::load_float_value(lhs.tensor_info().dtype, lhs.data(), index) *
                                      detail::load_float_value(rhs.tensor_info().dtype, rhs.data(), index));
    }

    return result;
}

} // namespace cppinf::ops
