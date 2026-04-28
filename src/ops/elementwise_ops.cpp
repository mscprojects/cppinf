#include "ops/elementwise_ops.h"

#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

#include "ops/one_dnn_utils.h"
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
    return detail::one_dnn_add(lhs, rhs);
}

tensors::Tensor mul(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    validate_elementwise_inputs(lhs, rhs, "mul");
    return detail::one_dnn_mul(lhs, rhs);
}

} // namespace cppinf::ops
