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

} // namespace

tensors::Tensor add(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    validate_elementwise_inputs(lhs, rhs, "add");
    return detail::binary_with_one_dnn("add_result", lhs, rhs, dnnl::algorithm::binary_add, lhs.tensor_info().dtype);
}

tensors::Tensor mul(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    validate_elementwise_inputs(lhs, rhs, "mul");
    return detail::binary_with_one_dnn("mul_result", lhs, rhs, dnnl::algorithm::binary_mul, lhs.tensor_info().dtype);
}

} // namespace cppinf::ops
