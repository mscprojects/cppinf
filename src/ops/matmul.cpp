#include "ops/matmul.h"

#include <stdexcept>

#include "ops/one_dnn_utils.h"
#include "ops/op_utils.h"

namespace cppinf::ops {
namespace {

void validate_matmul_inputs(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    if (lhs.tensor_info().dtype != rhs.tensor_info().dtype) {
        throw std::invalid_argument("matmul requires matching tensor dtypes.");
    }
    detail::validate_supported_float_dtype(lhs.tensor_info().dtype, "matmul");
    if (lhs.tensor_info().shape.rank() != 2 || rhs.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("matmul requires rank-2 tensors.");
    }

    const auto& lhs_dims = lhs.tensor_info().shape.dims();
    const auto& rhs_dims = rhs.tensor_info().shape.dims();
    if (lhs_dims[1] != rhs_dims[0]) {
        throw std::invalid_argument("matmul inner dimensions must match.");
    }
}

} // namespace

tensors::Tensor matmul(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    validate_matmul_inputs(lhs, rhs);
    return detail::one_dnn_matmul(lhs, rhs);
}

} // namespace cppinf::ops
