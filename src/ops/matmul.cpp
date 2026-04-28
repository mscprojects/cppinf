#include "ops/matmul.h"

#include <stdexcept>

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

    const auto& lhs_dims = lhs.tensor_info().shape.dims();
    const auto& rhs_dims = rhs.tensor_info().shape.dims();
    const std::size_t rows = static_cast<std::size_t>(lhs_dims[0]);
    const std::size_t inner = static_cast<std::size_t>(lhs_dims[1]);
    const std::size_t cols = static_cast<std::size_t>(rhs_dims[1]);
    const tensors::DType result_dtype = lhs.tensor_info().dtype;

    tensors::Tensor result = tensors::Tensor::zeros(tensors::TensorInfo{
        .name = "matmul_result",
        .dtype = result_dtype,
        .shape = tensors::Shape({static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)}),
        .byte_offset = 0,
    });

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            float sum = 0.0f;
            for (std::size_t index = 0; index < inner; ++index) {
                const float lhs_value =
                    detail::load_float_value(lhs.tensor_info().dtype, lhs.data(), row * inner + index);
                const float rhs_value =
                    detail::load_float_value(rhs.tensor_info().dtype, rhs.data(), index * cols + col);
                sum += lhs_value * rhs_value;
            }

            detail::store_float_value(result_dtype, result.mutable_data(), row * cols + col, sum);
        }
    }

    return result;
}

} // namespace cppinf::ops
