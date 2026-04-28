#include "ops/matmul.h"

#include <optional>
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

    const auto output_dtype = lhs.tensor_info().dtype;
    const auto compute_dtype = output_dtype == tensors::DType::BF16 ? tensors::DType::F32 : output_dtype;

    std::optional<tensors::Tensor> lhs_storage;
    std::optional<tensors::Tensor> rhs_storage;
    const auto lhs_compute = detail::maybe_cast_to_dtype(lhs, compute_dtype, lhs_storage, "matmul_result");
    const auto rhs_compute = detail::maybe_cast_to_dtype(rhs, compute_dtype, rhs_storage, "matmul_result");

    const auto& lhs_dims = lhs_compute.tensor_info().shape.dims();
    const auto& rhs_dims = rhs_compute.tensor_info().shape.dims();
    auto result =
        detail::make_result_tensor("matmul_result", compute_dtype, tensors::Shape({lhs_dims[0], rhs_dims[1]}));

    const auto src_desc = detail::make_dense_desc(lhs_compute.tensor_info().shape, compute_dtype, "matmul");
    const auto weights_desc = detail::make_dense_desc(rhs_compute.tensor_info().shape, compute_dtype, "matmul");
    const auto dst_desc = detail::make_dense_desc(result.tensor_info().shape, compute_dtype, "matmul");
    const auto primitive_desc = dnnl::matmul::primitive_desc(detail::cpu_engine(), src_desc, weights_desc, dst_desc);
    const auto primitive = dnnl::matmul(primitive_desc);
    auto src_memory = detail::make_memory(src_desc, lhs_compute.data());
    auto weights_memory = detail::make_memory(weights_desc, rhs_compute.data());
    auto dst_memory = detail::make_memory(dst_desc, result.mutable_data());
    auto stream = dnnl::stream(detail::cpu_engine());
    primitive.execute(stream,
                      {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_WEIGHTS, weights_memory}, {DNNL_ARG_DST, dst_memory}});
    stream.wait();
    return detail::maybe_cast_result(std::move(result), output_dtype, "matmul_result");
}

} // namespace cppinf::ops
