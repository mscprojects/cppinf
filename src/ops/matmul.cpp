#include "ops/matmul.h"

#include <optional>
#include <stdexcept>
#include <vector>

#include "ops/one_dnn_utils.h"
#include "ops/op_utils.h"

namespace cppinf::ops {
namespace {

tensors::DType resolve_output_dtype(const tensors::TensorView& lhs, const tensors::TensorView& rhs,
                                    MatmulOptions options) {
    if (options.output_dtype.has_value()) {
        return *options.output_dtype;
    }

    if (lhs.tensor_info().dtype == rhs.tensor_info().dtype) {
        return lhs.tensor_info().dtype;
    }

    return tensors::DType::F32;
}

void validate_matmul_inputs(const tensors::TensorView& lhs, const tensors::TensorView& rhs,
                            tensors::DType output_dtype) {
    detail::validate_supported_float_dtype(lhs.tensor_info().dtype, "matmul");
    detail::validate_supported_float_dtype(rhs.tensor_info().dtype, "matmul");
    detail::validate_supported_float_dtype(output_dtype, "matmul");

    const bool inputs_match = lhs.tensor_info().dtype == rhs.tensor_info().dtype;
    if (inputs_match) {
        if (output_dtype != lhs.tensor_info().dtype &&
            (lhs.tensor_info().dtype != tensors::DType::BF16 || output_dtype != tensors::DType::F32)) {
            throw std::invalid_argument(
                "matmul output dtype must match the input dtype, except BF16 inputs may request an F32 result.");
        }
    } else if (output_dtype != tensors::DType::F32) {
        throw std::invalid_argument("matmul requires an F32 result when input dtypes differ.");
    }

    if (lhs.tensor_info().shape.rank() != rhs.tensor_info().shape.rank()) {
        throw std::invalid_argument("matmul requires tensors with matching rank.");
    }

    if (lhs.tensor_info().shape.rank() != 2 && lhs.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("matmul requires rank-2 or rank-3 tensors.");
    }

    const auto& lhs_dims = lhs.tensor_info().shape.dims();
    const auto& rhs_dims = rhs.tensor_info().shape.dims();
    if (lhs.tensor_info().shape.rank() == 3 && lhs_dims[0] != rhs_dims[0]) {
        throw std::invalid_argument("matmul requires matching batch dimensions.");
    }

    const auto lhs_inner = lhs_dims.back();
    const auto rhs_inner = rhs_dims[rhs_dims.size() - 2];
    if (lhs_inner != rhs_inner) {
        throw std::invalid_argument("matmul inner dimensions must match.");
    }
}

tensors::Shape make_matmul_result_shape(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    const auto& lhs_dims = lhs.tensor_info().shape.dims();
    const auto& rhs_dims = rhs.tensor_info().shape.dims();
    if (lhs.tensor_info().shape.rank() == 2) {
        return tensors::Shape({lhs_dims[0], rhs_dims[1]});
    }

    return tensors::Shape({lhs_dims[0], lhs_dims[1], rhs_dims[2]});
}

bool supports_native_matmul_input_dtypes(tensors::DType lhs_dtype, tensors::DType rhs_dtype,
                                         tensors::DType output_dtype) {
    if (lhs_dtype == tensors::DType::BF16 && rhs_dtype == tensors::DType::BF16) {
        return output_dtype == tensors::DType::BF16 || output_dtype == tensors::DType::F32;
    }

    return lhs_dtype == tensors::DType::F32 && rhs_dtype == tensors::DType::BF16 && output_dtype == tensors::DType::F32;
}

} // namespace

tensors::Tensor matmul(const tensors::TensorView& lhs, const tensors::TensorView& rhs, MatmulOptions options) {
    const auto output_dtype = resolve_output_dtype(lhs, rhs, options);
    validate_matmul_inputs(lhs, rhs, output_dtype);

    const auto compute_dtype = output_dtype;
    const bool can_keep_input_dtypes =
        supports_native_matmul_input_dtypes(lhs.tensor_info().dtype, rhs.tensor_info().dtype, output_dtype);

    std::optional<tensors::Tensor> lhs_storage;
    std::optional<tensors::Tensor> rhs_storage;
    const auto lhs_compute =
        can_keep_input_dtypes ? lhs : detail::maybe_cast_to_dtype(lhs, compute_dtype, lhs_storage, "matmul_result");
    const auto rhs_compute =
        can_keep_input_dtypes ? rhs : detail::maybe_cast_to_dtype(rhs, compute_dtype, rhs_storage, "matmul_result");

    auto result = detail::make_result_tensor("matmul_result", compute_dtype, make_matmul_result_shape(lhs, rhs));

    const auto src_desc =
        detail::make_dense_desc(lhs_compute.tensor_info().shape, lhs_compute.tensor_info().dtype, "matmul");
    const auto weights_desc =
        detail::make_dense_desc(rhs_compute.tensor_info().shape, rhs_compute.tensor_info().dtype, "matmul");
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
