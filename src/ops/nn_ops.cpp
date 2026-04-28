#include "ops/nn_ops.h"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "ops/one_dnn_utils.h"
#include "ops/op_utils.h"
#include "ops/tensor_ops.h"

namespace cppinf::ops {
namespace {

tensors::Tensor make_f32_tensor(std::string_view name, const tensors::Shape& shape, float value) {
    tensors::Tensor tensor = detail::make_result_tensor(name, tensors::DType::F32, shape);
    for (std::size_t index = 0; index < shape.num_elements(); ++index) {
        std::memcpy(tensor.mutable_data().data() + index * sizeof(float), &value, sizeof(float));
    }
    return tensor;
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
    return detail::unary_with_one_dnn("silu_result", input, dnnl::algorithm::eltwise_swish, input.tensor_info().dtype,
                                      1.0f);
}

tensors::Tensor softmax_last_dim(const tensors::TensorView& input) {
    validate_last_dim_operation_input(input, "softmax_last_dim");

    const tensors::DType output_dtype = input.tensor_info().dtype;
    const tensors::DType compute_dtype = output_dtype == tensors::DType::BF16 ? tensors::DType::F32 : output_dtype;

    std::optional<tensors::Tensor> input_storage;
    const tensors::TensorView input_compute =
        detail::maybe_cast_to_dtype(input, compute_dtype, input_storage, "softmax_last_dim_result");

    tensors::Tensor result =
        detail::make_result_tensor("softmax_last_dim_result", compute_dtype, input_compute.tensor_info().shape);
    const dnnl::memory::desc src_desc =
        detail::make_dense_desc(input_compute.tensor_info().shape, compute_dtype, "softmax_last_dim");
    const dnnl::memory::desc dst_desc =
        detail::make_dense_desc(result.tensor_info().shape, compute_dtype, "softmax_last_dim");
    const dnnl::softmax_forward::primitive_desc primitive_desc(
        detail::cpu_engine(), dnnl::prop_kind::forward_inference, dnnl::algorithm::softmax_accurate, src_desc, dst_desc,
        static_cast<int>(input_compute.tensor_info().shape.rank() - 1));
    const dnnl::softmax_forward primitive(primitive_desc);
    dnnl::memory src_memory = detail::make_memory(src_desc, input_compute.data());
    dnnl::memory dst_memory = detail::make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(detail::cpu_engine());
    primitive.execute(stream, {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
    stream.wait();
    return detail::maybe_cast_result(std::move(result), output_dtype, "softmax_last_dim_result");
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

    std::optional<tensors::Tensor> input_storage;
    std::optional<tensors::Tensor> weight_storage;
    const tensors::TensorView input_f32 =
        detail::maybe_cast_to_dtype(input, tensors::DType::F32, input_storage, "rms_norm_result");
    const tensors::TensorView weight_f32 =
        detail::maybe_cast_to_dtype(weight, tensors::DType::F32, weight_storage, "rms_norm_result");

    std::vector<std::int64_t> reduced_dims = input_f32.tensor_info().shape.dims();
    reduced_dims.back() = 1;
    const tensors::Shape reduced_shape(std::move(reduced_dims));

    tensors::Tensor squared = detail::binary_with_one_dnn("rms_norm_squared", input_f32, input_f32,
                                                          dnnl::algorithm::binary_mul, tensors::DType::F32);
    tensors::Tensor mean_squares = detail::make_result_tensor("rms_norm_mean", tensors::DType::F32, reduced_shape);
    const dnnl::memory::desc squared_desc =
        detail::make_dense_desc(squared.tensor_info().shape, tensors::DType::F32, "rms_norm_mean");
    const dnnl::memory::desc mean_desc = detail::make_dense_desc(reduced_shape, tensors::DType::F32, "rms_norm_mean");
    const dnnl::reduction::primitive_desc mean_primitive_desc(
        detail::cpu_engine(), dnnl::algorithm::reduction_mean, squared_desc, mean_desc, 0.0f, 0.0f);
    const dnnl::reduction mean_primitive(mean_primitive_desc);
    dnnl::memory squared_memory = detail::make_memory(squared_desc, squared.data());
    dnnl::memory mean_memory = detail::make_memory(mean_desc, mean_squares.mutable_data());
    dnnl::stream stream(detail::cpu_engine());
    mean_primitive.execute(stream, {{DNNL_ARG_SRC, squared_memory}, {DNNL_ARG_DST, mean_memory}});
    stream.wait();

    tensors::Tensor epsilon_tensor = make_f32_tensor("rms_norm_epsilon", reduced_shape, epsilon);
    tensors::Tensor variance =
        detail::binary_with_one_dnn("rms_norm_variance", mean_squares.view(), epsilon_tensor.view(),
                                    dnnl::algorithm::binary_add, tensors::DType::F32);
    tensors::Tensor rms =
        detail::unary_with_one_dnn("rms_norm_rms", variance.view(), dnnl::algorithm::eltwise_sqrt, tensors::DType::F32);
    tensors::Tensor ones_tensor = make_f32_tensor("rms_norm_one", reduced_shape, 1.0f);
    tensors::Tensor inv_rms =
        detail::binary_with_one_dnn("rms_norm_inv_rms", ones_tensor.view(), rms.view(), dnnl::algorithm::binary_div,
                                    tensors::DType::F32);
    tensors::Tensor normalized = detail::binary_with_one_dnn("rms_norm_normalized", input_f32, inv_rms.view(),
                                                             dnnl::algorithm::binary_mul, tensors::DType::F32);

    std::vector<std::int64_t> weight_dims(input_f32.tensor_info().shape.rank(), 1);
    weight_dims.back() = weight_f32.tensor_info().shape.dims()[0];
    const tensors::TensorView weight_broadcast = reshape(weight_f32, tensors::Shape(std::move(weight_dims)));
    tensors::Tensor result = detail::binary_with_one_dnn("rms_norm_result", normalized.view(), weight_broadcast,
                                                         dnnl::algorithm::binary_mul, tensors::DType::F32);
    return detail::maybe_cast_result(std::move(result), input.tensor_info().dtype, "rms_norm_result");
}

} // namespace cppinf::ops
