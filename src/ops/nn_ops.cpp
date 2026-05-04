#include "ops/nn_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "common/checked.h"
#include "ops/one_dnn_utils.h"
#include "ops/op_utils.h"
#include "ops/tensor_ops.h"

namespace cppinf::ops {
namespace {

tensors::Tensor make_f32_tensor(std::string_view name, const tensors::Shape& shape, float value) {
    auto tensor = detail::make_result_tensor(name, tensors::DType::F32, shape);
    for (std::size_t index = 0; index < shape.num_elements(); ++index) {
        std::memcpy(tensor.mutable_data().data() + index * sizeof(float), &value, sizeof(float));
    }
    return tensor;
}

void validate_last_dim_operation_input(const tensors::TensorView& input, std::string_view op_name) {
    detail::validate_supported_float_dtype(input.tensor_info().dtype, op_name);
    if (input.tensor_info().shape.rank() == 0) {
        throw std::invalid_argument(fmt::format("{} requires a tensor with rank at least 1.", op_name));
    }

    if (common::checked_non_negative_dim_to_size(input.tensor_info().shape.dims().back(),
                                                 fmt::format("{} last dim", op_name)) == 0) {
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

    const auto output_dtype = input.tensor_info().dtype;
    const auto compute_dtype = output_dtype;

    std::optional<tensors::Tensor> input_storage;
    const auto input_compute =
        detail::maybe_cast_to_dtype(input, compute_dtype, input_storage, "softmax_last_dim_result");

    auto result =
        detail::make_result_tensor("softmax_last_dim_result", compute_dtype, input_compute.tensor_info().shape);
    const dnnl::memory::desc src_desc = detail::make_desc(input_compute, "softmax_last_dim");
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

tensors::Tensor scaled_causal_softmax_last_dim(const tensors::TensorView& input, float scale) {
    validate_last_dim_operation_input(input, "scaled_causal_softmax_last_dim");
    if (input.tensor_info().shape.rank() < 2) {
        throw std::invalid_argument("scaled_causal_softmax_last_dim requires a tensor with rank at least 2.");
    }

    if (!std::isfinite(scale)) {
        throw std::invalid_argument("scaled_causal_softmax_last_dim requires a finite scale.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto query_length =
        common::checked_non_negative_dim_to_size(dims[dims.size() - 2], "scaled_causal_softmax_last_dim query length");
    const auto key_length =
        common::checked_non_negative_dim_to_size(dims[dims.size() - 1], "scaled_causal_softmax_last_dim key length");
    if (query_length == 0 || key_length == 0) {
        throw std::invalid_argument("scaled_causal_softmax_last_dim requires non-empty attention dimensions.");
    }

    if (query_length != key_length) {
        throw std::invalid_argument("scaled_causal_softmax_last_dim requires square attention score matrices.");
    }

    std::optional<tensors::Tensor> input_storage;
    const auto input_f32 =
        detail::maybe_cast_to_dtype(input, tensors::DType::F32, input_storage, "scaled_causal_softmax_last_dim_result");

    auto result = detail::make_result_tensor("scaled_causal_softmax_last_dim_result", tensors::DType::F32,
                                             input.tensor_info().shape);
    const auto matrix_size = query_length * key_length;
    const auto matrix_count = input.tensor_info().shape.num_elements() / matrix_size;

    for (std::size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        const auto matrix_offset = matrix_index * matrix_size;
        for (std::size_t query_index = 0; query_index < query_length; ++query_index) {
            const auto row_offset = matrix_offset + query_index * key_length;
            auto max_value = -std::numeric_limits<float>::infinity();
            for (std::size_t key_index = 0; key_index <= query_index; ++key_index) {
                const auto value = detail::load_float_value(input_f32, row_offset + key_index) * scale;
                max_value = std::max(max_value, value);
            }

            auto sum = 0.0f;
            for (std::size_t key_index = 0; key_index < key_length; ++key_index) {
                const auto value =
                    key_index <= query_index
                        ? std::exp(detail::load_float_value(input_f32, row_offset + key_index) * scale - max_value)
                        : 0.0f;
                detail::store_float_value(tensors::DType::F32, result.mutable_data(), row_offset + key_index, value);
                sum += value;
            }

            for (std::size_t key_index = 0; key_index < key_length; ++key_index) {
                const auto flat_index = row_offset + key_index;
                const auto value = detail::load_float_value(tensors::DType::F32, result.data(), flat_index) / sum;
                detail::store_float_value(tensors::DType::F32, result.mutable_data(), flat_index, value);
            }
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

    const auto last_dim =
        common::checked_non_negative_dim_to_size(input.tensor_info().shape.dims().back(), "rms_norm last dim");
    if (common::checked_non_negative_dim_to_size(weight.tensor_info().shape.dims()[0], "rms_norm weight size") !=
        last_dim) {
        throw std::invalid_argument("rms_norm weight size must match the input last dimension.");
    }

    std::optional<tensors::Tensor> input_storage;
    std::optional<tensors::Tensor> weight_storage;
    const auto input_f32 = detail::maybe_cast_to_dtype(input, tensors::DType::F32, input_storage, "rms_norm_result");
    const auto weight_f32 = detail::maybe_cast_to_dtype(weight, tensors::DType::F32, weight_storage, "rms_norm_result");

    std::vector<std::int64_t> reduced_dims = input_f32.tensor_info().shape.dims();
    reduced_dims.back() = 1;
    const auto reduced_shape = tensors::Shape(std::move(reduced_dims));

    auto squared = detail::binary_with_one_dnn("rms_norm_squared", input_f32, input_f32, dnnl::algorithm::binary_mul,
                                               tensors::DType::F32);
    auto mean_squares = detail::make_result_tensor("rms_norm_mean", tensors::DType::F32, reduced_shape);
    const dnnl::memory::desc squared_desc = detail::make_desc(squared.view(), "rms_norm_mean");
    const dnnl::memory::desc mean_desc = detail::make_dense_desc(reduced_shape, tensors::DType::F32, "rms_norm_mean");
    const dnnl::reduction::primitive_desc mean_primitive_desc(detail::cpu_engine(), dnnl::algorithm::reduction_mean,
                                                              squared_desc, mean_desc, 0.0f, 0.0f);
    const dnnl::reduction mean_primitive(mean_primitive_desc);
    dnnl::memory squared_memory = detail::make_memory(squared_desc, squared.data());
    dnnl::memory mean_memory = detail::make_memory(mean_desc, mean_squares.mutable_data());
    dnnl::stream stream(detail::cpu_engine());
    mean_primitive.execute(stream, {{DNNL_ARG_SRC, squared_memory}, {DNNL_ARG_DST, mean_memory}});
    stream.wait();

    auto epsilon_tensor = make_f32_tensor("rms_norm_epsilon", reduced_shape, epsilon);
    auto variance = detail::binary_with_one_dnn("rms_norm_variance", mean_squares.view(), epsilon_tensor.view(),
                                                dnnl::algorithm::binary_add, tensors::DType::F32);
    auto rms =
        detail::unary_with_one_dnn("rms_norm_rms", variance.view(), dnnl::algorithm::eltwise_sqrt, tensors::DType::F32);
    auto ones_tensor = make_f32_tensor("rms_norm_one", reduced_shape, 1.0f);
    auto inv_rms = detail::binary_with_one_dnn("rms_norm_inv_rms", ones_tensor.view(), rms.view(),
                                               dnnl::algorithm::binary_div, tensors::DType::F32);
    auto normalized = detail::binary_with_one_dnn("rms_norm_normalized", input_f32, inv_rms.view(),
                                                  dnnl::algorithm::binary_mul, tensors::DType::F32);

    std::vector<std::int64_t> weight_dims(input_f32.tensor_info().shape.rank(), 1);
    weight_dims.back() = weight_f32.tensor_info().shape.dims()[0];
    const auto weight_broadcast = reshape(weight_f32, tensors::Shape(std::move(weight_dims)));
    auto result = detail::binary_with_one_dnn("rms_norm_result", normalized.view(), weight_broadcast,
                                              dnnl::algorithm::binary_mul, tensors::DType::F32);
    return detail::maybe_cast_result(std::move(result), input.tensor_info().dtype, "rms_norm_result");
}

} // namespace cppinf::ops
