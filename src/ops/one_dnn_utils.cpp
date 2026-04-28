#include "ops/one_dnn_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <oneapi/dnnl/dnnl.hpp>

#include "ops/tensor_ops.h"

namespace cppinf::ops::detail {
namespace {

const dnnl::engine& cpu_engine() {
    static const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    return engine;
}

tensors::TensorInfo make_result_info(std::string_view name, tensors::DType dtype, const tensors::Shape& shape) {
    return tensors::TensorInfo{
        .name = std::string(name),
        .dtype = dtype,
        .shape = shape,
        .byte_offset = 0,
    };
}

tensors::Tensor make_result_tensor(std::string_view name, tensors::DType dtype, const tensors::Shape& shape) {
    return tensors::Tensor::zeros(make_result_info(name, dtype, shape));
}

tensors::Tensor make_f32_tensor(std::string_view name, const tensors::Shape& shape, float value) {
    tensors::Tensor tensor = make_result_tensor(name, tensors::DType::F32, shape);
    for (std::size_t index = 0; index < shape.num_elements(); ++index) {
        std::memcpy(tensor.mutable_data().data() + index * sizeof(float), &value, sizeof(float));
    }
    return tensor;
}

dnnl::memory::data_type to_dnnl_dtype(tensors::DType dtype) {
    switch (dtype) {
    case tensors::DType::F16:
        return dnnl::memory::data_type::f16;
    case tensors::DType::BF16:
        return dnnl::memory::data_type::bf16;
    case tensors::DType::F32:
        return dnnl::memory::data_type::f32;
    case tensors::DType::I32:
        return dnnl::memory::data_type::s32;
    case tensors::DType::I64:
        break;
    case tensors::DType::U8:
        return dnnl::memory::data_type::u8;
    }

    throw std::invalid_argument("Unsupported dtype for oneDNN.");
}

dnnl::memory::dims to_dnnl_dims(const tensors::Shape& shape, std::string_view op_name) {
    dnnl::memory::dims dims;
    dims.reserve(shape.rank());
    for (const std::int64_t dim : shape.dims()) {
        if (dim < 0) {
            throw std::invalid_argument(fmt::format("{} requires non-negative tensor dimensions.", op_name));
        }
        dims.push_back(static_cast<dnnl::memory::dim>(dim));
    }
    return dims;
}

dnnl::memory::dims make_dense_strides(const dnnl::memory::dims& dims) {
    dnnl::memory::dims strides(dims.size());
    dnnl::memory::dim stride = 1;
    for (std::size_t index = dims.size(); index-- > 0;) {
        strides[index] = stride;
        stride *= std::max<dnnl::memory::dim>(dims[index], 1);
    }
    return strides;
}

dnnl::memory::desc make_dense_desc(const tensors::Shape& shape, tensors::DType dtype, std::string_view op_name) {
    const dnnl::memory::dims dims = to_dnnl_dims(shape, op_name);
    return dnnl::memory::desc(dims, to_dnnl_dtype(dtype), make_dense_strides(dims));
}

dnnl::memory::desc make_desc(const dnnl::memory::dims& dims, const dnnl::memory::dims& strides, tensors::DType dtype) {
    return dnnl::memory::desc(dims, to_dnnl_dtype(dtype), strides);
}

dnnl::memory make_memory(const dnnl::memory::desc& desc, std::span<const std::byte> bytes) {
    return dnnl::memory(desc, cpu_engine(), const_cast<std::byte*>(bytes.data()));
}

dnnl::memory make_memory(const dnnl::memory::desc& desc, std::span<std::byte> bytes) {
    return dnnl::memory(desc, cpu_engine(), bytes.data());
}

tensors::Tensor one_dnn_cast_impl(const tensors::TensorView& input, tensors::DType dtype,
                                  std::string_view result_name) {
    tensors::Tensor result = make_result_tensor(result_name, dtype, input.tensor_info().shape);
    const dnnl::memory::desc src_desc = make_dense_desc(input.tensor_info().shape, input.tensor_info().dtype, "cast");
    const dnnl::memory::desc dst_desc = make_dense_desc(result.tensor_info().shape, dtype, "cast");
    dnnl::memory src_memory = make_memory(src_desc, input.data());
    dnnl::memory dst_memory = make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(cpu_engine());
    dnnl::reorder(src_memory, dst_memory).execute(stream, src_memory, dst_memory);
    stream.wait();
    return result;
}

tensors::TensorView maybe_cast_to_dtype(const tensors::TensorView& input, tensors::DType dtype,
                                        std::optional<tensors::Tensor>& storage, std::string_view result_name) {
    if (input.tensor_info().dtype == dtype) {
        return input;
    }

    storage.emplace(one_dnn_cast_impl(input, dtype, result_name));
    return storage->view();
}

tensors::Tensor maybe_cast_result(tensors::Tensor tensor, tensors::DType dtype, std::string_view result_name) {
    if (tensor.tensor_info().dtype == dtype) {
        return tensor;
    }

    return one_dnn_cast_impl(tensor.view(), dtype, result_name);
}

tensors::Tensor one_dnn_binary_impl(std::string_view result_name, const tensors::TensorView& lhs,
                                    const tensors::TensorView& rhs, dnnl::algorithm algorithm,
                                    tensors::DType output_dtype) {
    const tensors::DType compute_dtype = output_dtype == tensors::DType::BF16 ? tensors::DType::F32 : output_dtype;

    std::optional<tensors::Tensor> lhs_storage;
    std::optional<tensors::Tensor> rhs_storage;
    const tensors::TensorView lhs_compute = maybe_cast_to_dtype(lhs, compute_dtype, lhs_storage, result_name);
    const tensors::TensorView rhs_compute = maybe_cast_to_dtype(rhs, compute_dtype, rhs_storage, result_name);

    tensors::Tensor result = make_result_tensor(result_name, compute_dtype, lhs_compute.tensor_info().shape);
    const dnnl::memory::desc src0_desc = make_dense_desc(lhs_compute.tensor_info().shape, compute_dtype, result_name);
    const dnnl::memory::desc src1_desc = make_dense_desc(rhs_compute.tensor_info().shape, compute_dtype, result_name);
    const dnnl::memory::desc dst_desc = make_dense_desc(result.tensor_info().shape, compute_dtype, result_name);
    const dnnl::binary::primitive_desc primitive_desc(cpu_engine(), algorithm, src0_desc, src1_desc, dst_desc);
    const dnnl::binary primitive(primitive_desc);
    dnnl::memory src0_memory = make_memory(src0_desc, lhs_compute.data());
    dnnl::memory src1_memory = make_memory(src1_desc, rhs_compute.data());
    dnnl::memory dst_memory = make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(cpu_engine());
    primitive.execute(stream,
                      {{DNNL_ARG_SRC_0, src0_memory}, {DNNL_ARG_SRC_1, src1_memory}, {DNNL_ARG_DST, dst_memory}});
    stream.wait();
    return maybe_cast_result(std::move(result), output_dtype, result_name);
}

tensors::Tensor one_dnn_unary_impl(std::string_view result_name, const tensors::TensorView& input,
                                   dnnl::algorithm algorithm, tensors::DType output_dtype, float alpha = 0.0f,
                                   float beta = 0.0f) {
    const tensors::DType compute_dtype = output_dtype == tensors::DType::BF16 ? tensors::DType::F32 : output_dtype;

    std::optional<tensors::Tensor> input_storage;
    const tensors::TensorView input_compute = maybe_cast_to_dtype(input, compute_dtype, input_storage, result_name);

    tensors::Tensor result = make_result_tensor(result_name, compute_dtype, input_compute.tensor_info().shape);
    const dnnl::memory::desc src_desc = make_dense_desc(input_compute.tensor_info().shape, compute_dtype, result_name);
    const dnnl::memory::desc dst_desc = make_dense_desc(result.tensor_info().shape, compute_dtype, result_name);
    const dnnl::eltwise_forward::primitive_desc primitive_desc(cpu_engine(), dnnl::prop_kind::forward_inference,
                                                               algorithm, src_desc, dst_desc, alpha, beta);
    const dnnl::eltwise_forward primitive(primitive_desc);
    dnnl::memory src_memory = make_memory(src_desc, input_compute.data());
    dnnl::memory dst_memory = make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(cpu_engine());
    primitive.execute(stream, {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
    stream.wait();
    return maybe_cast_result(std::move(result), output_dtype, result_name);
}

tensors::Tensor one_dnn_reduction_impl(std::string_view result_name, const tensors::TensorView& input,
                                       const tensors::Shape& output_shape, dnnl::algorithm algorithm, float p = 0.0f,
                                       float epsilon = 0.0f) {
    tensors::Tensor result = make_result_tensor(result_name, input.tensor_info().dtype, output_shape);
    const dnnl::memory::desc src_desc =
        make_dense_desc(input.tensor_info().shape, input.tensor_info().dtype, result_name);
    const dnnl::memory::desc dst_desc = make_dense_desc(output_shape, result.tensor_info().dtype, result_name);
    const dnnl::reduction::primitive_desc primitive_desc(cpu_engine(), algorithm, src_desc, dst_desc, p, epsilon);
    const dnnl::reduction primitive(primitive_desc);
    dnnl::memory src_memory = make_memory(src_desc, input.data());
    dnnl::memory dst_memory = make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(cpu_engine());
    primitive.execute(stream, {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
    stream.wait();
    return result;
}

} // namespace

tensors::Tensor one_dnn_add(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    return one_dnn_binary_impl("add_result", lhs, rhs, dnnl::algorithm::binary_add, lhs.tensor_info().dtype);
}

tensors::Tensor one_dnn_mul(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    return one_dnn_binary_impl("mul_result", lhs, rhs, dnnl::algorithm::binary_mul, lhs.tensor_info().dtype);
}

tensors::Tensor one_dnn_matmul(const tensors::TensorView& lhs, const tensors::TensorView& rhs) {
    const tensors::DType output_dtype = lhs.tensor_info().dtype;
    const tensors::DType compute_dtype = output_dtype == tensors::DType::BF16 ? tensors::DType::F32 : output_dtype;

    std::optional<tensors::Tensor> lhs_storage;
    std::optional<tensors::Tensor> rhs_storage;
    const tensors::TensorView lhs_compute = maybe_cast_to_dtype(lhs, compute_dtype, lhs_storage, "matmul_result");
    const tensors::TensorView rhs_compute = maybe_cast_to_dtype(rhs, compute_dtype, rhs_storage, "matmul_result");

    const auto& lhs_dims = lhs_compute.tensor_info().shape.dims();
    const auto& rhs_dims = rhs_compute.tensor_info().shape.dims();
    tensors::Tensor result =
        make_result_tensor("matmul_result", compute_dtype, tensors::Shape({lhs_dims[0], rhs_dims[1]}));

    const dnnl::memory::desc src_desc = make_dense_desc(lhs_compute.tensor_info().shape, compute_dtype, "matmul");
    const dnnl::memory::desc weights_desc = make_dense_desc(rhs_compute.tensor_info().shape, compute_dtype, "matmul");
    const dnnl::memory::desc dst_desc = make_dense_desc(result.tensor_info().shape, compute_dtype, "matmul");
    const dnnl::matmul::primitive_desc primitive_desc(cpu_engine(), src_desc, weights_desc, dst_desc);
    const dnnl::matmul primitive(primitive_desc);
    dnnl::memory src_memory = make_memory(src_desc, lhs_compute.data());
    dnnl::memory weights_memory = make_memory(weights_desc, rhs_compute.data());
    dnnl::memory dst_memory = make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(cpu_engine());
    primitive.execute(stream,
                      {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_WEIGHTS, weights_memory}, {DNNL_ARG_DST, dst_memory}});
    stream.wait();
    return maybe_cast_result(std::move(result), output_dtype, "matmul_result");
}

tensors::Tensor one_dnn_cast(const tensors::TensorView& input, tensors::DType dtype) {
    return one_dnn_cast_impl(input, dtype, "cast_result");
}

tensors::Tensor one_dnn_transpose_2d(const tensors::TensorView& input) {
    const auto& dims = input.tensor_info().shape.dims();
    const dnnl::memory::dims transposed_dims = {
        static_cast<dnnl::memory::dim>(dims[1]),
        static_cast<dnnl::memory::dim>(dims[0]),
    };
    const dnnl::memory::dims src_strides = {1, transposed_dims[0]};
    const dnnl::memory::dims dst_strides = {transposed_dims[1], 1};

    tensors::Tensor result =
        make_result_tensor("transpose_2d_result", input.tensor_info().dtype, tensors::Shape({dims[1], dims[0]}));
    const dnnl::memory::desc src_desc = make_desc(transposed_dims, src_strides, input.tensor_info().dtype);
    const dnnl::memory::desc dst_desc = make_desc(transposed_dims, dst_strides, input.tensor_info().dtype);
    dnnl::memory src_memory = make_memory(src_desc, input.data());
    dnnl::memory dst_memory = make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(cpu_engine());
    dnnl::reorder(src_memory, dst_memory).execute(stream, src_memory, dst_memory);
    stream.wait();
    return result;
}

tensors::Tensor one_dnn_silu(const tensors::TensorView& input) {
    return one_dnn_unary_impl("silu_result", input, dnnl::algorithm::eltwise_swish, input.tensor_info().dtype, 1.0f);
}

tensors::Tensor one_dnn_softmax_last_dim(const tensors::TensorView& input) {
    const tensors::DType output_dtype = input.tensor_info().dtype;
    const tensors::DType compute_dtype = output_dtype == tensors::DType::BF16 ? tensors::DType::F32 : output_dtype;

    std::optional<tensors::Tensor> input_storage;
    const tensors::TensorView input_compute =
        maybe_cast_to_dtype(input, compute_dtype, input_storage, "softmax_last_dim_result");

    tensors::Tensor result =
        make_result_tensor("softmax_last_dim_result", compute_dtype, input_compute.tensor_info().shape);
    const dnnl::memory::desc src_desc =
        make_dense_desc(input_compute.tensor_info().shape, compute_dtype, "softmax_last_dim");
    const dnnl::memory::desc dst_desc = make_dense_desc(result.tensor_info().shape, compute_dtype, "softmax_last_dim");
    const dnnl::softmax_forward::primitive_desc primitive_desc(
        cpu_engine(), dnnl::prop_kind::forward_inference, dnnl::algorithm::softmax_accurate, src_desc, dst_desc,
        static_cast<int>(input_compute.tensor_info().shape.rank() - 1));
    const dnnl::softmax_forward primitive(primitive_desc);
    dnnl::memory src_memory = make_memory(src_desc, input_compute.data());
    dnnl::memory dst_memory = make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(cpu_engine());
    primitive.execute(stream, {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
    stream.wait();
    return maybe_cast_result(std::move(result), output_dtype, "softmax_last_dim_result");
}

tensors::Tensor one_dnn_rms_norm(const tensors::TensorView& input, const tensors::TensorView& weight, float epsilon) {
    std::optional<tensors::Tensor> input_storage;
    std::optional<tensors::Tensor> weight_storage;
    const tensors::TensorView input_f32 =
        maybe_cast_to_dtype(input, tensors::DType::F32, input_storage, "rms_norm_result");
    const tensors::TensorView weight_f32 =
        maybe_cast_to_dtype(weight, tensors::DType::F32, weight_storage, "rms_norm_result");

    std::vector<std::int64_t> reduced_dims = input_f32.tensor_info().shape.dims();
    reduced_dims.back() = 1;
    const tensors::Shape reduced_shape(std::move(reduced_dims));

    tensors::Tensor squared =
        one_dnn_binary_impl("rms_norm_squared", input_f32, input_f32, dnnl::algorithm::binary_mul, tensors::DType::F32);
    tensors::Tensor mean_squares =
        one_dnn_reduction_impl("rms_norm_mean", squared.view(), reduced_shape, dnnl::algorithm::reduction_mean);
    tensors::Tensor epsilon_tensor = make_f32_tensor("rms_norm_epsilon", reduced_shape, epsilon);
    tensors::Tensor variance = one_dnn_binary_impl("rms_norm_variance", mean_squares.view(), epsilon_tensor.view(),
                                                   dnnl::algorithm::binary_add, tensors::DType::F32);
    tensors::Tensor rms =
        one_dnn_unary_impl("rms_norm_rms", variance.view(), dnnl::algorithm::eltwise_sqrt, tensors::DType::F32);
    tensors::Tensor ones_tensor = make_f32_tensor("rms_norm_one", reduced_shape, 1.0f);
    tensors::Tensor inv_rms = one_dnn_binary_impl("rms_norm_inv_rms", ones_tensor.view(), rms.view(),
                                                  dnnl::algorithm::binary_div, tensors::DType::F32);
    tensors::Tensor normalized = one_dnn_binary_impl("rms_norm_normalized", input_f32, inv_rms.view(),
                                                     dnnl::algorithm::binary_mul, tensors::DType::F32);

    std::vector<std::int64_t> weight_dims(input_f32.tensor_info().shape.rank(), 1);
    weight_dims.back() = weight_f32.tensor_info().shape.dims()[0];
    const tensors::TensorView weight_broadcast = reshape(weight_f32, tensors::Shape(std::move(weight_dims)));
    tensors::Tensor result = one_dnn_binary_impl("rms_norm_result", normalized.view(), weight_broadcast,
                                                 dnnl::algorithm::binary_mul, tensors::DType::F32);

    return maybe_cast_result(std::move(result), input.tensor_info().dtype, "rms_norm_result");
}

} // namespace cppinf::ops::detail
