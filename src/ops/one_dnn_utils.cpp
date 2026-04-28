#include "ops/one_dnn_utils.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>
#include <oneapi/dnnl/dnnl.hpp>

namespace cppinf::ops::detail {
namespace {

tensors::TensorInfo make_result_info(std::string_view name, tensors::DType dtype, const tensors::Shape& shape) {
    return tensors::TensorInfo{
        .name = std::string(name),
        .dtype = dtype,
        .shape = shape,
        .byte_offset = 0,
    };
}

dnnl::memory::data_type to_dnnl_dtype_impl(tensors::DType dtype) {
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

} // namespace

const dnnl::engine& cpu_engine() {
    static const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    return engine;
}

tensors::Tensor make_result_tensor(std::string_view name, tensors::DType dtype, const tensors::Shape& shape) {
    return tensors::Tensor::zeros(make_result_info(name, dtype, shape));
}

dnnl::memory::desc make_dense_desc(const tensors::Shape& shape, tensors::DType dtype, std::string_view op_name) {
    const dnnl::memory::dims dims = to_dnnl_dims(shape, op_name);
    return dnnl::memory::desc(dims, to_dnnl_dtype(dtype), make_dense_strides(dims));
}

dnnl::memory::data_type to_dnnl_dtype(tensors::DType dtype) {
    return to_dnnl_dtype_impl(dtype);
}

dnnl::memory make_memory(const dnnl::memory::desc& desc, std::span<const std::byte> bytes) {
    return dnnl::memory(desc, cpu_engine(), const_cast<std::byte*>(bytes.data()));
}

dnnl::memory make_memory(const dnnl::memory::desc& desc, std::span<std::byte> bytes) {
    return dnnl::memory(desc, cpu_engine(), bytes.data());
}

tensors::Tensor cast_with_one_dnn(const tensors::TensorView& input, tensors::DType dtype,
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

    storage.emplace(cast_with_one_dnn(input, dtype, result_name));
    return storage->view();
}

tensors::Tensor maybe_cast_result(tensors::Tensor tensor, tensors::DType dtype, std::string_view result_name) {
    if (tensor.tensor_info().dtype == dtype) {
        return tensor;
    }

    return cast_with_one_dnn(tensor.view(), dtype, result_name);
}

tensors::Tensor binary_with_one_dnn(std::string_view result_name, const tensors::TensorView& lhs,
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

tensors::Tensor unary_with_one_dnn(std::string_view result_name, const tensors::TensorView& input,
                                   dnnl::algorithm algorithm, tensors::DType output_dtype, float alpha, float beta) {
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

} // namespace cppinf::ops::detail
