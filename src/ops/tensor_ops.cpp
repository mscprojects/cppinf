#include "ops/tensor_ops.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ops/one_dnn_utils.h"
#include "ops/op_utils.h"

namespace cppinf::ops {
namespace {

std::size_t checked_dim_to_size(std::int64_t dim, std::string_view field_name) {
    if (dim < 0) {
        throw std::invalid_argument(fmt::format("{} must be non-negative.", field_name));
    }

    return static_cast<std::size_t>(dim);
}

std::int64_t checked_size_to_dim(std::size_t value, std::string_view field_name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(fmt::format("{} does not fit in int64_t.", field_name));
    }

    return static_cast<std::int64_t>(value);
}

} // namespace

tensors::Tensor cast(const tensors::TensorView& input, tensors::DType dtype) {
    const tensors::DType source_dtype = input.tensor_info().dtype;
    detail::validate_supported_float_dtype(source_dtype, "cast");
    detail::validate_supported_float_dtype(dtype, "cast");
    return detail::cast_with_one_dnn(input, dtype, "cast_result");
}

tensors::TensorView reshape(const tensors::TensorView& input, tensors::Shape shape) {
    tensors::TensorInfo reshaped_info{
        .name = input.tensor_info().name,
        .dtype = input.tensor_info().dtype,
        .shape = std::move(shape),
        .byte_offset = input.tensor_info().byte_offset,
    };
    if (reshaped_info.byte_size() != input.byte_size()) {
        throw std::invalid_argument("reshape requires the same total byte size.");
    }

    return tensors::TensorView(std::move(reshaped_info), input.data());
}

tensors::Tensor transpose_2d(const tensors::TensorView& input) {
    if (input.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("transpose_2d requires a rank-2 tensor.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const dnnl::memory::dims transposed_dims = {
        static_cast<dnnl::memory::dim>(dims[1]),
        static_cast<dnnl::memory::dim>(dims[0]),
    };
    const dnnl::memory::dims src_strides = {1, transposed_dims[0]};
    const dnnl::memory::dims dst_strides = {transposed_dims[1], 1};

    auto result = detail::make_result_tensor("transpose_2d_result", input.tensor_info().dtype,
                                             tensors::Shape({dims[1], dims[0]}));
    const dnnl::memory::data_type data_type = detail::to_dnnl_dtype(input.tensor_info().dtype);
    dnnl::memory::desc src_desc(transposed_dims, data_type, src_strides);
    dnnl::memory::desc dst_desc(transposed_dims, data_type, dst_strides);
    dnnl::memory src_memory = detail::make_memory(src_desc, input.data());
    dnnl::memory dst_memory = detail::make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(detail::cpu_engine());
    dnnl::reorder(src_memory, dst_memory).execute(stream, src_memory, dst_memory);
    stream.wait();
    return result;
}

tensors::TensorView narrow(const tensors::TensorView& input, std::size_t dim, std::size_t start, std::size_t length) {
    if (input.tensor_info().shape.rank() == 0) {
        throw std::invalid_argument("narrow requires a tensor with rank at least 1.");
    }

    if (dim != 0) {
        throw std::invalid_argument("narrow currently supports only dim=0 for contiguous slices.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const std::size_t dim_size = checked_dim_to_size(dims[0], "narrow dim size");
    if (start > dim_size || length > dim_size - start) {
        throw std::out_of_range("narrow range is out of bounds.");
    }

    std::size_t elements_per_slice = 1;
    for (std::size_t index = 1; index < dims.size(); ++index) {
        elements_per_slice *= checked_dim_to_size(dims[index], "narrow trailing dim");
    }

    const std::size_t bytes_per_slice = elements_per_slice * tensors::element_size_bytes(input.tensor_info().dtype);
    const std::size_t byte_start = start * bytes_per_slice;
    const std::size_t byte_length = length * bytes_per_slice;

    std::vector<std::int64_t> narrowed_dims = dims;
    narrowed_dims[0] = checked_size_to_dim(length, "narrow length");

    tensors::TensorInfo narrowed_info{
        .name = input.tensor_info().name,
        .dtype = input.tensor_info().dtype,
        .shape = tensors::Shape(std::move(narrowed_dims)),
        .byte_offset = input.tensor_info().byte_offset + byte_start,
    };

    return tensors::TensorView(std::move(narrowed_info), input.data().subspan(byte_start, byte_length));
}

} // namespace cppinf::ops
