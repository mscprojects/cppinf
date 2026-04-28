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

tensors::TensorInfo make_result_info(std::string_view name, tensors::DType dtype, const tensors::Shape& shape) {
    return tensors::TensorInfo{
        .name = std::string(name),
        .dtype = dtype,
        .shape = shape,
        .byte_offset = 0,
    };
}

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
    return detail::one_dnn_cast(input, dtype);
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

    if (input.tensor_info().dtype == tensors::DType::I64) {
        const auto& dims = input.tensor_info().shape.dims();
        const std::size_t rows = checked_dim_to_size(dims[0], "transpose_2d rows");
        const std::size_t cols = checked_dim_to_size(dims[1], "transpose_2d cols");
        const std::size_t element_size = tensors::element_size_bytes(input.tensor_info().dtype);

        tensors::Tensor result = tensors::Tensor::zeros(
            make_result_info("transpose_2d_result", input.tensor_info().dtype,
                             tensors::Shape({checked_size_to_dim(cols, "transpose_2d output cols"),
                                             checked_size_to_dim(rows, "transpose_2d output rows")})));

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                const std::size_t source_index = row * cols + col;
                const std::size_t destination_index = col * rows + row;
                std::memcpy(result.mutable_data().data() + destination_index * element_size,
                            input.data().data() + source_index * element_size, element_size);
            }
        }

        return result;
    }

    return detail::one_dnn_transpose_2d(input);
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
