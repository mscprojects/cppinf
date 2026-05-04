#include "ops/tensor_ops.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/checked.h"
#include "ops/one_dnn_utils.h"
#include "ops/op_utils.h"

namespace cppinf::ops {
namespace {

using common::checked_non_negative_dim_to_size;
using common::checked_positive_size;
using common::checked_size_to_dim;

std::vector<std::size_t> coordinates_from_flat_index(std::size_t flat_index, const std::vector<std::int64_t>& dims) {
    std::vector<std::size_t> coordinates(dims.size(), 0);
    for (std::size_t axis = dims.size(); axis-- > 0;) {
        const auto dim = checked_non_negative_dim_to_size(dims[axis], "logical index dim");
        if (dim == 0) {
            throw std::invalid_argument("Cannot index into a tensor with an empty dimension.");
        }

        coordinates[axis] = flat_index % dim;
        flat_index /= dim;
    }

    return coordinates;
}

std::size_t byte_offset_from_coordinates(const std::vector<std::size_t>& coordinates,
                                         const std::vector<std::size_t>& strides_bytes) {
    std::size_t byte_offset = 0;
    for (std::size_t axis = 0; axis < coordinates.size(); ++axis) {
        const auto axis_offset = coordinates[axis] * strides_bytes[axis];
        if (axis_offset > std::numeric_limits<std::size_t>::max() - byte_offset) {
            throw std::overflow_error("Tensor byte offset overflowed.");
        }
        byte_offset += axis_offset;
    }

    return byte_offset;
}

void validate_permutation_axes(std::span<const std::size_t> axes, std::size_t rank) {
    if (axes.size() != rank) {
        throw std::invalid_argument("permute requires one axis per input dimension.");
    }

    std::vector<bool> seen(rank, false);
    for (const auto axis : axes) {
        if (axis >= rank) {
            throw std::out_of_range("permute axis is out of bounds.");
        }

        if (seen[axis]) {
            throw std::invalid_argument("permute axes must be unique.");
        }
        seen[axis] = true;
    }
}

tensors::Tensor transpose_last_two_dims_impl(const tensors::TensorView& input, std::string_view result_name) {
    if (input.tensor_info().shape.rank() != 2 && input.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("transpose_last_two_dims requires a rank-2 or rank-3 tensor.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    std::vector<std::int64_t> output_dims = dims;
    std::swap(output_dims[output_dims.size() - 2], output_dims[output_dims.size() - 1]);

    auto result =
        detail::make_result_tensor(result_name, input.tensor_info().dtype, tensors::Shape(std::move(output_dims)));
    const dnnl::memory::data_type data_type = detail::to_dnnl_dtype(input.tensor_info().dtype);
    const auto element_size = tensors::element_size_bytes(input.tensor_info().dtype);
    dnnl::memory::dims input_strides;
    input_strides.reserve(input.strides_bytes().size());
    for (const auto stride_bytes : input.strides_bytes()) {
        input_strides.push_back(static_cast<dnnl::memory::dim>(stride_bytes / element_size));
    }

    if (dims.size() == 2) {
        const dnnl::memory::dims transposed_dims = {
            static_cast<dnnl::memory::dim>(dims[1]),
            static_cast<dnnl::memory::dim>(dims[0]),
        };
        const dnnl::memory::dims src_strides = {input_strides[1], input_strides[0]};
        const dnnl::memory::dims dst_strides = {transposed_dims[1], 1};
        dnnl::memory::desc src_desc(transposed_dims, data_type, src_strides);
        dnnl::memory::desc dst_desc(transposed_dims, data_type, dst_strides);
        dnnl::memory src_memory = detail::make_memory(src_desc, input.data());
        dnnl::memory dst_memory = detail::make_memory(dst_desc, result.mutable_data());
        dnnl::stream stream(detail::cpu_engine());
        dnnl::reorder(src_memory, dst_memory).execute(stream, src_memory, dst_memory);
        stream.wait();
        return result;
    }

    const auto batch_size = checked_non_negative_dim_to_size(dims[0], "transpose_last_two_dims batch");
    const auto rows = checked_non_negative_dim_to_size(dims[1], "transpose_last_two_dims rows");
    const auto cols = checked_non_negative_dim_to_size(dims[2], "transpose_last_two_dims cols");
    const dnnl::memory::dims transposed_dims = {
        static_cast<dnnl::memory::dim>(batch_size),
        static_cast<dnnl::memory::dim>(cols),
        static_cast<dnnl::memory::dim>(rows),
    };
    const dnnl::memory::dims src_strides = {
        input_strides[0],
        input_strides[2],
        input_strides[1],
    };
    const dnnl::memory::dims dst_strides = {
        static_cast<dnnl::memory::dim>(rows * cols),
        static_cast<dnnl::memory::dim>(rows),
        1,
    };
    dnnl::memory::desc src_desc(transposed_dims, data_type, src_strides);
    dnnl::memory::desc dst_desc(transposed_dims, data_type, dst_strides);
    dnnl::memory src_memory = detail::make_memory(src_desc, input.data());
    dnnl::memory dst_memory = detail::make_memory(dst_desc, result.mutable_data());
    dnnl::stream stream(detail::cpu_engine());
    dnnl::reorder(src_memory, dst_memory).execute(stream, src_memory, dst_memory);
    stream.wait();
    return result;
}

} // namespace

tensors::Tensor cast(const tensors::TensorView& input, tensors::DType dtype) {
    const tensors::DType source_dtype = input.tensor_info().dtype;
    detail::validate_supported_float_dtype(source_dtype, "cast");
    detail::validate_supported_float_dtype(dtype, "cast");
    return detail::cast_with_one_dnn(input, dtype, "cast_result");
}

tensors::TensorView reshape(const tensors::TensorView& input, tensors::Shape shape) {
    if (!input.is_contiguous()) {
        throw std::invalid_argument("reshape requires a contiguous tensor view.");
    }

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

tensors::TensorView squeeze(const tensors::TensorView& input, std::size_t dim) {
    if (input.tensor_info().shape.rank() == 0) {
        throw std::invalid_argument("squeeze requires a tensor with rank at least 1.");
    }

    if (dim >= input.tensor_info().shape.rank()) {
        throw std::out_of_range("squeeze dimension is out of bounds.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    if (checked_non_negative_dim_to_size(dims[dim], "squeeze dim size") != 1) {
        throw std::invalid_argument("squeeze requires the selected dimension to have size 1.");
    }

    std::vector<std::int64_t> squeezed_dims;
    squeezed_dims.reserve(dims.size() - 1);
    std::vector<std::size_t> squeezed_strides;
    squeezed_strides.reserve(input.strides_bytes().size() - 1);
    for (std::size_t index = 0; index < dims.size(); ++index) {
        if (index == dim) {
            continue;
        }

        squeezed_dims.push_back(dims[index]);
        squeezed_strides.push_back(input.strides_bytes()[index]);
    }

    tensors::TensorInfo squeezed_info{
        .name = input.tensor_info().name,
        .dtype = input.tensor_info().dtype,
        .shape = tensors::Shape(std::move(squeezed_dims)),
        .byte_offset = input.tensor_info().byte_offset,
    };
    return tensors::TensorView(std::move(squeezed_info), input.data(), std::move(squeezed_strides));
}

tensors::Tensor transpose_2d(const tensors::TensorView& input) {
    if (input.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("transpose_2d requires a rank-2 tensor.");
    }

    return transpose_last_two_dims_impl(input, "transpose_2d_result");
}

tensors::Tensor transpose_last_two_dims(const tensors::TensorView& input) {
    return transpose_last_two_dims_impl(input, "transpose_last_two_dims_result");
}

tensors::TensorView narrow(const tensors::TensorView& input, std::size_t dim, std::size_t start, std::size_t length) {
    if (input.tensor_info().shape.rank() == 0) {
        throw std::invalid_argument("narrow requires a tensor with rank at least 1.");
    }

    if (dim >= input.tensor_info().shape.rank()) {
        throw std::out_of_range("narrow dimension is out of bounds.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const std::size_t dim_size = checked_non_negative_dim_to_size(dims[dim], "narrow dim size");
    if (start > dim_size || length > dim_size - start) {
        throw std::out_of_range("narrow range is out of bounds.");
    }

    const std::size_t byte_start = start * input.strides_bytes()[dim];

    std::vector<std::int64_t> narrowed_dims = dims;
    narrowed_dims[dim] = checked_size_to_dim(length, "narrow length");

    tensors::TensorInfo narrowed_info{
        .name = input.tensor_info().name,
        .dtype = input.tensor_info().dtype,
        .shape = tensors::Shape(std::move(narrowed_dims)),
        .byte_offset = input.tensor_info().byte_offset + byte_start,
    };

    return tensors::TensorView(std::move(narrowed_info), input.data(), input.strides_bytes(), byte_start);
}

tensors::Tensor permute(const tensors::TensorView& input, std::span<const std::size_t> axes) {
    validate_permutation_axes(axes, input.tensor_info().shape.rank());

    const auto& input_dims = input.tensor_info().shape.dims();
    std::vector<std::int64_t> output_dims;
    output_dims.reserve(axes.size());
    for (const auto axis : axes) {
        output_dims.push_back(input_dims[axis]);
    }

    const auto output_shape = tensors::Shape(std::move(output_dims));
    auto result = detail::make_result_tensor("permute_result", input.tensor_info().dtype, output_shape);
    const auto element_size = tensors::element_size_bytes(input.tensor_info().dtype);
    const auto input_data = input.data();
    auto result_data = result.mutable_data();

    // Materialize the permuted tensor by mapping each dense output coordinate back through the requested axis order.
    for (std::size_t output_index = 0; output_index < output_shape.num_elements(); ++output_index) {
        const auto output_coordinates = coordinates_from_flat_index(output_index, output_shape.dims());
        std::vector<std::size_t> input_coordinates(input_dims.size(), 0);
        for (std::size_t output_axis = 0; output_axis < axes.size(); ++output_axis) {
            input_coordinates[axes[output_axis]] = output_coordinates[output_axis];
        }

        const auto input_offset = byte_offset_from_coordinates(input_coordinates, input.strides_bytes());
        std::memcpy(result_data.data() + output_index * element_size, input_data.data() + input_offset, element_size);
    }

    return result;
}

tensors::Tensor repeat_interleave(const tensors::TensorView& input, std::size_t dim, std::size_t repeats) {
    if (dim >= input.tensor_info().shape.rank()) {
        throw std::out_of_range("repeat_interleave dimension is out of bounds.");
    }

    checked_positive_size(repeats, "repeat_interleave repeats");

    const auto& input_dims = input.tensor_info().shape.dims();
    std::vector<std::int64_t> output_dims = input_dims;
    const auto repeated_dim_size = checked_non_negative_dim_to_size(input_dims[dim], "repeat_interleave dim size");
    if (repeated_dim_size > std::numeric_limits<std::size_t>::max() / repeats) {
        throw std::overflow_error("repeat_interleave output dimension overflowed.");
    }
    output_dims[dim] = checked_size_to_dim(repeated_dim_size * repeats, "repeat_interleave output dim");

    const auto output_shape = tensors::Shape(std::move(output_dims));
    auto result = detail::make_result_tensor("repeat_interleave_result", input.tensor_info().dtype, output_shape);
    const auto element_size = tensors::element_size_bytes(input.tensor_info().dtype);
    const auto input_data = input.data();
    auto result_data = result.mutable_data();

    // Repeated output coordinates map back to the same input coordinate along dim, other axes are unchanged.
    for (std::size_t output_index = 0; output_index < output_shape.num_elements(); ++output_index) {
        auto input_coordinates = coordinates_from_flat_index(output_index, output_shape.dims());
        input_coordinates[dim] /= repeats;

        const auto input_offset = byte_offset_from_coordinates(input_coordinates, input.strides_bytes());
        std::memcpy(result_data.data() + output_index * element_size, input_data.data() + input_offset, element_size);
    }

    return result;
}

} // namespace cppinf::ops
