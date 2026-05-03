#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

#include "tensors/bfloat16.h"
#include "tensors/dtype.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops::detail {

inline void validate_supported_float_dtype(tensors::DType dtype, std::string_view op_name) {
    switch (dtype) {
    case tensors::DType::BF16:
    case tensors::DType::F32:
        return;
    }

    throw std::invalid_argument(fmt::format("{} received an unsupported dtype.", op_name));
}

inline float load_float_value(tensors::DType dtype, std::span<const std::byte> bytes, std::size_t index) {
    switch (dtype) {
    case tensors::DType::F32: {
        float value = 0.0f;
        std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
        return value;
    }
    case tensors::DType::BF16: {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes.data() + index * sizeof(std::uint16_t), sizeof(std::uint16_t));
        return tensors::bfloat16_bits_to_float(bits);
    }
    }

    throw std::invalid_argument("Unsupported dtype for floating-point load.");
}

inline float load_float_value(const tensors::TensorView& tensor_view, std::size_t index) {
    const auto& dims = tensor_view.tensor_info().shape.dims();
    std::size_t byte_offset = 0;
    std::size_t remaining = index;

    // Convert a logical flat element index into coordinates so strided views, for example narrow/permute inputs, load
    // the value at the logical position rather than assuming dense contiguous storage.
    for (std::size_t axis = dims.size(); axis-- > 0;) {
        const auto dim = static_cast<std::size_t>(dims[axis]);
        if (dim == 0) {
            throw std::invalid_argument("Cannot index into a tensor view with an empty dimension.");
        }

        const auto coord = remaining % dim;
        remaining /= dim;
        byte_offset += coord * tensor_view.strides_bytes()[axis];
    }

    if (remaining != 0) {
        throw std::out_of_range("TensorView flat index is out of bounds.");
    }

    return load_float_value(tensor_view.tensor_info().dtype, tensor_view.data().subspan(byte_offset), 0);
}

inline void store_float_value(tensors::DType dtype, std::span<std::byte> bytes, std::size_t index, float value) {
    switch (dtype) {
    case tensors::DType::F32:
        std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
        return;
    case tensors::DType::BF16: {
        const std::uint16_t bits = tensors::float_to_bfloat16_bits(value);
        std::memcpy(bytes.data() + index * sizeof(std::uint16_t), &bits, sizeof(std::uint16_t));
        return;
    }
    }

    throw std::invalid_argument("Unsupported dtype for floating-point store.");
}

} // namespace cppinf::ops::detail
