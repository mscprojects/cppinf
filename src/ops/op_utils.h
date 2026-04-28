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

namespace cppinf::ops::detail {

inline void validate_supported_float_dtype(tensors::DType dtype, std::string_view op_name) {
    switch (dtype) {
    case tensors::DType::BF16:
    case tensors::DType::F32:
        return;
    case tensors::DType::F16:
    case tensors::DType::I32:
    case tensors::DType::I64:
    case tensors::DType::U8:
        throw std::invalid_argument(fmt::format("{} currently supports only f32 and bf16 tensors.", op_name));
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
    case tensors::DType::F16:
    case tensors::DType::I32:
    case tensors::DType::I64:
    case tensors::DType::U8:
        break;
    }

    throw std::invalid_argument("Unsupported dtype for floating-point load.");
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
    case tensors::DType::F16:
    case tensors::DType::I32:
    case tensors::DType::I64:
    case tensors::DType::U8:
        break;
    }

    throw std::invalid_argument("Unsupported dtype for floating-point store.");
}

} // namespace cppinf::ops::detail
