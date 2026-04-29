#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tensors/bfloat16.h"
#include "tensors/shape.h"
#include "tensors/tensor.h"
#include "tensors/tensor_info.h"

namespace cppinf::tensors {

// Returns zero-offset metadata for a newly created output tensor.
inline TensorInfo make_result_tensor_info(std::string_view name, DType dtype, const Shape& shape) {
    return TensorInfo{
        .name = std::string(name),
        .dtype = dtype,
        .shape = shape,
        .byte_offset = 0,
    };
}

// Returns an owned float tensor with bytes encoded in the requested model dtype.
inline Tensor make_float_tensor(std::string_view name, DType dtype, const Shape& shape, std::span<const float> values) {
    if (shape.num_elements() != values.size()) {
        throw std::invalid_argument("Tensor values must match the requested shape.");
    }

    std::vector<std::byte> bytes(values.size() * element_size_bytes(dtype));
    for (std::size_t index = 0; index < values.size(); ++index) {
        switch (dtype) {
        case DType::BF16: {
            const auto bits = float_to_bfloat16_bits(values[index]);
            std::memcpy(bytes.data() + index * sizeof(bits), &bits, sizeof(bits));
            break;
        }
        case DType::F32:
            std::memcpy(bytes.data() + index * sizeof(float), &values[index], sizeof(float));
            break;
        }
    }

    return Tensor(make_result_tensor_info(name, dtype, shape), std::move(bytes));
}

// Returns an owned f32 tensor from literal float values.
inline Tensor make_f32_tensor(std::string_view name, const Shape& shape, std::span<const float> values) {
    return make_float_tensor(name, DType::F32, shape, values);
}

// Returns an owned bf16 tensor from literal float values.
inline Tensor make_bf16_tensor(std::string_view name, const Shape& shape, std::span<const float> values) {
    return make_float_tensor(name, DType::BF16, shape, values);
}

// Returns an owned bf16 tensor from already-quantized bf16 bit patterns.
inline Tensor make_bf16_bits_tensor(std::string_view name, const Shape& shape, std::span<const std::uint16_t> values) {
    if (shape.num_elements() != values.size()) {
        throw std::invalid_argument("Tensor values must match the requested shape.");
    }

    std::vector<std::byte> bytes(values.size() * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::memcpy(bytes.data() + index * sizeof(std::uint16_t), &values[index], sizeof(std::uint16_t));
    }

    return Tensor(make_result_tensor_info(name, DType::BF16, shape), std::move(bytes));
}

// Returns an owned tensor copy with the same bytes and a different tensor name.
inline Tensor rename_tensor(std::string_view name, const Tensor& tensor) {
    return Tensor(make_result_tensor_info(name, tensor.tensor_info().dtype, tensor.tensor_info().shape),
                  std::vector<std::byte>(tensor.bytes().begin(), tensor.bytes().end()));
}

} // namespace cppinf::tensors
