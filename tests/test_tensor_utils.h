#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ops/op_utils.h"
#include "tensors/shape.h"
#include "tensors/tensor.h"
#include "tensors/tensor_info.h"
#include "tensors/tensor_utils.h"
#include "tensors/tensor_view.h"

namespace cppinf::tests::tensor_test_utils {
namespace detail {

inline tensors::Shape make_shape(std::initializer_list<std::int64_t> dims) {
    return tensors::Shape(std::vector<std::int64_t>(dims));
}

} // namespace detail

// Creates an owned f32 tensor from literal float values.
inline tensors::Tensor make_f32_tensor(std::string_view name, std::initializer_list<std::int64_t> dims,
                                       std::initializer_list<float> values) {
    return tensors::make_f32_tensor(name, detail::make_shape(dims),
                                    std::span<const float>(values.begin(), values.size()));
}

// Creates an owned bf16 tensor from literal float values.
inline tensors::Tensor make_bf16_tensor(std::string_view name, std::initializer_list<std::int64_t> dims,
                                        std::initializer_list<float> values) {
    return tensors::make_bf16_tensor(name, detail::make_shape(dims),
                                     std::span<const float>(values.begin(), values.size()));
}

// Creates an owned f32 tensor filled with a single repeated value.
inline tensors::Tensor make_f32_filled_tensor(std::string_view name, std::initializer_list<std::int64_t> dims,
                                              float value) {
    std::size_t element_count = 1;
    for (const auto dim : dims) {
        element_count *= static_cast<std::size_t>(dim);
    }

    std::vector<float> values(element_count, value);
    return tensors::make_f32_tensor(name, detail::make_shape(dims),
                                    std::span<const float>(values.data(), values.size()));
}

// Creates an owned bf16 tensor from already-quantized bf16 bit patterns.
inline tensors::Tensor make_bf16_bits_tensor(std::string_view name, std::initializer_list<std::int64_t> dims,
                                             std::initializer_list<std::uint16_t> values) {
    return tensors::make_bf16_bits_tensor(name, detail::make_shape(dims),
                                          std::span<const std::uint16_t>(values.begin(), values.size()));
}

// Reads f32 or bf16 tensor values back as float literals for assertions.
inline std::vector<float> read_float_values(const tensors::TensorView& tensor_view) {
    ops::detail::validate_supported_float_dtype(tensor_view.tensor_info().dtype, "read_float_values");

    std::vector<float> values;
    values.reserve(tensor_view.tensor_info().shape.num_elements());
    for (std::size_t index = 0; index < tensor_view.tensor_info().shape.num_elements(); ++index) {
        values.push_back(ops::detail::load_float_value(tensor_view.tensor_info().dtype, tensor_view.data(), index));
    }
    return values;
}

// Asserts that a float tensor matches expected values elementwise within tolerance.
inline void expect_float_values_near(const tensors::TensorView& tensor_view, std::initializer_list<float> expected,
                                     float tolerance) {
    const auto actual_values = read_float_values(tensor_view);
    ASSERT_EQ(expected.size(), actual_values.size());

    std::size_t index = 0;
    for (const float expected_value : expected) {
        EXPECT_NEAR(expected_value, actual_values[index], tolerance) << "index=" << index;
        ++index;
    }
}

} // namespace cppinf::tests::tensor_test_utils
