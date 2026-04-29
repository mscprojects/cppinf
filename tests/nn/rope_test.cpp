#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "nn/rope.h"
#include "tensors/bfloat16.h"
#include "tensors/tensor.h"

namespace cppinf::tests {

using nn::apply_rope;
using tensors::bfloat16_bits_to_float;
using tensors::DType;
using tensors::float_to_bfloat16_bits;
using tensors::Shape;
using tensors::Tensor;
using tensors::TensorInfo;
using tensors::TensorView;

class RopeTest : public ::testing::Test {
  protected:
    Tensor make_f32_tensor(std::string_view name, std::initializer_list<std::int64_t> dims,
                           std::initializer_list<float> values) const {
        std::vector<std::byte> bytes(values.size() * sizeof(float));
        std::size_t index = 0;
        for (const float value : values) {
            std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
            ++index;
        }

        return Tensor(
            TensorInfo{
                .name = std::string(name),
                .dtype = DType::F32,
                .shape = Shape(std::vector<std::int64_t>(dims)),
                .byte_offset = 0,
            },
            std::move(bytes));
    }

    Tensor make_bf16_tensor(std::string_view name, std::initializer_list<std::int64_t> dims,
                            std::initializer_list<float> values) const {
        std::vector<std::byte> bytes(values.size() * sizeof(std::uint16_t));
        std::size_t index = 0;
        for (const float value : values) {
            const auto bits = float_to_bfloat16_bits(value);
            std::memcpy(bytes.data() + index * sizeof(std::uint16_t), &bits, sizeof(std::uint16_t));
            ++index;
        }

        return Tensor(
            TensorInfo{
                .name = std::string(name),
                .dtype = DType::BF16,
                .shape = Shape(std::vector<std::int64_t>(dims)),
                .byte_offset = 0,
            },
            std::move(bytes));
    }

    std::vector<float> read_float_values(const TensorView& tensor_view) const {
        std::vector<float> values;
        values.reserve(tensor_view.tensor_info().shape.num_elements());

        if (tensor_view.tensor_info().dtype == DType::F32) {
            for (std::size_t index = 0; index < tensor_view.tensor_info().shape.num_elements(); ++index) {
                float value = 0.0f;
                std::memcpy(&value, tensor_view.data().data() + index * sizeof(float), sizeof(float));
                values.push_back(value);
            }
            return values;
        }

        if (tensor_view.tensor_info().dtype == DType::BF16) {
            for (std::size_t index = 0; index < tensor_view.tensor_info().shape.num_elements(); ++index) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, tensor_view.data().data() + index * sizeof(std::uint16_t), sizeof(std::uint16_t));
                values.push_back(bfloat16_bits_to_float(bits));
            }
            return values;
        }

        throw std::invalid_argument("read_float_values supports only f32 and bf16 tensors.");
    }

    void expect_float_values_near(const TensorView& tensor_view, std::initializer_list<float> expected,
                                  float tolerance) const {
        const auto actual_values = read_float_values(tensor_view);
        ASSERT_EQ(expected.size(), actual_values.size());

        std::size_t index = 0;
        for (const auto expected_value : expected) {
            EXPECT_NEAR(expected_value, actual_values[index], tolerance) << "index=" << index;
            ++index;
        }
    }
};

TEST_F(RopeTest, GivenZeroPositionOffset_WhenApplyingRope_ThenValuesStayTheSame) {
    const auto input = make_f32_tensor("input", {1, 1, 4}, {1.0f, -0.5f, 0.25f, 1.5f});
    const auto expected = make_f32_tensor("rope_result", {1, 1, 4}, {1.0f, -0.5f, 0.25f, 1.5f});

    EXPECT_EQ(expected, apply_rope(input.view()));
}

TEST_F(RopeTest, GivenTorchOracleBf16Inputs_WhenApplyingRope_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/rope_oracle.py.
    // Case: bf16_position_offset_2.
    const auto input = make_bf16_tensor("input", {2, 3, 4}, {1.0f,  -0.5f, 0.25f,  1.5f, -1.25f, 0.75f, 1.0f,  -0.25f,
                                                             0.5f,  1.25f, -0.75f, 0.5f, -0.5f,  1.0f,  1.5f,  -1.25f,
                                                             0.75f, -1.5f, 0.5f,   1.0f, 1.25f,  0.25f, -1.0f, 0.75f});

    const auto result = apply_rope(input.view(), 2);

    EXPECT_EQ(std::string("rope_result"), result.tensor_info().name);
    EXPECT_EQ(DType::BF16, result.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 3, 4}), result.tensor_info().shape);
    expect_float_values_near(result.view(),
                             {-0.64453125f,  -0.50390625f,  0.8046875f,   1.5f,          1.09375f,       0.75f,
                              -1.1640625f,   -0.248046875f, -0.89453125f, 1.25f,         0.11181640625f, 0.50390625f,
                              -1.15625f,     1.0f,          -1.078125f,   -1.25f,        -0.8125f,       -1.5f,
                              -0.388671875f, 0.99609375f,   -1.5703125f,  0.2470703125f, -0.29296875f,   0.75f},
                             1e-6f);
}

TEST_F(RopeTest, GivenOddHeadSize_WhenApplyingRope_ThenItThrows) {
    const auto input = make_f32_tensor("input", {1, 2, 3}, {1.0f, -0.5f, 0.25f, 0.75f, -1.0f, 0.5f});

    EXPECT_THROW(static_cast<void>(apply_rope(input.view())), std::invalid_argument);
}

} // namespace cppinf::tests
