#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "nn/rope.h"
#include "test_tensor_utils.h"

namespace cppinf::tests {

using nn::apply_rope;
using tensor_test_utils::expect_float_values_near;
using tensor_test_utils::make_bf16_tensor;
using tensor_test_utils::make_f32_tensor;
using tensors::DType;
using tensors::Shape;

class RopeTest : public ::testing::Test {};

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

    EXPECT_THROW(apply_rope(input.view()), std::invalid_argument);
}

} // namespace cppinf::tests
