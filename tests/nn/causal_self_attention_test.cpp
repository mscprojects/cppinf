#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "nn/causal_self_attention.h"
#include "test_tensor_utils.h"

namespace cppinf::tests {

using nn::causal_self_attention;
using tensor_test_utils::expect_float_values_near;
using tensor_test_utils::make_bf16_tensor;
using tensor_test_utils::make_f32_tensor;
using tensors::DType;
using tensors::Shape;

class CausalSelfAttentionTest : public ::testing::Test {};

TEST_F(CausalSelfAttentionTest, GivenSingleTokenHead_WhenApplyingCausalSelfAttention_ThenValueIsReturned) {
    const auto query = make_f32_tensor("query", {1, 1, 2}, {0.5f, -1.0f});
    const auto key = make_f32_tensor("key", {1, 1, 2}, {1.0f, 0.25f});
    const auto value = make_f32_tensor("value", {1, 1, 2}, {2.0f, -3.0f});
    const auto expected = make_f32_tensor("causal_self_attention_result", {1, 1, 2}, {2.0f, -3.0f});

    EXPECT_EQ(expected, causal_self_attention(query.view(), key.view(), value.view()));
}

TEST_F(CausalSelfAttentionTest,
       GivenTorchOracleF32Inputs_WhenApplyingCausalSelfAttention_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/causal_self_attention_oracle.py.
    // Case: f32_full_sequence.
    const auto query = make_f32_tensor(
        "query", {2, 3, 2}, {0.5f, -1.0f, 1.25f, 0.75f, -0.5f, 2.0f, 1.5f, 0.5f, -1.0f, 1.0f, 0.25f, -0.75f});
    const auto key = make_f32_tensor("key", {2, 3, 2},
                                     {1.0f, 0.0f, 0.5f, -1.5f, -0.75f, 1.25f, 0.25f, 1.5f, -1.25f, 0.5f, 1.0f, -0.5f});
    const auto value = make_f32_tensor(
        "value", {2, 3, 2}, {2.0f, -1.0f, 0.5f, 1.5f, -1.25f, 0.75f, -0.5f, 2.0f, 1.75f, -1.25f, 0.25f, 0.5f});

    const auto result = causal_self_attention(query.view(), key.view(), value.view());

    EXPECT_EQ(std::string("causal_self_attention_result"), result.tensor_info().name);
    EXPECT_EQ(DType::F32, result.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 3, 2}), result.tensor_info().shape);
    expect_float_values_near(result.view(),
                             {2.0f, -1.0f, 1.6626762152f, -0.4377938509f, -0.9587478638f, 0.6133154035f, -0.5f, 2.0f,
                              0.8218277097f, 0.0906931758f, 0.4652084112f, 0.3605027199f},
                             1e-6f);
}

TEST_F(CausalSelfAttentionTest,
       GivenTorchOracleBf16InputsWithPast_WhenApplyingCausalSelfAttention_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/causal_self_attention_oracle.py.
    // Case: bf16_with_past.
    const auto query = make_bf16_tensor("query", {2, 2, 2}, {1.0f, -0.5f, 0.25f, 1.5f, -1.25f, 0.75f, 1.5f, -0.25f});
    const auto key = make_bf16_tensor(
        "key", {2, 4, 2},
        {0.5f, -1.0f, 1.25f, 0.5f, -0.75f, 1.5f, 1.0f, -0.25f, -0.5f, 1.0f, 1.5f, -1.25f, 0.75f, 0.25f, -1.0f, 0.5f});
    const auto value = make_bf16_tensor(
        "value", {2, 4, 2},
        {1.0f, 0.5f, -1.5f, 0.75f, 0.25f, -0.5f, 1.75f, 1.25f, -0.75f, 1.5f, 1.25f, -0.25f, 0.5f, 0.75f, -1.5f, 1.0f});

    const auto result = causal_self_attention(query.view(), key.view(), value.view(), 2);

    EXPECT_EQ(std::string("causal_self_attention_result"), result.tensor_info().name);
    EXPECT_EQ(DType::BF16, result.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 2, 2}), result.tensor_info().shape);
    expect_float_values_near(
        result.view(),
        {-0.2109375f, 0.53515625f, -0.0161132812f, 0.1000976562f, -0.451171875f, 1.296875f, 0.87109375f, 0.123046875f},
        1e-6f);
}

TEST_F(CausalSelfAttentionTest, GivenMismatchedPastLength_WhenApplyingCausalSelfAttention_ThenItThrows) {
    const auto query = make_f32_tensor("query", {1, 2, 2}, {1.0f, 0.5f, -0.5f, 1.5f});
    const auto key = make_f32_tensor("key", {1, 2, 2}, {0.25f, 1.0f, -1.25f, 0.5f});
    const auto value = make_f32_tensor("value", {1, 2, 2}, {1.5f, -0.25f, 0.5f, 0.75f});

    EXPECT_THROW(causal_self_attention(query.view(), key.view(), value.view(), 1), std::invalid_argument);
}

} // namespace cppinf::tests
