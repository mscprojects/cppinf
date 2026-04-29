#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "nn/causal_self_attention.h"
#include "tensors/bfloat16.h"
#include "tensors/tensor.h"

namespace cppinf::tests {

using nn::causal_self_attention;
using tensors::bfloat16_bits_to_float;
using tensors::DType;
using tensors::float_to_bfloat16_bits;
using tensors::Shape;
using tensors::Tensor;
using tensors::TensorInfo;
using tensors::TensorView;

class CausalSelfAttentionTest : public ::testing::Test {
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
        for (const float expected_value : expected) {
            EXPECT_NEAR(expected_value, actual_values[index], tolerance) << "index=" << index;
            ++index;
        }
    }
};

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

    EXPECT_THROW(static_cast<void>(causal_self_attention(query.view(), key.view(), value.view(), 1)),
                 std::invalid_argument);
}

} // namespace cppinf::tests
