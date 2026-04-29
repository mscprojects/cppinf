#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "nn/qwen_attention.h"
#include "tensors/bfloat16.h"
#include "tensors/tensor.h"

using cppinf::nn::qwen_attention;
using cppinf::tensors::DType;
using cppinf::tensors::float_to_bfloat16_bits;
using cppinf::tensors::Shape;
using cppinf::tensors::Tensor;
using cppinf::tensors::TensorInfo;
using cppinf::tensors::TensorView;

class QwenAttentionTest : public ::testing::Test {
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
                values.push_back(cppinf::tensors::bfloat16_bits_to_float(bits));
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

TEST_F(QwenAttentionTest, GivenTorchOracleF32Inputs_WhenApplyingQwenAttention_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/qwen_attention_oracle.py.
    // Case: f32_grouped_kv.
    const auto hidden_states = make_f32_tensor(
        "hidden_states", {3, 8},
        {-0.61f, 0.05f, -0.74f, 0.57f, -1.28f, 1.10f, -1.09f, -1.19f, -0.95f, 0.68f, -0.55f, 0.56f, -1.27f,
         -0.91f, -0.55f, -0.29f, -1.14f, 0.98f, -0.35f, 0.48f, 1.06f, 0.28f, 0.41f, 1.45f});
    const auto q_proj_weight = make_f32_tensor(
        "q_proj_weight", {8, 8},
        {-0.68f, 0.48f, -0.67f, 1.07f, 1.20f, -1.38f, 1.28f, 0.72f, 0.65f, 0.62f, 1.25f, -0.20f, -1.27f, -0.43f,
         -1.06f, 0.10f, -0.28f, -0.80f, -0.14f, 1.42f, -0.12f, 0.05f, -0.23f, 0.24f, 1.34f, 0.92f, 0.53f, 0.33f,
         0.35f, 0.58f, -0.19f, -1.39f, -0.93f, 1.28f, 0.09f, -1.22f, 0.24f, 1.24f, -1.42f, -1.01f, -0.60f, 0.06f,
         -0.35f, -0.16f, -1.46f, 0.70f, 1.32f, 0.92f, -1.06f, -1.21f, 0.62f, 0.03f, 0.61f, -1.47f, -0.09f, 1.06f,
         0.70f, 0.05f, 0.29f, -0.14f, -0.82f, -0.57f, -0.91f, 1.25f});
    const auto k_proj_weight = make_f32_tensor(
        "k_proj_weight", {4, 8},
        {0.83f, 0.52f, -1.15f, 1.16f, 0.47f, 1.04f, -0.59f, 0.32f, 1.46f, 1.01f, 1.20f, -0.31f, 1.14f, -1.17f,
         0.13f, -0.84f, -0.35f, -0.38f, 0.11f, 1.37f, 0.74f, -0.01f, 1.06f, -0.77f, 0.77f, -0.14f, -0.26f, 0.18f,
         -1.15f, 0.17f, 0.50f, 1.28f});
    const auto v_proj_weight = make_f32_tensor(
        "v_proj_weight", {4, 8},
        {-0.47f, 0.54f, 1.50f, -0.64f, 1.43f, -0.74f, 0.66f, 0.59f, 0.42f, 1.19f, -0.61f, 0.39f, 0.01f, -1.13f,
         -0.36f, -1.00f, 0.66f, 0.13f, 0.15f, -0.46f, 0.01f, -0.47f, 0.43f, 1.46f, 0.23f, -0.66f, -0.92f, 0.11f,
         -1.11f, -1.13f, -0.98f, -0.51f});
    const auto o_proj_weight = make_f32_tensor(
        "o_proj_weight", {8, 8},
        {0.11f, 1.03f, 0.58f, 1.15f, -0.94f, 0.13f, -1.33f, 0.86f, 0.31f, 1.45f, -1.07f, 1.20f, 1.27f, 1.21f,
         0.21f, 1.36f, 1.00f, 1.12f, -0.10f, -1.15f, -0.02f, 0.28f, -1.02f, -0.86f, -1.44f, -0.53f, 1.31f, 0.26f,
         -0.09f, 0.06f, 0.94f, -1.32f, -1.16f, -0.50f, -0.86f, 0.77f, 1.06f, -1.46f, -1.27f, -1.46f, 0.57f, 1.21f,
         -1.16f, -0.69f, 0.48f, -0.98f, 1.27f, 0.35f, -0.42f, 0.10f, 0.47f, -0.53f, -1.16f, 0.01f, 0.03f, 0.03f,
         -0.22f, 0.96f, -0.42f, -0.15f, 0.62f, -0.94f, 0.40f, -0.33f});

    const auto result = qwen_attention(hidden_states.view(), q_proj_weight.view(), k_proj_weight.view(),
                                       v_proj_weight.view(), o_proj_weight.view(), 2, 1);

    EXPECT_EQ(std::string("qwen_attention_result"), result.tensor_info().name);
    EXPECT_EQ(DType::F32, result.tensor_info().dtype);
    EXPECT_EQ(Shape({3, 8}), result.tensor_info().shape);
    expect_float_values_near(result.view(),
                             {12.7699165344f, 3.0965251923f, -4.9420266151f, -2.8355686665f, 4.7417807579f,
                              -6.5129480362f, 5.3829059601f, -3.1677117348f, 12.0490608215f, 9.6016778946f,
                              -3.6127755642f, -3.6095561981f, -2.0740697384f, -3.7650949955f, 2.5245223045f,
                              -2.4809310436f, -5.9976835251f, -4.7422828674f, 5.1256465912f, -0.0880783796f,
                              -2.7990372181f, 1.7909069061f, -0.4297478199f, 0.1040751934f},
                             1e-5f);
}

TEST_F(QwenAttentionTest, GivenTorchOracleBf16Inputs_WhenApplyingQwenAttention_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/qwen_attention_oracle.py.
    // Case: bf16_grouped_kv.
    const auto hidden_states = make_bf16_tensor(
        "hidden_states", {2, 8},
        {0.76171875f, -0.6796875f, 1.109375f, -0.94921875f, 0.71875f, -0.080078125f, -0.240234375f, 0.33984375f,
         0.6015625f, -0.87890625f, 0.8203125f, 0.08984375f, -0.419921875f, -0.7109375f, -0.44921875f, -0.6796875f});
    const auto q_proj_weight = make_bf16_tensor(
        "q_proj_weight", {8, 8},
        {0.87890625f, -0.62890625f, -0.150390625f, 0.83984375f, 1.171875f, -0.87109375f, -0.380859375f,
         -0.69921875f, -0.419921875f, 0.94140625f, -0.44921875f, -0.380859375f, 1.1875f, -0.33984375f,
         -0.30078125f, 0.349609375f, -0.390625f, -0.419921875f, 0.0f, -0.9296875f, -0.3203125f, -0.69140625f,
         0.1904296875f, 0.3203125f, -1.140625f, -0.1904296875f, -0.44921875f, -1.171875f, -0.80078125f,
         0.150390625f, -1.1875f, 1.2109375f, -0.390625f, -0.94140625f, 1.2265625f, 0.240234375f, -1.15625f,
         1.1875f, 0.0f, -0.2197265625f, -1.15625f, 0.69140625f, -0.640625f, -1.109375f, -0.25f, 0.78125f,
         0.330078125f, -0.010009765625f, 1.1015625f, 0.390625f, -0.080078125f, 1.0234375f, -0.349609375f,
         0.2890625f, -0.62890625f, 1.21875f, 0.080078125f, 1.0390625f, -0.310546875f, 0.62109375f, -0.91015625f,
         -0.828125f, -0.7890625f, -0.310546875f});
    const auto k_proj_weight = make_bf16_tensor(
        "k_proj_weight", {4, 8},
        {-0.359375f, -0.150390625f, 0.41015625f, -0.16015625f, 0.55078125f, -0.1396484375f, 0.78125f, 0.87109375f,
         -1.1875f, -1.15625f, -0.030029296875f, 0.390625f, 0.380859375f, -0.08984375f, -1.2109375f, 0.9296875f,
         0.9609375f, 1.0234375f, -1.2265625f, -0.73046875f, 0.94921875f, 0.010009765625f, -0.69140625f, 1.0234375f,
         -0.02001953125f, -0.62890625f, 0.240234375f, -0.8203125f, -0.62890625f, -0.87890625f, -0.62109375f,
         0.6484375f});
    const auto v_proj_weight = make_bf16_tensor(
        "v_proj_weight", {4, 8},
        {-0.69921875f, 0.5390625f, -0.7890625f, 0.6484375f, 1.2109375f, -0.1796875f, -0.470703125f, 0.66015625f,
         -1.078125f, 0.80078125f, -0.2099609375f, -0.98046875f, -0.80859375f, 0.349609375f, -0.169921875f,
         0.1796875f, 0.69140625f, 0.73828125f, -0.51953125f, -0.51953125f, 0.06982421875f, -0.48046875f,
         -0.310546875f, 1.1171875f, -0.33984375f, 1.109375f, 0.62109375f, 0.94921875f, -1.0390625f, 0.7890625f,
         -1.2421875f, 0.400390625f});
    const auto o_proj_weight = make_bf16_tensor(
        "o_proj_weight", {8, 8},
        {-1.0078125f, -1.171875f, -0.25f, 0.2890625f, -1.046875f, -0.330078125f, 0.87109375f, 0.1796875f,
         0.69140625f, -0.030029296875f, 0.8203125f, -0.76953125f, -0.349609375f, -0.80859375f, -0.2099609375f,
         0.7890625f, 0.41015625f, 0.010009765625f, -0.58984375f, -0.80859375f, -0.240234375f, -0.66015625f,
         0.53125f, -0.349609375f, 0.75f, -0.0400390625f, -0.51953125f, 0.44921875f, -0.8203125f, 0.1904296875f,
         -0.671875f, -0.25f, 0.8515625f, 1.15625f, 0.1904296875f, -0.859375f, -0.23046875f, 0.76953125f,
         0.240234375f, 0.9296875f, 1.1171875f, -0.69921875f, 0.69921875f, -0.08984375f, 0.2197265625f, 1.1875f,
         0.16015625f, 0.2197265625f, -0.30078125f, -0.87890625f, -0.30078125f, -0.8515625f, -0.4296875f,
         -0.4609375f, -0.55078125f, -1.1328125f, 0.671875f, -1.1484375f, -0.33984375f, -0.419921875f, -0.83984375f,
         0.41015625f, 1.203125f, -1.1796875f});

    const auto result = qwen_attention(hidden_states.view(), q_proj_weight.view(), k_proj_weight.view(),
                                       v_proj_weight.view(), o_proj_weight.view(), 2, 1, 1);

    EXPECT_EQ(std::string("qwen_attention_result"), result.tensor_info().name);
    EXPECT_EQ(DType::BF16, result.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 8}), result.tensor_info().shape);
    expect_float_values_near(result.view(),
                             {3.71875f, 0.8515625f, 2.390625f, -0.9921875f, -2.890625f, -1.9296875f, 5.1875f,
                              4.03125f, 4.03125f, 1.0234375f, 2.09375f, -0.455078125f, -3.078125f, -2.421875f,
                              5.21875f, 3.21875f},
                             1e-6f);
}

TEST_F(QwenAttentionTest, GivenUndividedHeadCounts_WhenApplyingQwenAttention_ThenItThrows) {
    const auto hidden_states = make_f32_tensor("hidden_states", {2, 8},
                                               {1.0f, 0.5f, -0.5f, 1.5f, -1.0f, 0.25f, 0.75f, -0.25f,
                                                0.5f, -1.5f, 1.0f, 0.25f, -0.75f, 1.25f, -0.5f, 0.75f});
    const auto q_proj_weight = make_f32_tensor(
        "q_proj_weight", {8, 8},
        {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
    const auto k_proj_weight = make_f32_tensor(
        "k_proj_weight", {4, 8},
        {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    const auto v_proj_weight = k_proj_weight;
    const auto o_proj_weight = q_proj_weight;

    EXPECT_THROW(static_cast<void>(qwen_attention(hidden_states.view(), q_proj_weight.view(), k_proj_weight.view(),
                                                  v_proj_weight.view(), o_proj_weight.view(), 3, 2)),
                 std::invalid_argument);
}
