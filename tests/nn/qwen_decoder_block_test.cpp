#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "nn/qwen_decoder_block.h"
#include "tensors/bfloat16.h"
#include "tensors/tensor.h"

namespace cppinf::tests {

using nn::qwen_decoder_block;
using nn::QwenAttentionWeights;
using nn::QwenDecoderBlockWeights;
using nn::QwenMlpWeights;
using tensors::bfloat16_bits_to_float;
using tensors::DType;
using tensors::float_to_bfloat16_bits;
using tensors::Shape;
using tensors::Tensor;
using tensors::TensorInfo;
using tensors::TensorView;

class QwenDecoderBlockTest : public ::testing::Test {
  protected:
    Tensor make_f32_tensor(std::string_view name, std::initializer_list<std::int64_t> dims,
                           std::initializer_list<float> values) const {
        std::vector<std::byte> bytes(values.size() * sizeof(float));
        std::size_t index = 0;
        for (const auto value : values) {
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
        for (const auto value : values) {
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

    Tensor make_f32_filled_tensor(std::string_view name, std::initializer_list<std::int64_t> dims, float value) const {
        std::size_t element_count = 1;
        for (const auto dim : dims) {
            element_count *= static_cast<std::size_t>(dim);
        }

        std::vector<std::byte> bytes(element_count * sizeof(float));
        for (std::size_t index = 0; index < element_count; ++index) {
            std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
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

TEST_F(QwenDecoderBlockTest, GivenTorchOracleF32Inputs_WhenApplyingQwenDecoderBlock_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/qwen_decoder_block_oracle.py.
    // Case: f32_basic.
    const auto hidden_states = make_f32_tensor("hidden_states", {3, 6},
                                               {-0.37f, -1.03f, 1.22f, -0.20f, 1.09f, 0.76f, -0.73f, 0.55f, -0.82f,
                                                -0.63f, 0.97f, 0.11f, 1.24f, 0.43f, -0.98f, -0.45f, 0.35f, -0.50f});
    const auto input_layernorm_weight =
        make_f32_tensor("input_layernorm_weight", {6}, {0.51f, 0.66f, 1.49f, 0.57f, 0.85f, 0.77f});
    const auto post_attention_layernorm_weight =
        make_f32_tensor("post_attention_layernorm_weight", {6}, {1.30f, 1.33f, 0.68f, 1.31f, 0.70f, 0.58f});
    const auto q_proj_weight = make_f32_tensor(
        "q_proj_weight", {8, 6},
        {-0.49f, 0.85f,  1.24f,  0.56f,  -0.95f, -1.03f, -0.83f, 1.02f,  -0.13f, -0.08f, 0.70f,  0.03f,
         -0.80f, -0.00f, -0.59f, -0.13f, -0.67f, 1.18f,  0.99f,  -1.16f, 0.79f,  -1.11f, -0.35f, 0.67f,
         -1.22f, -0.08f, -0.83f, -1.16f, 0.43f,  -1.05f, -0.97f, -0.28f, -0.48f, 0.16f,  -0.43f, -1.09f,
         -0.59f, -0.74f, -0.03f, -0.34f, 0.13f,  0.18f,  0.28f,  0.39f,  1.17f,  0.33f,  1.16f,  -0.72f});
    const auto q_norm_weight = make_f32_tensor("q_norm_weight", {4}, {0.81f, 1.17f, 0.61f, 1.03f});
    const auto k_proj_weight =
        make_f32_tensor("k_proj_weight", {4, 6},
                        {0.06f, 1.00f,  -0.24f, -0.47f, -1.00f, -0.41f, -0.75f, -0.83f, -1.10f, -1.04f, 1.18f,  -0.06f,
                         0.08f, -0.48f, -0.29f, 0.36f,  0.81f,  -1.11f, -0.40f, 0.59f,  -0.01f, -0.46f, -0.83f, 0.79f});
    const auto k_norm_weight = make_f32_tensor("k_norm_weight", {4}, {0.53f, 0.82f, 0.79f, 1.14f});
    const auto v_proj_weight =
        make_f32_tensor("v_proj_weight", {4, 6},
                        {0.38f, 0.74f, 1.09f, -0.70f, -0.79f, -0.74f, 0.38f, -0.46f, -0.80f, -0.23f, 0.88f,  0.77f,
                         0.24f, 0.06f, 1.13f, 0.91f,  -0.05f, -0.10f, 0.47f, 0.56f,  -0.71f, -0.21f, -0.85f, 0.42f});
    const auto o_proj_weight =
        make_f32_tensor("o_proj_weight", {6, 8},
                        {0.97f,  -0.12f, -0.37f, -0.69f, -0.10f, -1.14f, 0.18f, 0.89f,  0.81f,  0.38f,  0.61f,  0.13f,
                         -0.04f, 0.96f,  0.98f,  -0.60f, -0.33f, 0.11f,  0.92f, 1.02f,  -0.34f, 0.89f,  0.29f,  0.03f,
                         0.61f,  0.62f,  -0.40f, -0.63f, -0.47f, -0.06f, 1.24f, 0.37f,  0.99f,  -0.21f, -0.55f, 0.59f,
                         0.52f,  -0.32f, 0.56f,  0.45f,  0.10f,  0.09f,  0.26f, -1.12f, 0.41f,  -0.70f, 0.89f,  0.82f});
    const auto gate_proj_weight = make_f32_tensor(
        "gate_proj_weight", {10, 6},
        {0.44f,  -0.70f, -1.21f, -0.90f, 0.64f,  -0.77f, -0.79f, -1.18f, 0.45f,  1.24f,  0.01f,  -0.01f,
         0.27f,  -1.07f, 0.91f,  -0.04f, -0.01f, -0.41f, -1.13f, 0.24f,  0.46f,  0.27f,  1.05f,  0.01f,
         -0.97f, 1.18f,  0.24f,  1.11f,  0.09f,  0.64f,  -0.66f, 1.12f,  -0.29f, 0.32f,  -0.84f, -1.05f,
         0.14f,  -0.25f, -0.20f, -0.46f, -0.25f, -0.63f, 1.08f,  0.19f,  0.16f,  -0.51f, -0.82f, -0.18f,
         -0.82f, -0.84f, -0.63f, 0.22f,  0.22f,  0.88f,  -0.22f, 0.73f,  -0.76f, 0.47f,  0.71f,  0.03f});
    const auto up_proj_weight = make_f32_tensor(
        "up_proj_weight", {10, 6},
        {0.09f,  -0.67f, -0.55f, -1.15f, 1.16f,  0.45f,  -0.22f, -0.95f, 1.01f,  0.88f,  -0.59f, 0.15f,
         0.69f,  0.85f,  -1.11f, 0.59f,  -0.46f, 0.12f,  0.97f,  0.45f,  -0.19f, -0.20f, -0.23f, 0.51f,
         0.74f,  0.15f,  -0.87f, 0.93f,  -0.96f, -1.24f, 1.02f,  -0.29f, -0.85f, 0.12f,  0.68f,  -1.09f,
         -0.57f, -0.44f, 0.77f,  0.80f,  -0.17f, -1.08f, -0.47f, 1.16f,  1.18f,  -0.15f, 0.40f,  0.24f,
         0.06f,  -0.86f, -0.15f, -0.98f, 0.55f,  -0.22f, -0.04f, 0.97f,  -0.14f, 0.88f,  -0.32f, -0.33f});
    const auto down_proj_weight = make_f32_tensor(
        "down_proj_weight", {6, 10},
        {0.29f,  -0.93f, 0.27f,  -1.02f, 0.99f,  -0.92f, -1.02f, 1.02f,  0.76f,  0.96f,  0.40f,  -0.65f,
         -1.09f, -0.70f, -1.11f, -0.34f, -0.22f, 0.36f,  -0.85f, 1.06f,  -0.59f, -0.42f, 0.69f,  0.73f,
         1.18f,  -0.10f, 0.27f,  -0.18f, -0.41f, -0.03f, 1.04f,  1.03f,  1.04f,  -0.22f, -0.81f, -0.10f,
         -0.94f, 0.54f,  -0.18f, -0.94f, -0.30f, 0.96f,  -0.48f, 0.15f,  0.51f,  -0.84f, -0.48f, -0.35f,
         -1.24f, -0.08f, 0.29f,  0.47f,  -1.14f, -0.49f, -0.09f, -0.64f, -0.86f, -1.18f, 0.69f,  -0.89f});

    const auto weights = QwenDecoderBlockWeights{
        .input_layernorm_weight = input_layernorm_weight.view(),
        .post_attention_layernorm_weight = post_attention_layernorm_weight.view(),
        .attention =
            QwenAttentionWeights{
                .q_proj_weight = q_proj_weight.view(),
                .q_norm_weight = q_norm_weight.view(),
                .k_proj_weight = k_proj_weight.view(),
                .k_norm_weight = k_norm_weight.view(),
                .v_proj_weight = v_proj_weight.view(),
                .o_proj_weight = o_proj_weight.view(),
            },
        .mlp =
            QwenMlpWeights{
                .gate_proj_weight = gate_proj_weight.view(),
                .up_proj_weight = up_proj_weight.view(),
                .down_proj_weight = down_proj_weight.view(),
            },
    };

    const auto result = qwen_decoder_block(hidden_states.view(), weights, 2, 1, 4, 1e-6f, 2);

    EXPECT_EQ(std::string("qwen_decoder_block_result"), result.tensor_info().name);
    EXPECT_EQ(DType::F32, result.tensor_info().dtype);
    EXPECT_EQ(Shape({3, 6}), result.tensor_info().shape);
    expect_float_values_near(result.view(),
                             {6.2129516602f, 9.4969549179f, 0.9222573638f, -1.9176485538f, 2.2632308006f, 4.5144519806f,
                              4.3219985962f, 4.7326850891f, -1.5501689911f, 1.2058897018f, 2.7489449978f, 3.3651835918f,
                              -2.3978850842f, -0.0118984580f, 0.0132583976f, -2.1403999329f, -2.0698964596f,
                              -3.5824365616f},
                             1e-5f);
}

TEST_F(QwenDecoderBlockTest, GivenTorchOracleBf16Inputs_WhenApplyingQwenDecoderBlock_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/qwen_decoder_block_oracle.py.
    // Case: bf16_basic.
    const auto hidden_states = make_bf16_tensor("hidden_states", {3, 6},
                                                {-0.37f, -1.03f, 1.22f, -0.20f, 1.09f, 0.76f, -0.73f, 0.55f, -0.82f,
                                                 -0.63f, 0.97f, 0.11f, 1.24f, 0.43f, -0.98f, -0.45f, 0.35f, -0.50f});
    const auto input_layernorm_weight =
        make_bf16_tensor("input_layernorm_weight", {6}, {0.51f, 0.66f, 1.49f, 0.57f, 0.85f, 0.77f});
    const auto post_attention_layernorm_weight =
        make_bf16_tensor("post_attention_layernorm_weight", {6}, {1.30f, 1.33f, 0.68f, 1.31f, 0.70f, 0.58f});
    const auto q_proj_weight = make_bf16_tensor(
        "q_proj_weight", {8, 6},
        {-0.49f, 0.85f,  1.24f,  0.56f,  -0.95f, -1.03f, -0.83f, 1.02f,  -0.13f, -0.08f, 0.70f,  0.03f,
         -0.80f, -0.00f, -0.59f, -0.13f, -0.67f, 1.18f,  0.99f,  -1.16f, 0.79f,  -1.11f, -0.35f, 0.67f,
         -1.22f, -0.08f, -0.83f, -1.16f, 0.43f,  -1.05f, -0.97f, -0.28f, -0.48f, 0.16f,  -0.43f, -1.09f,
         -0.59f, -0.74f, -0.03f, -0.34f, 0.13f,  0.18f,  0.28f,  0.39f,  1.17f,  0.33f,  1.16f,  -0.72f});
    const auto q_norm_weight = make_bf16_tensor("q_norm_weight", {4}, {0.81f, 1.17f, 0.61f, 1.03f});
    const auto k_proj_weight =
        make_bf16_tensor("k_proj_weight", {4, 6}, {0.06f,  1.00f,  -0.24f, -0.47f, -1.00f, -0.41f, -0.75f, -0.83f,
                                                   -1.10f, -1.04f, 1.18f,  -0.06f, 0.08f,  -0.48f, -0.29f, 0.36f,
                                                   0.81f,  -1.11f, -0.40f, 0.59f,  -0.01f, -0.46f, -0.83f, 0.79f});
    const auto k_norm_weight = make_bf16_tensor("k_norm_weight", {4}, {0.53f, 0.82f, 0.79f, 1.14f});
    const auto v_proj_weight =
        make_bf16_tensor("v_proj_weight", {4, 6},
                         {0.38f, 0.74f, 1.09f, -0.70f, -0.79f, -0.74f, 0.38f, -0.46f, -0.80f, -0.23f, 0.88f,  0.77f,
                          0.24f, 0.06f, 1.13f, 0.91f,  -0.05f, -0.10f, 0.47f, 0.56f,  -0.71f, -0.21f, -0.85f, 0.42f});
    const auto o_proj_weight = make_bf16_tensor(
        "o_proj_weight", {6, 8},
        {0.97f,  -0.12f, -0.37f, -0.69f, -0.10f, -1.14f, 0.18f, 0.89f,  0.81f,  0.38f,  0.61f,  0.13f,
         -0.04f, 0.96f,  0.98f,  -0.60f, -0.33f, 0.11f,  0.92f, 1.02f,  -0.34f, 0.89f,  0.29f,  0.03f,
         0.61f,  0.62f,  -0.40f, -0.63f, -0.47f, -0.06f, 1.24f, 0.37f,  0.99f,  -0.21f, -0.55f, 0.59f,
         0.52f,  -0.32f, 0.56f,  0.45f,  0.10f,  0.09f,  0.26f, -1.12f, 0.41f,  -0.70f, 0.89f,  0.82f});
    const auto gate_proj_weight = make_bf16_tensor(
        "gate_proj_weight", {10, 6},
        {0.44f,  -0.70f, -1.21f, -0.90f, 0.64f,  -0.77f, -0.79f, -1.18f, 0.45f,  1.24f,  0.01f,  -0.01f,
         0.27f,  -1.07f, 0.91f,  -0.04f, -0.01f, -0.41f, -1.13f, 0.24f,  0.46f,  0.27f,  1.05f,  0.01f,
         -0.97f, 1.18f,  0.24f,  1.11f,  0.09f,  0.64f,  -0.66f, 1.12f,  -0.29f, 0.32f,  -0.84f, -1.05f,
         0.14f,  -0.25f, -0.20f, -0.46f, -0.25f, -0.63f, 1.08f,  0.19f,  0.16f,  -0.51f, -0.82f, -0.18f,
         -0.82f, -0.84f, -0.63f, 0.22f,  0.22f,  0.88f,  -0.22f, 0.73f,  -0.76f, 0.47f,  0.71f,  0.03f});
    const auto up_proj_weight = make_bf16_tensor(
        "up_proj_weight", {10, 6},
        {0.09f,  -0.67f, -0.55f, -1.15f, 1.16f,  0.45f,  -0.22f, -0.95f, 1.01f,  0.88f,  -0.59f, 0.15f,
         0.69f,  0.85f,  -1.11f, 0.59f,  -0.46f, 0.12f,  0.97f,  0.45f,  -0.19f, -0.20f, -0.23f, 0.51f,
         0.74f,  0.15f,  -0.87f, 0.93f,  -0.96f, -1.24f, 1.02f,  -0.29f, -0.85f, 0.12f,  0.68f,  -1.09f,
         -0.57f, -0.44f, 0.77f,  0.80f,  -0.17f, -1.08f, -0.47f, 1.16f,  1.18f,  -0.15f, 0.40f,  0.24f,
         0.06f,  -0.86f, -0.15f, -0.98f, 0.55f,  -0.22f, -0.04f, 0.97f,  -0.14f, 0.88f,  -0.32f, -0.33f});
    const auto down_proj_weight = make_bf16_tensor(
        "down_proj_weight", {6, 10},
        {0.29f,  -0.93f, 0.27f,  -1.02f, 0.99f,  -0.92f, -1.02f, 1.02f,  0.76f,  0.96f,  0.40f,  -0.65f,
         -1.09f, -0.70f, -1.11f, -0.34f, -0.22f, 0.36f,  -0.85f, 1.06f,  -0.59f, -0.42f, 0.69f,  0.73f,
         1.18f,  -0.10f, 0.27f,  -0.18f, -0.41f, -0.03f, 1.04f,  1.03f,  1.04f,  -0.22f, -0.81f, -0.10f,
         -0.94f, 0.54f,  -0.18f, -0.94f, -0.30f, 0.96f,  -0.48f, 0.15f,  0.51f,  -0.84f, -0.48f, -0.35f,
         -1.24f, -0.08f, 0.29f,  0.47f,  -1.14f, -0.49f, -0.09f, -0.64f, -0.86f, -1.18f, 0.69f,  -0.89f});

    const auto weights = QwenDecoderBlockWeights{
        .input_layernorm_weight = input_layernorm_weight.view(),
        .post_attention_layernorm_weight = post_attention_layernorm_weight.view(),
        .attention =
            QwenAttentionWeights{
                .q_proj_weight = q_proj_weight.view(),
                .q_norm_weight = q_norm_weight.view(),
                .k_proj_weight = k_proj_weight.view(),
                .k_norm_weight = k_norm_weight.view(),
                .v_proj_weight = v_proj_weight.view(),
                .o_proj_weight = o_proj_weight.view(),
            },
        .mlp =
            QwenMlpWeights{
                .gate_proj_weight = gate_proj_weight.view(),
                .up_proj_weight = up_proj_weight.view(),
                .down_proj_weight = down_proj_weight.view(),
            },
    };

    const auto result = qwen_decoder_block(hidden_states.view(), weights, 2, 1, 4, 1e-6f, 2);

    EXPECT_EQ(std::string("qwen_decoder_block_result"), result.tensor_info().name);
    EXPECT_EQ(DType::BF16, result.tensor_info().dtype);
    EXPECT_EQ(Shape({3, 6}), result.tensor_info().shape);
    expect_float_values_near(result.view(),
                             {6.4375f, 9.5f, 1.0078125f, -2.046875f, 2.328125f, 4.5f, 4.4375f, 4.78125f, -1.546875f,
                              1.203125f, 2.78125f, 3.40625f, -2.40625f, -0.00390625f, 0.0048828125f, -2.125f,
                              -2.078125f, -3.59375f},
                             1e-6f);
}

TEST_F(QwenDecoderBlockTest, GivenMismatchedLayerNormWeight_WhenApplyingQwenDecoderBlock_ThenItThrows) {
    const auto hidden_states = make_f32_tensor(
        "hidden_states", {2, 6}, {1.0f, 0.5f, -0.5f, 1.5f, -1.0f, 0.25f, 0.75f, -0.25f, 0.5f, -1.5f, 1.0f, 0.25f});
    const auto bad_norm = make_f32_tensor("bad_norm", {5}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    const auto q_proj_weight = make_f32_filled_tensor("q_proj_weight", {8, 6}, 0.0f);
    const auto q_norm_weight = make_f32_tensor("q_norm_weight", {4}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto k_proj_weight = make_f32_filled_tensor("k_proj_weight", {4, 6}, 0.0f);
    const auto k_norm_weight = make_f32_tensor("k_norm_weight", {4}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto v_proj_weight = make_f32_filled_tensor("v_proj_weight", {4, 6}, 0.0f);
    const auto o_proj_weight = make_f32_filled_tensor("o_proj_weight", {6, 8}, 0.0f);
    const auto gate_proj_weight = make_f32_filled_tensor("gate_proj_weight", {4, 6}, 0.0f);
    const auto up_proj_weight = make_f32_filled_tensor("up_proj_weight", {4, 6}, 0.0f);
    const auto down_proj_weight = make_f32_filled_tensor("down_proj_weight", {6, 4}, 0.0f);

    const auto weights = QwenDecoderBlockWeights{
        .input_layernorm_weight = bad_norm.view(),
        .post_attention_layernorm_weight = bad_norm.view(),
        .attention =
            QwenAttentionWeights{
                .q_proj_weight = q_proj_weight.view(),
                .q_norm_weight = q_norm_weight.view(),
                .k_proj_weight = k_proj_weight.view(),
                .k_norm_weight = k_norm_weight.view(),
                .v_proj_weight = v_proj_weight.view(),
                .o_proj_weight = o_proj_weight.view(),
            },
        .mlp =
            QwenMlpWeights{
                .gate_proj_weight = gate_proj_weight.view(),
                .up_proj_weight = up_proj_weight.view(),
                .down_proj_weight = down_proj_weight.view(),
            },
    };

    EXPECT_THROW(static_cast<void>(qwen_decoder_block(hidden_states.view(), weights, 2, 1, 4, 1e-6f)),
                 std::invalid_argument);
}

} // namespace cppinf::tests
