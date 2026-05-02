#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "nn/qwen_decoder_block.h"
#include "test_tensor_utils.h"

namespace cppinf::tests {

using nn::qwen_decoder_block;
using nn::QwenAttentionWeights;
using nn::QwenDecoderBlockWeights;
using nn::QwenMlpWeights;
using tensor_test_utils::expect_float_values_near;
using tensor_test_utils::make_bf16_tensor;
using tensor_test_utils::make_f32_filled_tensor;
using tensor_test_utils::make_f32_tensor;
using tensors::DType;
using tensors::Shape;

class QwenDecoderBlockTest : public ::testing::Test {};

TEST_F(QwenDecoderBlockTest, GivenHfQwen3OracleF32Inputs_WhenApplyingQwenDecoderBlock_ThenExpectedValuesAreReturned) {
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

TEST_F(QwenDecoderBlockTest, GivenHfQwen3OracleBf16Inputs_WhenApplyingQwenDecoderBlock_ThenExpectedValuesAreReturned) {
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
                              1.203125f, 2.8125f, 3.421875f, -2.359375f, 0.0f, 0.0029296875f, -2.109375f, -2.0625f,
                              -3.5625f},
                             0.05f);
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

    EXPECT_THROW(qwen_decoder_block(hidden_states.view(), weights, 2, 1, 4, 1e-6f), std::invalid_argument);
}

} // namespace cppinf::tests
