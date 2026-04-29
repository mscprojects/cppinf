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

namespace cppinf::tests {

using nn::qwen_attention;
using nn::QwenAttentionWeights;
using tensors::DType;
using tensors::bfloat16_bits_to_float;
using tensors::float_to_bfloat16_bits;
using tensors::Shape;
using tensors::Tensor;
using tensors::TensorInfo;
using tensors::TensorView;

class QwenAttentionTest : public ::testing::Test {
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

TEST_F(QwenAttentionTest, GivenTorchOracleF32Inputs_WhenApplyingQwenAttention_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/qwen_attention_oracle.py.
    // Case: f32_explicit_head_dim.
    const auto hidden_states = make_f32_tensor("hidden_states", {3, 6},
                                               {0.29f, 1.20f, 0.75f, -0.67f, 0.26f, 0.29f, 1.30f, 0.24f, -0.61f, -1.14f,
                                                -1.26f, 0.13f, 0.10f, -1.28f, 0.28f, -0.54f, 0.39f, -0.86f});
    const auto q_proj_weight = make_f32_tensor(
        "q_proj_weight", {8, 6},
        {0.60f, 0.95f,  0.54f,  0.02f,  -0.86f, -0.76f, -0.09f, -1.07f, -0.08f, -1.01f, -0.84f, 0.82f,
         0.24f, 0.93f,  -1.25f, -0.46f, 0.89f,  -0.49f, -0.49f, 0.79f,  -0.77f, 1.27f,  0.47f,  0.74f,
         0.19f, -0.81f, -0.42f, 0.26f,  -0.08f, -0.54f, 0.89f,  -0.34f, -1.07f, 1.08f,  -0.38f, -0.80f,
         0.36f, 0.44f,  0.39f,  -0.38f, -0.22f, -0.69f, 1.26f,  -0.55f, -0.36f, 1.20f,  0.19f,  -1.37f});
    const auto q_norm_weight = make_f32_tensor("q_norm_weight", {4}, {1.16f, 0.83f, 1.30f, 1.01f});
    const auto k_proj_weight =
        make_f32_tensor("k_proj_weight", {4, 6},
                        {0.55f, -1.33f, 0.44f, 0.76f,  0.36f,  0.63f,  -1.19f, -1.37f, 0.85f, -0.95f, -0.13f, -1.33f,
                         0.58f, 1.08f,  0.27f, -0.48f, -0.25f, -0.17f, -0.11f, 0.08f,  1.26f, -0.28f, 0.74f,  0.32f});
    const auto k_norm_weight = make_f32_tensor("k_norm_weight", {4}, {0.60f, 0.59f, 0.82f, 0.77f});
    const auto v_proj_weight =
        make_f32_tensor("v_proj_weight", {4, 6},
                        {-0.70f, 0.14f,  -1.00f, -0.32f, -0.95f, 0.29f,  0.44f, -0.86f, 0.57f, -0.77f, -0.69f, -0.73f,
                         1.23f,  -1.25f, -0.15f, -0.33f, -1.17f, -0.29f, 1.36f, -1.13f, 0.21f, 0.13f,  -0.61f, 0.08f});
    const auto o_proj_weight = make_f32_tensor(
        "o_proj_weight", {6, 8},
        {-0.29f, 0.44f,  -0.06f, 0.34f,  -1.39f, -0.64f, 0.78f,  -0.16f, -1.35f, -1.29f, -0.30f, -0.11f,
         -0.11f, -0.50f, -0.79f, 0.16f,  -0.88f, 0.82f,  0.74f,  -0.86f, -0.55f, -0.61f, 0.82f,  0.63f,
         1.00f,  -0.49f, -1.30f, -0.85f, -0.06f, 0.50f,  -0.32f, 1.07f,  1.33f,  0.77f,  1.39f,  -0.28f,
         0.31f,  0.94f,  -1.12f, -0.20f, 0.14f,  0.63f,  0.22f,  -0.86f, 0.21f,  0.75f,  0.06f,  -0.50f});

    const auto weights = QwenAttentionWeights{
        .q_proj_weight = q_proj_weight.view(),
        .q_norm_weight = q_norm_weight.view(),
        .k_proj_weight = k_proj_weight.view(),
        .k_norm_weight = k_norm_weight.view(),
        .v_proj_weight = v_proj_weight.view(),
        .o_proj_weight = o_proj_weight.view(),
    };

    const auto result = qwen_attention(hidden_states.view(), weights, 2, 1, 4, 1e-6f, 1);

    EXPECT_EQ(std::string("qwen_attention_result"), result.tensor_info().name);
    EXPECT_EQ(DType::F32, result.tensor_info().dtype);
    EXPECT_EQ(Shape({3, 6}), result.tensor_info().shape);
    expect_float_values_near(result.view(),
                             {0.0933519751f, 3.2009088993f, -1.0087980032f, 1.3863968849f, -1.6964730024f,
                              0.2551130056f, 0.1748879999f, -2.9569544792f, 0.9807962775f, -1.5669782162f,
                              2.3072979450f, 0.5903985500f, 1.3345407248f, -3.1161534786f, 1.4469410181f,
                              -4.2059197426f, 4.1574010849f, 0.6187804341f},
                             1e-5f);
}

TEST_F(QwenAttentionTest, GivenTorchOracleBf16Inputs_WhenApplyingQwenAttention_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/qwen_attention_oracle.py.
    // Case: bf16_explicit_head_dim.
    const auto hidden_states = make_bf16_tensor("hidden_states", {3, 6},
                                                {0.29f, 1.20f, 0.75f, -0.67f, 0.26f, 0.29f, 1.30f, 0.24f, -0.61f,
                                                 -1.14f, -1.26f, 0.13f, 0.10f, -1.28f, 0.28f, -0.54f, 0.39f, -0.86f});
    const auto q_proj_weight = make_bf16_tensor(
        "q_proj_weight", {8, 6},
        {0.60f, 0.95f,  0.54f,  0.02f,  -0.86f, -0.76f, -0.09f, -1.07f, -0.08f, -1.01f, -0.84f, 0.82f,
         0.24f, 0.93f,  -1.25f, -0.46f, 0.89f,  -0.49f, -0.49f, 0.79f,  -0.77f, 1.27f,  0.47f,  0.74f,
         0.19f, -0.81f, -0.42f, 0.26f,  -0.08f, -0.54f, 0.89f,  -0.34f, -1.07f, 1.08f,  -0.38f, -0.80f,
         0.36f, 0.44f,  0.39f,  -0.38f, -0.22f, -0.69f, 1.26f,  -0.55f, -0.36f, 1.20f,  0.19f,  -1.37f});
    const auto q_norm_weight = make_bf16_tensor("q_norm_weight", {4}, {1.16f, 0.83f, 1.30f, 1.01f});
    const auto k_proj_weight =
        make_bf16_tensor("k_proj_weight", {4, 6},
                         {0.55f, -1.33f, 0.44f, 0.76f,  0.36f,  0.63f,  -1.19f, -1.37f, 0.85f, -0.95f, -0.13f, -1.33f,
                          0.58f, 1.08f,  0.27f, -0.48f, -0.25f, -0.17f, -0.11f, 0.08f,  1.26f, -0.28f, 0.74f,  0.32f});
    const auto k_norm_weight = make_bf16_tensor("k_norm_weight", {4}, {0.60f, 0.59f, 0.82f, 0.77f});
    const auto v_proj_weight =
        make_bf16_tensor("v_proj_weight", {4, 6},
                         {-0.70f, 0.14f,  -1.00f, -0.32f, -0.95f, 0.29f,  0.44f, -0.86f, 0.57f, -0.77f, -0.69f, -0.73f,
                          1.23f,  -1.25f, -0.15f, -0.33f, -1.17f, -0.29f, 1.36f, -1.13f, 0.21f, 0.13f,  -0.61f, 0.08f});
    const auto o_proj_weight = make_bf16_tensor(
        "o_proj_weight", {6, 8},
        {-0.29f, 0.44f,  -0.06f, 0.34f,  -1.39f, -0.64f, 0.78f,  -0.16f, -1.35f, -1.29f, -0.30f, -0.11f,
         -0.11f, -0.50f, -0.79f, 0.16f,  -0.88f, 0.82f,  0.74f,  -0.86f, -0.55f, -0.61f, 0.82f,  0.63f,
         1.00f,  -0.49f, -1.30f, -0.85f, -0.06f, 0.50f,  -0.32f, 1.07f,  1.33f,  0.77f,  1.39f,  -0.28f,
         0.31f,  0.94f,  -1.12f, -0.20f, 0.14f,  0.63f,  0.22f,  -0.86f, 0.21f,  0.75f,  0.06f,  -0.50f});

    const auto weights = QwenAttentionWeights{
        .q_proj_weight = q_proj_weight.view(),
        .q_norm_weight = q_norm_weight.view(),
        .k_proj_weight = k_proj_weight.view(),
        .k_norm_weight = k_norm_weight.view(),
        .v_proj_weight = v_proj_weight.view(),
        .o_proj_weight = o_proj_weight.view(),
    };

    const auto result = qwen_attention(hidden_states.view(), weights, 2, 1, 4, 1e-6f, 1);

    EXPECT_EQ(std::string("qwen_attention_result"), result.tensor_info().name);
    EXPECT_EQ(DType::BF16, result.tensor_info().dtype);
    EXPECT_EQ(Shape({3, 6}), result.tensor_info().shape);
    expect_float_values_near(result.view(),
                             {0.08154296875f, 3.203125f, -1.0234375f, 1.3984375f, -1.6953125f, 0.259765625f,
                              0.1611328125f, -2.96875f, 0.96875f, -1.546875f, 2.3125f, 0.59765625f, 1.328125f,
                              -3.109375f, 1.4296875f, -4.1875f, 4.15625f, 0.6171875f},
                             1e-6f);
}

TEST_F(QwenAttentionTest, GivenMismatchedHeadShape_WhenApplyingQwenAttention_ThenItThrows) {
    const auto hidden_states = make_f32_tensor(
        "hidden_states", {2, 6}, {1.0f, 0.5f, -0.5f, 1.5f, -1.0f, 0.25f, 0.75f, -0.25f, 0.5f, -1.5f, 1.0f, 0.25f});
    const auto q_proj_weight = make_f32_filled_tensor("q_proj_weight", {8, 6}, 0.0f);
    const auto q_norm_weight = make_f32_tensor("q_norm_weight", {4}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto k_proj_weight = make_f32_filled_tensor("k_proj_weight", {4, 6}, 0.0f);
    const auto k_norm_weight = make_f32_tensor("k_norm_weight", {4}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto v_proj_weight = make_f32_filled_tensor("v_proj_weight", {4, 6}, 0.0f);
    const auto o_proj_weight = make_f32_filled_tensor("o_proj_weight", {6, 8}, 0.0f);

    const auto weights = QwenAttentionWeights{
        .q_proj_weight = q_proj_weight.view(),
        .q_norm_weight = q_norm_weight.view(),
        .k_proj_weight = k_proj_weight.view(),
        .k_norm_weight = k_norm_weight.view(),
        .v_proj_weight = v_proj_weight.view(),
        .o_proj_weight = o_proj_weight.view(),
    };

    EXPECT_THROW(static_cast<void>(qwen_attention(hidden_states.view(), weights, 3, 2, 4, 1e-6f)),
                 std::invalid_argument);
}

} // namespace cppinf::tests
