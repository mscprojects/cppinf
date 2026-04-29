#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "nn/qwen_mlp.h"
#include "tensors/tensor.h"

using cppinf::nn::qwen_mlp;
using cppinf::nn::QwenMlpWeights;
using cppinf::tensors::DType;
using cppinf::tensors::Shape;
using cppinf::tensors::Tensor;
using cppinf::tensors::TensorInfo;
using cppinf::tensors::TensorView;

class QwenMlpTest : public ::testing::Test {
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

    std::vector<float> read_float_values(const TensorView& tensor_view) const {
        std::vector<float> values;
        values.reserve(tensor_view.tensor_info().shape.num_elements());
        for (std::size_t index = 0; index < tensor_view.tensor_info().shape.num_elements(); ++index) {
            float value = 0.0f;
            std::memcpy(&value, tensor_view.data().data() + index * sizeof(float), sizeof(float));
            values.push_back(value);
        }
        return values;
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

TEST_F(QwenMlpTest, GivenTorchOracleF32Inputs_WhenApplyingQwenMlp_ThenExpectedValuesAreReturned) {
    // Golden values generated with tests/nn/qwen_mlp_oracle.py.
    // Case: f32_basic.
    const auto hidden_states = make_f32_tensor(
        "hidden_states", {3, 6},
        {-1.00f, 0.71f, -0.91f, -1.26f, 1.27f, -1.03f, 1.06f, -0.99f, 1.05f, 0.56f, 0.79f, 0.46f, -0.22f,
         0.83f, -1.14f, -0.28f, 1.28f, 1.29f});
    const auto gate_proj_weight = make_f32_tensor(
        "gate_proj_weight", {10, 6},
        {-1.03f, 1.29f, -0.58f, 0.84f, -0.61f, 0.98f, -0.79f, 0.07f, 0.61f, 0.99f, -0.75f, -0.30f, 0.83f, 0.70f,
         -0.55f, 1.03f, 1.28f, 1.26f, -0.51f, -1.23f, 0.66f, 0.61f, 0.63f, -0.31f, 0.84f, -0.54f, 1.24f, -0.70f,
         -0.16f, 0.52f, 1.20f, -0.28f, 0.98f, 0.22f, -0.48f, 1.17f, 0.67f, 1.26f, 1.05f, 0.01f, -0.49f, -0.86f,
         -1.03f, -0.68f, 1.06f, -0.92f, -0.61f, 0.22f, -1.16f, 0.61f, -1.25f, 0.75f, -0.64f, 1.13f, -1.20f,
         0.54f, 1.01f, 1.15f, -0.37f, -0.22f});
    const auto up_proj_weight = make_f32_tensor(
        "up_proj_weight", {10, 6},
        {-1.00f, -0.51f, -0.97f, -1.00f, -0.20f, 0.77f, -0.43f, 0.31f, 0.56f, 1.24f, 0.86f, 0.68f, -1.26f,
         -0.12f, 0.93f, 1.18f, 0.20f, 0.09f, -0.07f, 0.42f, -0.90f, 0.27f, 1.17f, -0.32f, -1.16f, 0.95f, 0.76f,
         -1.04f, 0.58f, 0.16f, 0.27f, 0.71f, 0.67f, 0.91f, -0.99f, -0.69f, 0.57f, -0.26f, 0.36f, 0.39f, -0.51f,
         -0.05f, 0.29f, 1.04f, 1.13f, -1.01f, 0.91f, -0.86f, -0.10f, -0.72f, 0.32f, 0.38f, -0.21f, 1.10f, -0.55f,
         -0.34f, -0.36f, -0.69f, 0.27f, 1.06f});
    const auto down_proj_weight = make_f32_tensor(
        "down_proj_weight", {6, 10},
        {0.80f, -0.40f, -0.50f, -0.20f, -0.60f, -0.59f, -0.93f, -0.75f, 0.30f, 0.77f, 0.98f, 1.16f, 0.61f, 0.73f,
         0.71f, -0.26f, -1.04f, -0.50f, 0.39f, 0.21f, -0.54f, -0.17f, 1.22f, -1.13f, 0.13f, 1.27f, -1.18f, 0.80f,
         0.45f, -0.17f, -0.73f, 0.30f, -0.05f, -0.34f, -0.69f, -0.14f, 0.83f, 0.12f, -0.43f, 0.28f, -0.96f, 0.78f,
         1.25f, 0.16f, 1.00f, 0.35f, -0.33f, -0.99f, -1.13f, -0.69f, -0.94f, 1.25f, -0.46f, -0.68f, 0.77f, 0.22f,
         -1.06f, 1.22f, -0.28f, 1.13f});

    const auto weights = QwenMlpWeights{
        .gate_proj_weight = gate_proj_weight.view(),
        .up_proj_weight = up_proj_weight.view(),
        .down_proj_weight = down_proj_weight.view(),
    };

    const auto result = qwen_mlp(hidden_states.view(), weights);

    EXPECT_EQ(std::string("qwen_mlp_result"), result.tensor_info().name);
    EXPECT_EQ(DType::F32, result.tensor_info().dtype);
    EXPECT_EQ(Shape({3, 6}), result.tensor_info().shape);
    expect_float_values_near(result.view(),
                             {-0.2274054438f, -0.9172383547f, 0.5187256336f, 1.0370498896f, 0.1260067672f,
                              -0.7102236152f, 2.5894837379f, -2.0916500092f, 0.5674530864f, 2.2954623699f,
                              -2.7154958248f, -2.8996183872f, 4.0634026527f, 0.0039536608f, -4.7101669312f,
                              -2.2685987949f, -8.3464097977f, -3.2446854115f},
                             1e-5f);
}

TEST_F(QwenMlpTest, GivenMismatchedDownProjection_WhenApplyingQwenMlp_ThenItThrows) {
    const auto hidden_states = make_f32_tensor("hidden_states", {2, 6},
                                               {1.0f, 0.5f, -0.5f, 1.5f, -1.0f, 0.25f, 0.75f, -0.25f, 0.5f,
                                                -1.5f, 1.0f, 0.25f});
    const auto gate_proj_weight = make_f32_tensor(
        "gate_proj_weight", {4, 6},
        {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
    const auto up_proj_weight = gate_proj_weight;
    const auto down_proj_weight = make_f32_tensor(
        "down_proj_weight", {6, 5},
        {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    const auto weights = QwenMlpWeights{
        .gate_proj_weight = gate_proj_weight.view(),
        .up_proj_weight = up_proj_weight.view(),
        .down_proj_weight = down_proj_weight.view(),
    };

    EXPECT_THROW(static_cast<void>(qwen_mlp(hidden_states.view(), weights)), std::invalid_argument);
}
