#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ops/elementwise_ops.h"
#include "ops/matmul.h"
#include "tensors/bfloat16.h"
#include "tensors/tensor.h"

using cppinf::ops::add;
using cppinf::ops::matmul;
using cppinf::ops::mul;
using cppinf::tensors::bfloat16_bits_to_float;
using cppinf::tensors::DType;
using cppinf::tensors::float_to_bfloat16_bits;
using cppinf::tensors::Shape;
using cppinf::tensors::Tensor;
using cppinf::tensors::TensorInfo;

class OpsTest : public ::testing::Test {
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
            const std::uint16_t bits = float_to_bfloat16_bits(value);
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
};

TEST_F(OpsTest, GivenMatchingTensors_WhenAdding_ThenElementwiseSumIsReturned) {
    const Tensor lhs = make_f32_tensor("lhs", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor rhs = make_f32_tensor("rhs", {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    const Tensor expected = make_f32_tensor("add_result", {2, 2}, {11.0f, 22.0f, 33.0f, 44.0f});

    EXPECT_EQ(expected, add(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenMatchingTensors_WhenMultiplying_ThenElementwiseProductIsReturned) {
    const Tensor lhs = make_f32_tensor("lhs", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor rhs = make_f32_tensor("rhs", {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    const Tensor expected = make_f32_tensor("mul_result", {2, 2}, {10.0f, 40.0f, 90.0f, 160.0f});

    EXPECT_EQ(expected, mul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenCompatibleMatrices_WhenMultiplying_ThenMatmulResultIsReturned) {
    const Tensor lhs = make_f32_tensor("lhs", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const Tensor rhs = make_f32_tensor("rhs", {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    const Tensor expected = make_f32_tensor("matmul_result", {2, 2}, {58.0f, 64.0f, 139.0f, 154.0f});

    EXPECT_EQ(expected, matmul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenMatchingBf16Tensors_WhenAdding_ThenElementwiseSumIsReturned) {
    const Tensor lhs = make_bf16_tensor("lhs", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor rhs = make_bf16_tensor("rhs", {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    const Tensor expected = make_bf16_tensor("add_result", {2, 2}, {11.0f, 22.0f, 33.0f, 44.0f});

    EXPECT_EQ(expected, add(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenCompatibleBf16Matrices_WhenMultiplying_ThenMatmulResultIsReturned) {
    const Tensor lhs = make_bf16_tensor("lhs", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const Tensor rhs = make_bf16_tensor("rhs", {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    const Tensor expected = make_bf16_tensor("matmul_result", {2, 2}, {58.0f, 64.0f, 139.0f, 154.0f});

    EXPECT_EQ(expected, matmul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenTorchOracleBf16Matrices_WhenMultiplying_ThenExpectedResultIsReturned) {
    // Golden values generated offline with:
    // uv run --with torch==2.11.0 python - <<'PY'
    // import torch
    // lhs = torch.tensor([[1.0, -2.5, 3.25], [4.5, 0.5, -1.75]], dtype=torch.bfloat16)
    // rhs = torch.tensor([[2.0, -1.0], [0.25, 3.5], [-4.0, 1.5]], dtype=torch.bfloat16)
    // print((lhs @ rhs).float())
    // PY
    const Tensor lhs = make_bf16_tensor("lhs", {2, 3}, {1.0f, -2.5f, 3.25f, 4.5f, 0.5f, -1.75f});
    const Tensor rhs = make_bf16_tensor("rhs", {3, 2}, {2.0f, -1.0f, 0.25f, 3.5f, -4.0f, 1.5f});
    const Tensor expected = make_bf16_tensor("matmul_result", {2, 2}, {-11.625f, -4.875f, 16.125f, -5.375f});

    EXPECT_EQ(expected, matmul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenUnsupportedTensorType_WhenAdding_ThenItThrows) {
    const Tensor lhs(
        TensorInfo{
            .name = "lhs",
            .dtype = DType::U8,
            .shape = Shape({4}),
            .byte_offset = 0,
        },
        std::vector<std::byte>{std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}});
    const Tensor rhs(
        TensorInfo{
            .name = "rhs",
            .dtype = DType::U8,
            .shape = Shape({4}),
            .byte_offset = 0,
        },
        std::vector<std::byte>{std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}});

    EXPECT_THROW(static_cast<void>(add(lhs.view(), rhs.view())), std::invalid_argument);
}
