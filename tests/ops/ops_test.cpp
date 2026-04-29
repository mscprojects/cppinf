#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ops/elementwise_ops.h"
#include "ops/matmul.h"
#include "ops/nn_ops.h"
#include "ops/tensor_ops.h"
#include "test_tensor_utils.h"

namespace cppinf::tests {

using ops::add;
using ops::cast;
using ops::matmul;
using ops::mul;
using ops::narrow;
using ops::reshape;
using ops::rms_norm;
using ops::silu;
using ops::softmax_last_dim;
using ops::squeeze;
using ops::transpose_2d;
using ops::transpose_last_two_dims;
using tensor_test_utils::expect_float_values_near;
using tensor_test_utils::make_bf16_bits_tensor;
using tensor_test_utils::make_bf16_tensor;
using tensor_test_utils::make_f32_tensor;
using tensors::DType;
using tensors::Shape;
using tensors::Tensor;
using tensors::TensorInfo;
using tensors::TensorView;

class OpsTest : public ::testing::Test {};

TEST_F(OpsTest, GivenMatchingTensors_WhenAdding_ThenElementwiseSumIsReturned) {
    const auto lhs = make_f32_tensor("lhs", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const auto rhs = make_f32_tensor("rhs", {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    const auto expected = make_f32_tensor("add_result", {2, 2}, {11.0f, 22.0f, 33.0f, 44.0f});

    EXPECT_EQ(expected, add(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenMatchingTensors_WhenMultiplying_ThenElementwiseProductIsReturned) {
    const auto lhs = make_f32_tensor("lhs", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const auto rhs = make_f32_tensor("rhs", {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    const auto expected = make_f32_tensor("mul_result", {2, 2}, {10.0f, 40.0f, 90.0f, 160.0f});

    EXPECT_EQ(expected, mul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenCompatibleMatrices_WhenMultiplying_ThenMatmulResultIsReturned) {
    const auto lhs = make_f32_tensor("lhs", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const auto rhs = make_f32_tensor("rhs", {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    const auto expected = make_f32_tensor("matmul_result", {2, 2}, {58.0f, 64.0f, 139.0f, 154.0f});

    EXPECT_EQ(expected, matmul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenCompatibleBatchedMatrices_WhenMultiplying_ThenBatchedMatmulResultIsReturned) {
    const auto lhs =
        make_f32_tensor("lhs", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, -1.0f, 0.5f, 2.0f, 3.0f, -2.0f, 1.0f});
    const auto rhs = make_f32_tensor("rhs", {2, 3, 2},
                                     {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 2.0f, -1.0f, 0.25f, 3.5f, -4.0f, 1.5f});
    const auto expected =
        make_f32_tensor("matmul_result", {2, 2, 2}, {58.0f, 64.0f, 139.0f, 154.0f, -9.875f, 5.75f, 1.5f, -8.5f});

    EXPECT_EQ(expected, matmul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenCompatibleBatchedBf16Matrices_WhenMultiplyingToF32_ThenF32MatmulResultIsReturned) {
    const auto lhs =
        make_bf16_tensor("lhs", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, -1.0f, 0.5f, 2.0f, 3.0f, -2.0f, 1.0f});
    const auto rhs = make_bf16_tensor("rhs", {2, 3, 2},
                                      {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 2.0f, -1.0f, 0.25f, 3.5f, -4.0f, 1.5f});

    const auto result = matmul(lhs.view(), rhs.view(), ops::MatmulOptions{.output_dtype = DType::F32});

    EXPECT_EQ(std::string("matmul_result"), result.tensor_info().name);
    EXPECT_EQ(DType::F32, result.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 2, 2}), result.tensor_info().shape);
    expect_float_values_near(result.view(), {58.0f, 64.0f, 139.0f, 154.0f, -9.875f, 5.75f, 1.5f, -8.5f}, 0.0f);
}

TEST_F(OpsTest, GivenF32AndBf16Matrices_WhenMultiplying_ThenF32MatmulResultIsReturned) {
    const auto lhs = make_f32_tensor("lhs", {2, 3}, {1.0f, -2.5f, 3.25f, 4.5f, 0.5f, -1.75f});
    const auto rhs = make_bf16_tensor("rhs", {3, 2}, {2.0f, -1.0f, 0.25f, 3.5f, -4.0f, 1.5f});
    const auto expected = make_f32_tensor("matmul_result", {2, 2}, {-11.625f, -4.875f, 16.125f, -5.375f});

    EXPECT_EQ(expected, matmul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenBf16AndF32Matrices_WhenMultiplying_ThenF32MatmulResultIsReturned) {
    const auto lhs = make_bf16_tensor("lhs", {2, 3}, {1.0f, -2.5f, 3.25f, 4.5f, 0.5f, -1.75f});
    const auto rhs = make_f32_tensor("rhs", {3, 2}, {2.0f, -1.0f, 0.25f, 3.5f, -4.0f, 1.5f});
    const auto expected = make_f32_tensor("matmul_result", {2, 2}, {-11.625f, -4.875f, 16.125f, -5.375f});

    EXPECT_EQ(expected, matmul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenF32Tensor_WhenCastingToBf16_ThenExpectedBitsAreReturned) {
    const auto input = make_f32_tensor("input", {2, 2}, {1.5f, -2.25f, 0.5f, -0.75f});
    const auto expected = make_bf16_bits_tensor("cast_result", {2, 2}, {0x3fc0U, 0xc010U, 0x3f00U, 0xbf40U});

    EXPECT_EQ(expected, cast(input.view(), DType::BF16));
}

TEST_F(OpsTest, GivenBf16Tensor_WhenCastingToF32_ThenExpectedValuesAreReturned) {
    const auto input = make_bf16_bits_tensor("input", {2, 2}, {0x3fc0U, 0xc010U, 0x3f00U, 0xbf40U});
    const auto expected = make_f32_tensor("cast_result", {2, 2}, {1.5f, -2.25f, 0.5f, -0.75f});

    EXPECT_EQ(expected, cast(input.view(), DType::F32));
}

TEST_F(OpsTest, GivenTensorView_WhenReshaping_ThenMetadataChangesOnly) {
    const auto input = make_f32_tensor("input", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    const auto reshaped = reshape(input.view(), Shape({4}));

    EXPECT_EQ(std::string("input"), reshaped.tensor_info().name);
    EXPECT_EQ(DType::F32, reshaped.tensor_info().dtype);
    EXPECT_EQ(Shape({4}), reshaped.tensor_info().shape);
    EXPECT_EQ(std::size_t{0}, reshaped.tensor_info().byte_offset);
    EXPECT_EQ(input.view().data().data(), reshaped.data().data());
}

TEST_F(OpsTest, GivenIncompatibleShape_WhenReshaping_ThenItThrows) {
    const auto input = make_f32_tensor("input", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    EXPECT_THROW(reshape(input.view(), Shape({3})), std::invalid_argument);
}

TEST_F(OpsTest, GivenRank2Tensor_WhenTransposing_ThenExpectedResultIsReturned) {
    const auto input = make_f32_tensor("input", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const auto expected = make_f32_tensor("transpose_2d_result", {3, 2}, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});

    EXPECT_EQ(expected, transpose_2d(input.view()));
}

TEST_F(OpsTest, GivenRank3Tensor_WhenTransposingLastTwoDims_ThenExpectedResultIsReturned) {
    const auto input =
        make_f32_tensor("input", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, -1.0f, 0.5f, 2.0f, 3.0f, -2.0f, 1.0f});
    const auto expected = make_f32_tensor("transpose_last_two_dims_result", {2, 3, 2},
                                          {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, 3.0f, 0.5f, -2.0f, 2.0f, 1.0f});

    EXPECT_EQ(expected, transpose_last_two_dims(input.view()));
}

TEST_F(OpsTest, GivenTensorView_WhenNarrowingFirstDimension_ThenSubspanIsReturned) {
    const auto input = make_f32_tensor("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    const auto narrowed = narrow(input.view(), 0, 1, 2);

    EXPECT_EQ(std::string("input"), narrowed.tensor_info().name);
    EXPECT_EQ(Shape({2, 2}), narrowed.tensor_info().shape);
    EXPECT_EQ(std::size_t{2 * sizeof(float)}, narrowed.tensor_info().byte_offset);
    EXPECT_EQ(input.view().data().data() + 2 * sizeof(float), narrowed.data().data());
    expect_float_values_near(narrowed, {3.0f, 4.0f, 5.0f, 6.0f}, 0.0f);
}

TEST_F(OpsTest, GivenTensorView_WhenNarrowingNonLeadingDimension_ThenStridedSubviewIsReturned) {
    const auto input = make_f32_tensor("input", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const auto reshaped = reshape(input.view(), Shape({2, 2, 2}));
    const auto narrowed = narrow(reshaped, 1, 1, 1);

    EXPECT_EQ(std::string("input"), narrowed.tensor_info().name);
    EXPECT_EQ(Shape({2, 1, 2}), narrowed.tensor_info().shape);
    EXPECT_EQ(std::size_t{2 * sizeof(float)}, narrowed.tensor_info().byte_offset);
    expect_float_values_near(narrowed, {3.0f, 4.0f, 7.0f, 8.0f}, 0.0f);
}

TEST_F(OpsTest, GivenUnitDimension_WhenSqueezing_ThenDimensionIsRemoved) {
    const auto input = make_f32_tensor("input", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const auto reshaped = reshape(input.view(), Shape({2, 2, 2}));
    const auto squeezed = squeeze(narrow(reshaped, 1, 1, 1), 1);

    EXPECT_EQ(Shape({2, 2}), squeezed.tensor_info().shape);
    expect_float_values_near(squeezed, {3.0f, 4.0f, 7.0f, 8.0f}, 0.0f);
}

TEST_F(OpsTest, GivenStridedTensorView_WhenReshaping_ThenItThrows) {
    const auto input = make_f32_tensor("input", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const auto reshaped = reshape(input.view(), Shape({2, 2, 2}));

    EXPECT_THROW(reshape(narrow(reshaped, 1, 1, 1), Shape({2, 2})), std::invalid_argument);
}

TEST_F(OpsTest, GivenStridedTensorView_WhenMultiplying_ThenMatmulUsesLogicalLayout) {
    const auto input = make_f32_tensor("input", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const auto rhs = make_f32_tensor("rhs", {2, 1}, {2.0f, -1.0f});
    const auto reshaped = reshape(input.view(), Shape({2, 2, 2}));
    const auto strided_lhs = squeeze(narrow(reshaped, 1, 1, 1), 1);
    const auto expected = make_f32_tensor("matmul_result", {2, 1}, {2.0f, 6.0f});

    EXPECT_EQ(expected, matmul(strided_lhs, rhs.view()));
}

TEST_F(OpsTest, GivenMatchingBf16Tensors_WhenAdding_ThenElementwiseSumIsReturned) {
    const auto lhs = make_bf16_tensor("lhs", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const auto rhs = make_bf16_tensor("rhs", {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    const auto expected = make_bf16_tensor("add_result", {2, 2}, {11.0f, 22.0f, 33.0f, 44.0f});

    EXPECT_EQ(expected, add(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenCompatibleBf16Matrices_WhenMultiplying_ThenMatmulResultIsReturned) {
    const auto lhs = make_bf16_tensor("lhs", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const auto rhs = make_bf16_tensor("rhs", {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    const auto expected = make_bf16_tensor("matmul_result", {2, 2}, {58.0f, 64.0f, 139.0f, 154.0f});

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
    const auto lhs = make_bf16_tensor("lhs", {2, 3}, {1.0f, -2.5f, 3.25f, 4.5f, 0.5f, -1.75f});
    const auto rhs = make_bf16_tensor("rhs", {3, 2}, {2.0f, -1.0f, 0.25f, 3.5f, -4.0f, 1.5f});
    const auto expected = make_bf16_tensor("matmul_result", {2, 2}, {-11.625f, -4.875f, 16.125f, -5.375f});

    EXPECT_EQ(expected, matmul(lhs.view(), rhs.view()));
}

TEST_F(OpsTest, GivenTorchOracleBf16Tensor_WhenApplyingSilu_ThenExpectedValuesAreReturned) {
    // Golden values generated offline with:
    // uv run --with torch==2.11.0 python - <<'PY'
    // import torch
    // import torch.nn.functional as F
    // x = torch.tensor([[-1.5, -0.25, 0.5, 2.0], [3.0, -4.5, 1.25, 0.75]], dtype=torch.bfloat16)
    // print(F.silu(x).float())
    // PY
    const auto input = make_bf16_tensor("input", {2, 4}, {-1.5f, -0.25f, 0.5f, 2.0f, 3.0f, -4.5f, 1.25f, 0.75f});

    const auto result = silu(input.view());

    EXPECT_EQ(std::string("silu_result"), result.tensor_info().name);
    EXPECT_EQ(DType::BF16, result.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 4}), result.tensor_info().shape);
    expect_float_values_near(
        result.view(),
        {-0.2734375f, -0.109375f, 0.310546875f, 1.7578125f, 2.859375f, -0.0495605469f, 0.97265625f, 0.5078125f}, 1e-6f);
}

TEST_F(OpsTest, GivenTorchOracleBf16Tensor_WhenApplyingSoftmaxLastDim_ThenExpectedValuesAreReturned) {
    // Golden values generated offline with:
    // uv run --with torch==2.11.0 python - <<'PY'
    // import torch
    // x = torch.tensor([[1.0, -2.5, 3.25, 0.75], [4.5, 0.5, -1.75, 2.25]], dtype=torch.bfloat16)
    // print(torch.softmax(x, dim=-1).float())
    // PY
    const auto input = make_bf16_tensor("input", {2, 4}, {1.0f, -2.5f, 3.25f, 0.75f, 4.5f, 0.5f, -1.75f, 2.25f});

    const auto result = softmax_last_dim(input.view());

    EXPECT_EQ(std::string("softmax_last_dim_result"), result.tensor_info().name);
    EXPECT_EQ(DType::BF16, result.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 4}), result.tensor_info().shape);
    expect_float_values_near(
        result.view(),
        {0.0883789062f, 0.0026702881f, 0.83984375f, 0.0688476562f, 0.88671875f, 0.0162353516f, 0.0017166138f, 0.09375f},
        1e-6f);
}

TEST_F(OpsTest, GivenTorchOracleBf16Tensor_WhenApplyingRmsNorm_ThenExpectedValuesAreReturned) {
    // Golden values generated offline with:
    // uv run --with torch==2.11.0 python - <<'PY'
    // import torch
    // x = torch.tensor([[-1.5, -0.25, 0.5, 2.0], [3.0, -4.5, 1.25, 0.75]], dtype=torch.bfloat16).float()
    // w = torch.tensor([1.0, 0.5, -1.5, 2.0], dtype=torch.bfloat16).float()
    // eps = 1e-5
    // print((x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * w).to(torch.bfloat16).float())
    // PY
    const auto input = make_bf16_tensor("input", {2, 4}, {-1.5f, -0.25f, 0.5f, 2.0f, 3.0f, -4.5f, 1.25f, 0.75f});
    const auto weight = make_bf16_tensor("weight", {4}, {1.0f, 0.5f, -1.5f, 2.0f});

    const auto result = rms_norm(input.view(), weight.view(), 1e-5f);

    EXPECT_EQ(std::string("rms_norm_result"), result.tensor_info().name);
    EXPECT_EQ(DType::BF16, result.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 4}), result.tensor_info().shape);
    expect_float_values_near(
        result.view(),
        {-1.171875f, -0.09765625f, -0.5859375f, 3.125f, 1.0703125f, -0.8046875f, -0.66796875f, 0.53515625f}, 1e-6f);
}

TEST_F(OpsTest, GivenMismatchedTensorDtypes_WhenAdding_ThenItThrows) {
    const auto lhs = make_f32_tensor("lhs", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
    const auto rhs = make_bf16_tensor("rhs", {4}, {5.0f, 6.0f, 7.0f, 8.0f});

    EXPECT_THROW(add(lhs.view(), rhs.view()), std::invalid_argument);
}

TEST_F(OpsTest, GivenMismatchedBatchDimensions_WhenMultiplying_ThenMatmulThrows) {
    const auto lhs =
        make_f32_tensor("lhs", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, -1.0f, 0.5f, 2.0f, 3.0f, -2.0f, 1.0f});
    const auto rhs = make_f32_tensor("rhs", {3, 3, 2},
                                     {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 2.0f, -1.0f, 0.25f, 3.5f, -4.0f, 1.5f,
                                      0.5f, 1.5f, -2.0f, 0.25f, 3.0f, -0.5f});

    EXPECT_THROW(matmul(lhs.view(), rhs.view()), std::invalid_argument);
}

TEST_F(OpsTest, GivenF32InputsAndBf16OutputRequest_WhenMultiplying_ThenMatmulThrows) {
    const auto lhs = make_f32_tensor("lhs", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const auto rhs = make_f32_tensor("rhs", {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    EXPECT_THROW(matmul(lhs.view(), rhs.view(), ops::MatmulOptions{.output_dtype = DType::BF16}),
                 std::invalid_argument);
}

} // namespace cppinf::tests
