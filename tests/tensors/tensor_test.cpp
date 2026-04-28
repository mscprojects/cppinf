#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tensors/bfloat16.h"
#include "tensors/tensor.h"
#include "tensors/tensor_info.h"
#include "tensors/tensor_view.h"

using cppinf::tensors::DType;
using cppinf::tensors::Shape;
using cppinf::tensors::Tensor;
using cppinf::tensors::TensorInfo;
using cppinf::tensors::TensorView;

class TensorTest : public ::testing::Test {
  protected:
    TensorInfo make_tensor_info() const {
        return TensorInfo{
            .name = "model.embed_tokens.weight",
            .dtype = DType::BF16,
            .shape = Shape({2, 4}),
            .byte_offset = 128,
        };
    }

    Tensor make_owned_tensor() const {
        std::vector<std::byte> bytes(4 * sizeof(float));
        const std::array<float, 4> values{1.0f, 2.0f, 3.0f, 4.0f};
        for (std::size_t index = 0; index < values.size(); ++index) {
            std::memcpy(bytes.data() + index * sizeof(float), &values[index], sizeof(float));
        }

        return Tensor(
            TensorInfo{
                .name = "activation",
                .dtype = DType::F32,
                .shape = Shape({2, 2}),
                .byte_offset = 0,
            },
            std::move(bytes));
    }
};

TEST_F(TensorTest, GivenBf16Type_WhenFormattingAndSizing_ThenMetadataMatches) {
    EXPECT_EQ(std::string("bf16"), std::string(cppinf::tensors::to_string(DType::BF16)));
    EXPECT_EQ(std::size_t{2}, cppinf::tensors::element_size_bytes(DType::BF16));
}

TEST_F(TensorTest, GivenFloatValue_WhenConvertingThroughBf16_ThenExpectedBitsRoundTrip) {
    const std::uint16_t bits = cppinf::tensors::float_to_bfloat16_bits(1.5f);

    EXPECT_EQ(1.5f, cppinf::tensors::bfloat16_bits_to_float(bits));
}

TEST_F(TensorTest, GivenShape_WhenQueryingRankAndElements_ThenCountsAreReturned) {
    const Shape shape({2, 3, 4});

    EXPECT_EQ(std::size_t{3}, shape.rank());
    EXPECT_EQ(std::size_t{24}, shape.num_elements());
    EXPECT_EQ(std::string("[2, 3, 4]"), cppinf::tensors::to_string(shape));
}

TEST_F(TensorTest, GivenNegativeDimension_WhenConstructingShape_ThenItThrows) {
    EXPECT_THROW(static_cast<void>(Shape({1, -1})), std::invalid_argument);
}

TEST_F(TensorTest, GivenTensorInfo_WhenFormattingMetadata_ThenStringMatches) {
    const TensorInfo tensor_info = make_tensor_info();

    EXPECT_EQ(std::size_t{16}, tensor_info.byte_size());
    EXPECT_EQ(std::string("TensorInfo(name=\"model.embed_tokens.weight\", dtype=bf16, shape=[2, 4], "
                          "offset=128, bytes=16)"),
              cppinf::tensors::to_string(tensor_info));
}

TEST_F(TensorTest, GivenMatchingBytes_WhenCreatingTensorView_ThenViewExposesMetadata) {
    const TensorInfo tensor_info = make_tensor_info();
    const std::array<std::byte, 16> bytes{};
    const TensorView tensor_view(tensor_info, bytes);

    EXPECT_EQ(std::size_t{16}, tensor_view.byte_size());
    EXPECT_EQ(std::string("TensorView(name=\"model.embed_tokens.weight\", dtype=bf16, shape=[2, 4], bytes=16)"),
              cppinf::tensors::to_string(tensor_view));
}

TEST_F(TensorTest, GivenOwnedTensor_WhenQueryingView_ThenTensorViewMatchesStorage) {
    const auto tensor = make_owned_tensor();
    const auto tensor_view = tensor.view();

    EXPECT_EQ(std::size_t{16}, tensor.byte_size());
    EXPECT_EQ(std::size_t{16}, tensor_view.byte_size());
    EXPECT_EQ(std::string("Tensor(name=\"activation\", dtype=f32, shape=[2, 2], bytes=16)"),
              cppinf::tensors::to_string(tensor));
}

TEST_F(TensorTest, GivenNonZeroOffset_WhenCreatingTensor_ThenItThrows) {
    const std::array<std::byte, 4 * sizeof(float)> bytes{};

    EXPECT_THROW(static_cast<void>(Tensor(
                     TensorInfo{
                         .name = "activation",
                         .dtype = DType::F32,
                         .shape = Shape({2, 2}),
                         .byte_offset = 16,
                     },
                     std::vector<std::byte>(bytes.begin(), bytes.end()))),
                 std::invalid_argument);
}

TEST_F(TensorTest, GivenMismatchedBytes_WhenCreatingTensorView_ThenItThrows) {
    const TensorInfo tensor_info = make_tensor_info();
    const std::array<std::byte, 8> wrong_bytes{};

    EXPECT_THROW(static_cast<void>(TensorView(tensor_info, wrong_bytes)), std::invalid_argument);
}
