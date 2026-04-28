#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "tensors/tensor_info.h"
#include "tensors/tensor_view.h"

using cppinf::tensors::DType;
using cppinf::tensors::Shape;
using cppinf::tensors::TensorInfo;
using cppinf::tensors::TensorView;

TEST(dtype_test, formats_names) {
  EXPECT_EQ(std::string("bf16"), std::string(cppinf::tensors::to_string(DType::BF16)));
  EXPECT_EQ(std::size_t{2}, cppinf::tensors::element_size_bytes(DType::BF16));
}

TEST(shape_test, reports_rank_and_element_count) {
  const Shape shape({2, 3, 4});

  EXPECT_EQ(std::size_t{3}, shape.rank());
  EXPECT_EQ(std::size_t{24}, shape.num_elements());
  EXPECT_EQ(std::string("[2, 3, 4]"), cppinf::tensors::to_string(shape));
}

TEST(shape_test, rejects_negative_dimensions) {
  EXPECT_THROW(static_cast<void>(Shape({1, -1})), std::invalid_argument);
}

TEST(tensor_info_test, formats_metadata) {
  const TensorInfo tensor_info{
      .name = "model.embed_tokens.weight",
      .dtype = DType::BF16,
      .shape = Shape({2, 4}),
      .byte_offset = 128,
  };

  EXPECT_EQ(std::size_t{16}, tensor_info.byte_size());
  EXPECT_EQ(std::string("TensorInfo(name=\"model.embed_tokens.weight\", dtype=bf16, shape=[2, 4], "
                        "offset=128, bytes=16)"),
            cppinf::tensors::to_string(tensor_info));
}

TEST(tensor_view_test, validates_backing_bytes) {
  const TensorInfo tensor_info{
      .name = "model.embed_tokens.weight",
      .dtype = DType::BF16,
      .shape = Shape({2, 4}),
      .byte_offset = 128,
  };
  const std::array<std::byte, 16> bytes{};
  const TensorView tensor_view(tensor_info, bytes);

  EXPECT_EQ(std::size_t{16}, tensor_view.byte_size());
  EXPECT_EQ(
      std::string(
          "TensorView(name=\"model.embed_tokens.weight\", dtype=bf16, shape=[2, 4], bytes=16)"),
      cppinf::tensors::to_string(tensor_view));
}

TEST(tensor_view_test, rejects_mismatched_byte_sizes) {
  const TensorInfo tensor_info{
      .name = "model.embed_tokens.weight",
      .dtype = DType::BF16,
      .shape = Shape({2, 4}),
      .byte_offset = 128,
  };
  const std::array<std::byte, 8> wrong_bytes{};

  EXPECT_THROW(static_cast<void>(TensorView(tensor_info, wrong_bytes)), std::invalid_argument);
}
