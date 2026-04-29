#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "files/safetensors_file.h"
#include "test_file_utils.h"
#include "test_temp_dir.h"

namespace cppinf::tests {

using files::SafetensorsFile;
using tensors::DType;

class SafetensorsFileTest : public ::testing::Test {
  protected:
    std::filesystem::path write_temp_file(std::span<const std::byte> file_bytes) {
        const auto path = temp_dir_.path() / "weights.safetensors";
        file_test_utils::write_binary_file(path, file_bytes);
        return path;
    }

  private:
    TestTempDir temp_dir_{"cppinf-safetensors-file-test"};
};

TEST_F(SafetensorsFileTest, GivenValidFileBytes_WhenLoading_ThenMetadataAndTensorInfoAreAvailable) {
    const std::array<std::byte, 20> tensor_data{
        std::byte{0x10}, std::byte{0x11}, std::byte{0x12}, std::byte{0x13}, std::byte{0x14},
        std::byte{0x15}, std::byte{0x16}, std::byte{0x17}, std::byte{0x18}, std::byte{0x19},
        std::byte{0x1a}, std::byte{0x1b}, std::byte{0x1c}, std::byte{0x1d}, std::byte{0x1e},
        std::byte{0x1f}, std::byte{0x20}, std::byte{0x21}, std::byte{0x22}, std::byte{0x23},
    };
    const std::string header =
        R"({"__metadata__":{"format":"pt"},"embed":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},"token_ids":{"dtype":"U8","shape":[4],"data_offsets":[16,20]}})";

    const auto file = SafetensorsFile::from_bytes(file_test_utils::make_safetensors_file_bytes(header, tensor_data));

    ASSERT_EQ(std::size_t{2}, file.tensors().size());
    EXPECT_EQ(std::string("pt"), file.metadata().at("format"));
    EXPECT_TRUE(file.contains_tensor("embed"));

    const auto& embed = file.tensor_info("embed");
    EXPECT_EQ(DType::BF16, embed.dtype);
    EXPECT_EQ(std::size_t{16}, embed.byte_size());
    EXPECT_EQ(std::size_t{0}, embed.byte_offset);

    const auto& token_ids = file.tensor_info("token_ids");
    EXPECT_EQ(DType::U8, token_ids.dtype);
    EXPECT_EQ(std::size_t{16}, token_ids.byte_offset);
}

TEST_F(SafetensorsFileTest, GivenTensorName_WhenCreatingView_ThenViewPointsIntoOwnedBytes) {
    const std::array<std::byte, 4> tensor_data{
        std::byte{0x2a},
        std::byte{0x2b},
        std::byte{0x2c},
        std::byte{0x2d},
    };
    const std::string header = R"({"weights":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";

    const auto file = SafetensorsFile::from_bytes(file_test_utils::make_safetensors_file_bytes(header, tensor_data));
    const auto view = file.tensor_view("weights");

    ASSERT_EQ(std::size_t{4}, view.byte_size());
    EXPECT_EQ(std::byte{0x2a}, view.data()[0]);
    EXPECT_EQ(std::byte{0x2d}, view.data()[3]);
}

TEST_F(SafetensorsFileTest, GivenUnknownTensor_WhenQueryingTensorInfo_ThenItThrows) {
    const std::array<std::byte, 4> tensor_data{
        std::byte{0x00},
        std::byte{0x01},
        std::byte{0x02},
        std::byte{0x03},
    };
    const std::string header = R"({"weights":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";

    const auto file = SafetensorsFile::from_bytes(file_test_utils::make_safetensors_file_bytes(header, tensor_data));

    EXPECT_THROW(static_cast<void>(file.tensor_info("missing")), std::out_of_range);
}

TEST_F(SafetensorsFileTest, GivenFilePath_WhenLoading_ThenFileBytesAreParsed) {
    const std::array<std::byte, 4> tensor_data{
        std::byte{0x00},
        std::byte{0x01},
        std::byte{0x02},
        std::byte{0x03},
    };
    const std::string header = R"({"weights":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";

    const auto path = write_temp_file(file_test_utils::make_safetensors_file_bytes(header, tensor_data));
    const auto file = SafetensorsFile::from_file(path);

    EXPECT_TRUE(file.contains_tensor("weights"));
    EXPECT_EQ(DType::U8, file.tensor_info("weights").dtype);
}

TEST_F(SafetensorsFileTest, GivenInvalidTensorRange_WhenLoading_ThenItThrows) {
    const std::array<std::byte, 2> tensor_data{
        std::byte{0x00},
        std::byte{0x01},
    };
    const std::string header = R"({"weights":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";

    EXPECT_THROW(static_cast<void>(
                     SafetensorsFile::from_bytes(file_test_utils::make_safetensors_file_bytes(header, tensor_data))),
                 std::invalid_argument);
}

} // namespace cppinf::tests
